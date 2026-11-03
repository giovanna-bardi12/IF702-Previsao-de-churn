"""Fit the default TabPFN v2 configuration and evaluate it on the validation split.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import wandb

from src.metrics import METRIC_NAMES, plot_confusion_matrix_figure
from src.run_tags import wandb_provenance_tags
from tabpfn_v2.common import (
    ARMS, DEVICE, MODEL_DIR, calculate_metrics, effective_params, fit_classifier,
    ks_optimal_threshold, load_arm, positive_probabilities, tabpfn_version,
)


BASELINE_PARAMS = {}  # library defaults of the V2 checkpoint; effective values are logged below
BASELINE_DIR = MODEL_DIR / "baseline"
WANDB_ENTITY = "zovkar-ufpe"
WANDB_PROJECT = "IF702-Projeto-Final"


def config_path(arm):
    return BASELINE_DIR / f"config_{arm}.json"


def main(arm):
    data = load_arm(arm)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"TabPFNv2_baseline_{arm}",
        group="tabpfn_baseline",
        tags=[
            "model:tabpfn_v2",
            "task:churn",
            "stage:train",
            "strategy:baseline",
            f"arm:{arm}",
            f"device:{DEVICE}",
            f"tabpfn:{tabpfn_version()}",
            *wandb_provenance_tags(),
        ],
        config={
            **data.summary(),
            "data_description": data.description,
            "params": BASELINE_PARAMS,
            "device": DEVICE,
            "implementation": "tabpfn native, ModelVersion.V2",
        },
        dir=str(MODEL_DIR),
        reinit=True,
    )
    try:
        classifier, fit_seconds = fit_classifier(BASELINE_PARAMS, data)
        probabilities, inference_seconds = positive_probabilities(classifier, data.X_val)
        metrics = calculate_metrics(data.y_val, probabilities)
        ks_threshold = ks_optimal_threshold(data.y_val, probabilities)
        used = effective_params(classifier)

        wandb.log({
            "fit_time_seconds": fit_seconds,
            "val_inference_time_seconds": inference_seconds,
            "val_ks_threshold": ks_threshold,
            **{f"val_{key}": float(metrics[key]) for key in METRIC_NAMES},
        })
        fig = plot_confusion_matrix_figure(metrics["confusion_matrix"])
        wandb.log({"val_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        wandb.summary.update({
            "best_val_ks": float(metrics["ks"]),
            "fit_time_seconds": fit_seconds,
            "val_inference_time_seconds": inference_seconds,
            "effective_params": json.dumps(used),
        })

        config = {
            "arm": arm,
            "params": BASELINE_PARAMS,
            "effective_params": used,
            "tabpfn_version": tabpfn_version(),
            "data": data.summary(),
            "fit_time_seconds": fit_seconds,
            "inference_time_seconds": inference_seconds,
            "val_ks_threshold": ks_threshold,
            "val_metrics": {key: float(metrics[key]) for key in METRIC_NAMES},
        }
        config_path(arm).write_text(json.dumps(config, indent=2))

        print(f"================ TABPFN v2 BASELINE [{arm}] (validation) ================")
        print(f"Data: {data.description}")
        print(f"Effective params: {used}")
        print(f"Fit time: {fit_seconds:.2f} sec")
        print(f"Validation inference time: {inference_seconds:.4f} sec for {len(data.y_val)} rows")
        for key in METRIC_NAMES:
            print(f"Val {key}: {metrics[key]:.4f}")
        print(f"KS-optimal threshold (validation): {ks_threshold:.4f}")
        print(f"Config saved at {config_path(arm)}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="raw")
    main(parser.parse_args().arm)
