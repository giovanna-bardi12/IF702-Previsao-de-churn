"""Train the fixed-configuration MITRA baseline and persist its predictor."""
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import wandb

from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure
from src.run_tags import wandb_provenance_tags
from mitra.common import (
    DEVICE, MAX_FINE_TUNE_EPOCHS, MODEL_DIR, PATIENCE, VARIATION,
    evaluate_predictor, fit_predictor, load_splits,
    positive_probabilities,
)


BASELINE_PARAMS = {"fine_tune": True, "fine_tune_steps": MAX_FINE_TUNE_EPOCHS, "patience": PATIENCE}
FIT_TIME_LIMIT_SECONDS = None
BASELINE_DIR = MODEL_DIR / "baseline_natural_v2"
CONFIG_PATH = BASELINE_DIR / "config.json"


def main():
    train_df, val_df, _ = load_splits()
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity="zovkar-ufpe",
        project="IF702-Projeto-Final",
        name="MITRA_baseline",
        group="mitra_baseline",
        tags=[
            "model:mitra",
            "task:churn",
            "stage:train",
            "strategy:baseline",
            f"data:{VARIATION}",
            f"device:{DEVICE}",
            "fine-tune:on",
            f"fine-tune-steps:{BASELINE_PARAMS['fine_tune_steps']}",
            *wandb_provenance_tags(),
        ],
        config={
            **BASELINE_PARAMS,
            "variation": VARIATION,
            "input_dim": len(train_df.columns) - 1,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "device": DEVICE,
            "implementation": "AutoGluon MITRA",
        },
        dir=str(MODEL_DIR),
    )
    try:
        predictor_path = Path(run.dir) / "mitra_predictor"
        fit_started = time.perf_counter()
        predictor = fit_predictor(
            train_df, predictor_path, BASELINE_PARAMS,
            time_limit=FIT_TIME_LIMIT_SECONDS,
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = positive_probabilities(predictor, val_df)
        inference_seconds = time.perf_counter() - prediction_started
        metrics = calculate_metrics(val_df["Churn"].to_numpy(dtype=int), probabilities)
        train_metrics = evaluate_predictor(predictor, train_df)

        wandb.log({
            "fit_time_seconds": fit_seconds,
            "val_inference_time_seconds": inference_seconds,
            "train_mse": float(train_metrics["mse"]),
            "train_binary_cross_entropy": float(train_metrics["binary_cross_entropy"]),
            **{f"val_{key}": float(metrics[key]) for key in METRIC_NAMES},
        })
        fig = plot_confusion_matrix_figure(metrics["confusion_matrix"])
        wandb.log({"val_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
        wandb.summary.update({
            "fit_time_seconds": fit_seconds,
            "val_inference_time_seconds": inference_seconds,
            "best_val_ks": float(metrics["ks"]),
        })
        checkpoint_artifact = wandb.Artifact("mitra-baseline", type="model")
        checkpoint_artifact.add_dir(str(predictor_path))
        run.log_artifact(checkpoint_artifact)

        config = {
            "params": BASELINE_PARAMS,
            "variation": VARIATION,
            "input_dim": len(train_df.columns) - 1,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "predictor_path": str(predictor_path.relative_to(MODEL_DIR)),
            "fit_time_seconds": fit_seconds,
            "inference_time_seconds": inference_seconds,
            "train_metrics": {key: float(train_metrics[key]) for key in ("mse", "binary_cross_entropy")},
            "val_metrics": {key: float(metrics[key]) for key in METRIC_NAMES},
        }
        CONFIG_PATH.write_text(json.dumps(config, indent=2))

        print("================ MITRA BASELINE (validation) ================")
        print(f"Requested fine-tuning steps: {BASELINE_PARAMS['fine_tune_steps']}")
        print(f"Fit time: {fit_seconds:.2f} sec")
        print(f"Validation inference time: {inference_seconds:.4f} sec for {len(val_df)} rows")
        for key in ("mse", "binary_cross_entropy"):
            print(f"Train {key}: {train_metrics[key]:.4f}")
        for key in METRIC_NAMES:
            print(f"Val {key}: {metrics[key]:.4f}")
        print(f"Predictor saved at {predictor_path}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
