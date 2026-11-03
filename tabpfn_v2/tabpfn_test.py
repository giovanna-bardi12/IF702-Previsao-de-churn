"""Evaluate the TabPFN v2 baseline or the top Optuna trials on the held-out test split.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import wandb
from sklearn.metrics import roc_auc_score

from src.metrics import METRIC_NAMES, plot_confusion_matrix_figure
from src.run_tags import wandb_provenance_tags
from tabpfn_v2.common import (
    ARMS, DEVICE, MODEL_DIR, SEED, bootstrap_ci, calculate_metrics, fit_classifier,
    ks_optimal_threshold, ks_statistic, load_arm, positive_probabilities, release_gpu,
    tabpfn_version,
)
from tabpfn_v2.tabpfn_train import STORAGE, study_name
from tabpfn_v2.tabpfn_train_baseline import config_path


RESULTS_DIR = MODEL_DIR / "test_results"
THRESHOLD_METRICS = ["accuracy", "precision", "recall", "f1"]
WANDB_ENTITY = "zovkar-ufpe"
WANDB_PROJECT = "IF702-Projeto-Final"


def selected_configs(arm, baseline, top_k):
    """Configurations to evaluate: the fixed baseline, or the top-k validation-KS Optuna trials."""
    if baseline:
        path = config_path(arm)
        if not path.exists():
            raise RuntimeError(
                f"Baseline do TabPFN não encontrado. Rode: python -m tabpfn_v2.tabpfn_train_baseline --arm {arm}"
            )
        config = json.loads(path.read_text())
        return [{
            "model": f"TabPFNv2-baseline-{arm}", "rank": 1, "trial": "baseline",
            "val_ks": config["val_metrics"]["ks"], "params": config["params"],
        }]

    study = optuna.load_study(study_name=study_name(arm), storage=STORAGE)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError(f"Nenhum trial concluído. Rode: python -m tabpfn_v2.tabpfn_train --arm {arm}")
    best = sorted(complete, key=lambda t: t.value, reverse=True)[:top_k]
    return [{
        "model": f"TabPFNv2-optuna-{arm}", "rank": rank, "trial": trial.number,
        "val_ks": trial.value, "params": trial.user_attrs["params"],
    } for rank, trial in enumerate(best, 1)]


def evaluate_config(params, data, seed, refit_trainval=False):
    """Fit on train, pick the KS threshold on validation, predict test (optionally refit on train+val)."""
    classifier, fit_seconds = fit_classifier(params, data, seed=seed)
    val_probabilities, _ = positive_probabilities(classifier, data.X_val)
    ks_threshold = ks_optimal_threshold(data.y_val, val_probabilities)
    val_ks = float(ks_statistic(data.y_val, val_probabilities))

    if refit_trainval:
        X = np.vstack([data.X_train, data.X_val])
        y = np.concatenate([data.y_train, data.y_val])
        classifier, fit_seconds = fit_classifier(params, data, seed=seed, X=X, y=y)

    probabilities, inference_seconds = positive_probabilities(classifier, data.X_test)
    del classifier
    release_gpu()
    return {
        "probabilities": probabilities,
        "metrics": calculate_metrics(data.y_test, probabilities),
        "metrics_kst": calculate_metrics(data.y_test, probabilities, threshold=ks_threshold),
        "ks_threshold": ks_threshold,
        "val_ks_this_seed": val_ks,
        "fit_seconds": fit_seconds,
        "inference_seconds": inference_seconds,
    }


def make_row(model, arm, rank, trial, seed, val_ks, result, params, y_test):
    metrics, metrics_kst = result["metrics"], result["metrics_kst"]
    ks_low, ks_high = bootstrap_ci(y_test, result["probabilities"], ks_statistic)
    auc_low, auc_high = bootstrap_ci(y_test, result["probabilities"], roc_auc_score)
    return {
        "model": model,
        "arm": arm,
        "rank": rank,
        "trial": trial,
        "seed": seed,
        "val_ks": val_ks,
        "val_ks_this_seed": result["val_ks_this_seed"],
        "val_ks_threshold": result["ks_threshold"],
        **{f"test_{key}": float(metrics[key]) for key in METRIC_NAMES},
        "test_ks_ci_low": ks_low,
        "test_ks_ci_high": ks_high,
        "test_roc_auc_ci_low": auc_low,
        "test_roc_auc_ci_high": auc_high,
        **{f"test_kst_{key}": float(metrics_kst[key]) for key in THRESHOLD_METRICS},
        "fit_time_seconds": result["fit_seconds"],
        "test_inference_time_seconds_full_split": result["inference_seconds"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"].tolist()),
        "confusion_matrix_kst": json.dumps(metrics_kst["confusion_matrix"].tolist()),
        "hyperparameters": json.dumps(params),
    }


def report(row):
    print(f"================ TABPFN v2 TEST: {row['model']} (rank {row['rank']}, seed {row['seed']}) ================")
    print(f"Validation KS (selection): {row['val_ks']:.4f}")
    print(f"Fit time: {row['fit_time_seconds']:.2f} sec | "
          f"test inference: {row['test_inference_time_seconds_full_split']:.4f} sec for full test split")
    for key in METRIC_NAMES:
        print(f"Test {key}: {row[f'test_{key}']:.4f}")
    print(f"Test KS 95% CI: [{row['test_ks_ci_low']:.4f}, {row['test_ks_ci_high']:.4f}]")
    print(f"KS-optimal threshold (from validation): {row['val_ks_threshold']:.4f}")
    for key in THRESHOLD_METRICS:
        print(f"Test {key} @KS-threshold: {row[f'test_kst_{key}']:.4f}")
    print("Confusion matrix @0.5:\n", np.array(json.loads(row["confusion_matrix"])), "\n")


def save_confusion_matrices(row, tag):
    for suffix, column in (("", "confusion_matrix"), ("_kst", "confusion_matrix_kst")):
        fig = plot_confusion_matrix_figure(np.array(json.loads(row[column])))
        fig.savefig(RESULTS_DIR / f"confusion_matrix_{tag}{suffix}.png", bbox_inches="tight")
        plt.close(fig)


def summarize(rows):
    """Mean/std over seeds for every numeric test metric, per configuration."""
    frame = pd.DataFrame(rows)
    keys = ["model", "arm", "rank", "trial"]
    numeric = [c for c in frame.columns if c.startswith("test_") or c in ("fit_time_seconds", "val_ks_threshold")]
    summary = frame.groupby(keys, sort=False)[numeric].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_seeds", frame.groupby(keys, sort=False).size())
    return summary.reset_index()


def main(arm, top_k=1, baseline=False, n_seeds=5, refit_trainval=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_arm(arm)
    configs = selected_configs(arm, baseline, top_k)
    seeds = [SEED + i for i in range(n_seeds)]

    variants = [("", False)] + ([("-trainval", True)] if refit_trainval else [])
    rows = []
    for config in configs:
        for suffix, refit in variants:
            model = config["model"] + suffix
            for index, seed in enumerate(seeds):
                result = evaluate_config(config["params"], data, seed, refit_trainval=refit)
                row = make_row(model, arm, config["rank"], config["trial"], seed,
                               config["val_ks"], result, config["params"], data.y_test)
                rows.append(row)
                report(row)
                if index == 0:
                    save_confusion_matrices(row, f"{model}_rank{config['rank']}")

    stem = f"tabpfn_{arm}_baseline_test_metrics" if baseline else f"tabpfn_{arm}_test_metrics"
    output_path = RESULTS_DIR / f"{stem}.csv"
    summary_path = RESULTS_DIR / f"{stem}_summary.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    summary = summarize(rows)
    summary.to_csv(summary_path, index=False)
    print(f"Per-seed results saved to {output_path}")
    print(f"Mean/std over {n_seeds} seeds saved to {summary_path}")
    print(summary[["model", "rank", "test_ks_mean", "test_ks_std", "test_roc_auc_mean", "test_roc_auc_std"]]
          .to_string(index=False))

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"TabPFNv2_test_{'baseline' if baseline else 'top_trials'}_{arm}",
        group="tabpfn_test",
        tags=[
            "model:tabpfn_v2",
            "task:churn",
            "stage:test",
            "strategy:baseline" if baseline else "strategy:optuna",
            f"arm:{arm}",
            f"device:{DEVICE}",
            f"tabpfn:{tabpfn_version()}",
            "evaluation:fixed-checkpoint" if baseline else "evaluation:top-k",
            *wandb_provenance_tags(),
        ],
        config={"arm": arm, "baseline": baseline, "top_k": 1 if baseline else top_k,
                "n_seeds": n_seeds, "refit_trainval": refit_trainval, **data.summary()},
        dir=str(MODEL_DIR),
        reinit=True,
    )
    try:
        for row in rows:
            wandb.log({key: value for key, value in row.items()
                       if key.startswith("test_") and not isinstance(value, str)})
        for record in summary.to_dict("records"):
            prefix = f"{record['model']}_rank{record['rank']}"
            wandb.summary.update({f"{prefix}/{key}": value for key, value in record.items()
                                  if key.startswith("test_") and key.endswith(("_mean", "_std"))})
        wandb.save(str(output_path), base_path=str(MODEL_DIR))
        wandb.save(str(summary_path), base_path=str(MODEL_DIR))
        for image_path in RESULTS_DIR.glob("confusion_matrix_*.png"):
            wandb.save(str(image_path), base_path=str(MODEL_DIR))
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="raw")
    parser.add_argument("--top-k", type=int, default=1, help="Number of highest validation-KS trials to evaluate")
    parser.add_argument("--baseline", action="store_true", help="Evaluate the default-configuration baseline")
    parser.add_argument("--n-seeds", type=int, default=5, help="TabPFN random seeds per configuration")
    parser.add_argument("--refit-trainval", action="store_true",
                        help="Also evaluate a refit on train+val context (extra row, different protocol)")
    args = parser.parse_args()
    main(arm=args.arm, top_k=args.top_k, baseline=args.baseline,
         n_seeds=args.n_seeds, refit_trainval=args.refit_trainval)
