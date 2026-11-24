"""Evaluate the selected MITRA predictor on the held-out test split."""
import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import wandb
from autogluon.tabular import TabularPredictor

from src.metrics import (METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure,
                         plot_ks_figure, plot_roc_figure)
from src.run_tags import wandb_provenance_tags
from mitra.common import DATA_MODE, DEVICE, EXPERIMENT_ID, MODEL_DIR, VARIATION, load_splits, positive_probabilities
from mitra.mitra_train import STUDY_NAME, STUDY_STORAGE_PATH, study_storage
from mitra.mitra_train_baseline import CONFIG_PATH, ZERO_SHOT_CONFIG_PATH


RESULTS_DIR = MODEL_DIR / "test_results" / EXPERIMENT_ID


def evaluate(path, test_df, val_threshold):
    if not Path(path).exists():
        raise FileNotFoundError(f"Predictor not found at {path}. Restore the corresponding W&B "
                                "model artifact to this path, or retrain the model.")
    predictor = TabularPredictor.load(str(path))
    start = time.perf_counter()
    probabilities = positive_probabilities(predictor, test_df)
    inference_seconds = time.perf_counter() - start
    labels = test_df["Churn"].to_numpy(dtype=int)
    metrics = calculate_metrics(labels, probabilities)
    threshold_metrics = calculate_metrics(labels, probabilities, threshold=val_threshold)
    return inference_seconds, metrics, threshold_metrics, labels, probabilities


def report(title, val_ks, inference_seconds, metrics, threshold_metrics, val_threshold, labels, probabilities, tag):
    print(f"================ MITRA TEST: {title} ================")
    print(f"Validation KS (selection): {val_ks:.4f}")
    print(f"Test inference time: {inference_seconds:.4f} sec for full test split")
    for key in METRIC_NAMES:
        print(f"Test {key}: {metrics[key]:.4f}")
    print("Confusion matrix:\n", metrics["confusion_matrix"], "\n")
    print(f"Validation-selected threshold: {val_threshold:.6f}")
    for key in ("accuracy", "precision", "recall", "f1"):
        print(f"Test {key} at validation threshold: {threshold_metrics[key]:.4f}")

    fig = plot_confusion_matrix_figure(metrics["confusion_matrix"])
    fig.savefig(RESULTS_DIR / f"confusion_matrix_{tag}.png", bbox_inches="tight")
    plt.close(fig)
    images = [RESULTS_DIR / f"confusion_matrix_{tag}.png"]
    for name, plotter in (("ks", plot_ks_figure), ("roc", plot_roc_figure)):
        fig = plotter(labels, probabilities)
        path = RESULTS_DIR / f"{name}_{tag}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        images.append(path)
    return images


def make_row(model_name, rank, trial, val_ks, inference_seconds, metrics, params, val_threshold, threshold_metrics):
    return {
        "model": model_name,
        "rank": rank,
        "trial": trial,
        "val_ks": val_ks,
        **{f"test_{key}": float(metrics[key]) for key in METRIC_NAMES},
        "test_inference_time_seconds_full_split": inference_seconds,
        "confusion_matrix": json.dumps(metrics["confusion_matrix"].tolist()),
        "validation_ks_threshold": val_threshold,
        **{f"test_at_validation_threshold_{key}": float(threshold_metrics[key])
           for key in ("accuracy", "precision", "recall", "f1")},
        "confusion_matrix_at_validation_threshold": json.dumps(threshold_metrics["confusion_matrix"].tolist()),
        "hyperparameters": json.dumps(params),
    }


def test_baseline(test_df, zero_shot=False):
    config_path = ZERO_SHOT_CONFIG_PATH if zero_shot else CONFIG_PATH
    if not config_path.exists():
        suffix = " --zero-shot" if zero_shot else ""
        raise RuntimeError(f"MITRA checkpoint not found. Run: python -m mitra.mitra_train_baseline{suffix}")
    config = json.loads(config_path.read_text())
    predictor_path = MODEL_DIR / config["predictor_path"]
    val_threshold = config["val_ks_threshold"]
    inference_seconds, metrics, threshold_metrics, labels, probabilities = evaluate(predictor_path, test_df, val_threshold)
    val_ks = config["val_metrics"]["ks"]
    tag = "zero_shot" if zero_shot else "baseline"
    images = report(tag, val_ks, inference_seconds, metrics, threshold_metrics, val_threshold, labels, probabilities, tag)
    row = make_row(f"MITRA-{tag}", 1, tag, val_ks, inference_seconds, metrics, config["params"], val_threshold, threshold_metrics)
    return [row], RESULTS_DIR / f"mitra_{tag}_test_metrics.csv", images


def test_optuna(test_df):
    missing_study_message = (
        f"MITRA Optuna study {STUDY_NAME!r} was not found in {STUDY_STORAGE_PATH}. "
        "Run: python -m mitra.mitra_train. "
        "To evaluate the baseline instead, run: python -m mitra.mitra_test --baseline. "
        "If training ran in another session, restore its SQLite database and predictor files."
    )
    if not STUDY_STORAGE_PATH.exists():
        raise RuntimeError(missing_study_message)
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=study_storage())
    except KeyError as exc:
        raise RuntimeError(missing_study_message) from exc
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("No completed MITRA trials. Run: python -m mitra.mitra_train")
    trials = sorted(complete, key=lambda trial: trial.value, reverse=True)[:1]

    rows = []
    for rank, trial in enumerate(trials, 1):
        predictor_path = MODEL_DIR / trial.user_attrs["predictor_path"]
        val_threshold = trial.user_attrs["val_ks_threshold"]
        inference_seconds, metrics, threshold_metrics, labels, probabilities = evaluate(predictor_path, test_df, val_threshold)
        images = report(f"Top {rank} (Trial #{trial.number})", trial.value, inference_seconds, metrics,
                        threshold_metrics, val_threshold, labels, probabilities, f"trial{trial.number}")
        rows.append(make_row("MITRA-optuna", rank, trial.number, trial.value,
                             inference_seconds, metrics, trial.user_attrs["params"], val_threshold, threshold_metrics))
    return rows, RESULTS_DIR / "mitra_test_metrics.csv", images


def main(baseline=False, zero_shot=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _, _, test_df = load_splits()
    rows, output_path, image_paths = test_baseline(test_df, zero_shot) if baseline or zero_shot else test_optuna(test_df)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    run = wandb.init(
        entity="zovkar-ufpe",
        project="IF702-Projeto-Final",
        name="MITRA_test_zero_shot" if zero_shot else "MITRA_test_baseline" if baseline else "MITRA_test_best_trial",
        group="mitra_test",
        tags=[
            "model:mitra",
            "task:churn",
            "stage:test",
            "strategy:zero-shot" if zero_shot else "strategy:baseline" if baseline else "strategy:optuna",
            f"data:{VARIATION}",
            f"device:{DEVICE}",
            "evaluation:fixed-checkpoint" if baseline or zero_shot else "evaluation:best-validation",
            *wandb_provenance_tags(),
        ],
        config={"baseline": baseline, "zero_shot": zero_shot, "data_mode": DATA_MODE, "variation": VARIATION},
        dir=str(MODEL_DIR),
    )
    try:
        for row in rows:
            wandb.log({key: value for key, value in row.items()
                       if key.startswith("test_") and key != "test_confusion_matrix"})
        wandb.save(str(output_path), base_path=str(MODEL_DIR))
        for image_path in image_paths:
            wandb.save(str(image_path), base_path=str(MODEL_DIR))
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--baseline", action="store_true", help="Evaluate the fixed fine-tuning baseline")
    group.add_argument("--zero-shot", action="store_true", help="Evaluate the standalone zero-shot model")
    args = parser.parse_args()
    main(baseline=args.baseline, zero_shot=args.zero_shot)
