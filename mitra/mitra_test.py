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

from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure
from src.run_tags import wandb_provenance_tags
from mitra.common import DEVICE, MODEL_DIR, VARIATION, load_splits, positive_probabilities
from mitra.mitra_train import STORAGE, STUDY_DB_PATH, STUDY_NAME
from mitra.mitra_train_baseline import CONFIG_PATH


RESULTS_DIR = MODEL_DIR / "test_results_natural_v2"


def evaluate(path, test_df):
    predictor = TabularPredictor.load(str(path))
    start = time.perf_counter()
    probabilities = positive_probabilities(predictor, test_df)
    inference_seconds = time.perf_counter() - start
    metrics = calculate_metrics(test_df["Churn"].to_numpy(dtype=int), probabilities)
    return inference_seconds, metrics


def report(title, val_ks, inference_seconds, metrics, tag):
    print(f"================ MITRA TEST: {title} ================")
    print(f"Validation KS (selection): {val_ks:.4f}")
    print(f"Test inference time: {inference_seconds:.4f} sec for full test split")
    for key in METRIC_NAMES:
        print(f"Test {key}: {metrics[key]:.4f}")
    print("Confusion matrix:\n", metrics["confusion_matrix"], "\n")

    fig = plot_confusion_matrix_figure(metrics["confusion_matrix"])
    fig.savefig(RESULTS_DIR / f"confusion_matrix_{tag}.png", bbox_inches="tight")
    plt.close(fig)


def make_row(model_name, rank, trial, val_ks, inference_seconds, metrics, params):
    return {
        "model": model_name,
        "rank": rank,
        "trial": trial,
        "val_ks": val_ks,
        **{f"test_{key}": float(metrics[key]) for key in METRIC_NAMES},
        "test_inference_time_seconds_full_split": inference_seconds,
        "confusion_matrix": json.dumps(metrics["confusion_matrix"].tolist()),
        "hyperparameters": json.dumps(params),
    }


def test_baseline(test_df):
    if not CONFIG_PATH.exists():
        raise RuntimeError("MITRA baseline not found. Run: python -m mitra.mitra_train_baseline")
    config = json.loads(CONFIG_PATH.read_text())
    predictor_path = MODEL_DIR / config["predictor_path"]
    inference_seconds, metrics = evaluate(predictor_path, test_df)
    val_ks = config["val_metrics"]["ks"]
    report("baseline", val_ks, inference_seconds, metrics, "baseline")
    row = make_row("MITRA-baseline", 1, "baseline", val_ks, inference_seconds, metrics, config["params"])
    return [row], RESULTS_DIR / "mitra_baseline_test_metrics.csv"


def test_optuna(test_df):
    missing_study_message = (
        f"MITRA Optuna study {STUDY_NAME!r} was not found in {STUDY_DB_PATH}. "
        "Run: python -m mitra.mitra_train. "
        "To evaluate the baseline instead, run: python -m mitra.mitra_test --baseline. "
        "If training ran in another session, restore its SQLite database and predictor files."
    )
    if not STUDY_DB_PATH.exists():
        raise RuntimeError(missing_study_message)
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    except KeyError as exc:
        raise RuntimeError(missing_study_message) from exc
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("No completed MITRA trials. Run: python -m mitra.mitra_train")
    trials = sorted(complete, key=lambda trial: trial.value, reverse=True)[:1]

    rows = []
    for rank, trial in enumerate(trials, 1):
        predictor_path = MODEL_DIR / trial.user_attrs["predictor_path"]
        inference_seconds, metrics = evaluate(predictor_path, test_df)
        report(f"Top {rank} (Trial #{trial.number})", trial.value, inference_seconds, metrics,
               f"trial{trial.number}")
        rows.append(make_row("MITRA-optuna", rank, trial.number, trial.value,
                             inference_seconds, metrics, trial.user_attrs["params"]))
    return rows, RESULTS_DIR / "mitra_test_metrics.csv"


def main(baseline=False):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _, _, test_df = load_splits()
    rows, output_path = test_baseline(test_df) if baseline else test_optuna(test_df)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    run = wandb.init(
        entity="zovkar-ufpe",
        project="IF702-Projeto-Final",
        name="MITRA_test_baseline" if baseline else "MITRA_test_best_trial",
        group="mitra_test",
        tags=[
            "model:mitra",
            "task:churn",
            "stage:test",
            "strategy:baseline" if baseline else "strategy:optuna",
            f"data:{VARIATION}",
            f"device:{DEVICE}",
            "evaluation:fixed-checkpoint" if baseline else "evaluation:best-validation",
            *wandb_provenance_tags(),
        ],
        config={"baseline": baseline},
        dir=str(MODEL_DIR),
    )
    try:
        for row in rows:
            wandb.log({key: value for key, value in row.items()
                       if key.startswith("test_") and key != "test_confusion_matrix"})
        wandb.save(str(output_path), base_path=str(MODEL_DIR))
        for image_path in RESULTS_DIR.glob("confusion_matrix_*.png"):
            wandb.save(str(image_path), base_path=str(MODEL_DIR))
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true", help="Evaluate the fixed fine-tuning baseline")
    args = parser.parse_args()
    main(baseline=args.baseline)
