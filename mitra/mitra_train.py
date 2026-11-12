"""Optuna fine-tuning search for Amazon/AutoGluon MITRA.

Run from the repository root with ``python -m mitra.mitra_train``.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import optuna
import wandb

from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure
from src.run_tags import wandb_provenance_tags
from mitra.common import (
    DEVICE, MAX_FINE_TUNE_EPOCHS, MODEL_DIR, PATIENCE, VARIATION,
    evaluate_predictor, fit_predictor, load_splits,
    positive_probabilities,
)


OBJECTIVE_METRIC = "ks"
FINE_TUNE_STEPS = [0, MAX_FINE_TUNE_EPOCHS]
FIT_TIME_LIMIT_SECONDS = None
STUDY_NAME = f"mitra-churn-natural-v6-{OBJECTIVE_METRIC}"
DEFAULT_TRIALS = 20
STUDY_DB_PATH = MODEL_DIR / "churn_mitra_optuna.db"
STORAGE = f"sqlite:///{STUDY_DB_PATH}"
WANDB_ENTITY = "zovkar-ufpe"
WANDB_PROJECT = "IF702-Projeto-Final"


def _relative(path):
    return os.path.relpath(path, MODEL_DIR)


def suggest_model_params(trial):
    """Sample MITRA settings; fine-tuning-only settings are conditional."""
    fine_tune_steps = trial.suggest_categorical("fine_tune_steps", FINE_TUNE_STEPS)
    params = {
        "fine_tune": fine_tune_steps > 0,
        "n_estimators": trial.suggest_categorical("n_estimators", [1, 2]),
    }
    if fine_tune_steps:
        params.update({
            "fine_tune_steps": fine_tune_steps,
            "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "warmup_steps": trial.suggest_categorical("warmup_steps", [10, 50, 100, 1000]),
            "patience": PATIENCE,
            "shuffle_features": trial.suggest_categorical("shuffle_features", [False, True]),
            "shuffle_classes": trial.suggest_categorical("shuffle_classes", [False, True]),
        })
    return params


def objective(trial):
    model_params = suggest_model_params(trial)
    fine_tune = model_params["fine_tune"]
    fine_tune_steps = model_params.get("fine_tune_steps", 0)

    train_df, val_df, _ = load_splits()
    input_dim = len(train_df.columns) - 1
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"MITRA_trial-{trial.number}_ft-{fine_tune}_steps-{fine_tune_steps}",
        group="mitra_optimization",
        tags=[
            "model:mitra",
            "task:churn",
            "stage:train",
            "strategy:optuna",
            f"data:{VARIATION}",
            f"device:{DEVICE}",
            f"objective:{OBJECTIVE_METRIC}",
            f"fine-tune:{'on' if fine_tune else 'off'}",
            f"fine-tune-steps:{fine_tune_steps}",
            *wandb_provenance_tags(),
        ],
        config={
            **model_params,
            "variation": VARIATION,
            "input_dim": input_dim,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "device": DEVICE,
            "implementation": "AutoGluon MITRA",
        },
        dir=str(MODEL_DIR),
    )
    # Use the active run directory so each trial keeps a self-contained predictor.
    predictor_path = Path(run.dir) / "mitra_predictor"
    try:
        fit_started = time.perf_counter()
        predictor = fit_predictor(
            train_df, predictor_path, model_params,
            time_limit=FIT_TIME_LIMIT_SECONDS,
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = positive_probabilities(predictor, val_df)
        inference_seconds = time.perf_counter() - prediction_started
        metrics = calculate_metrics(val_df["Churn"].to_numpy(dtype=int), probabilities)
        train_metrics = evaluate_predictor(predictor, train_df)
        model_names = predictor.model_names()

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
            "best_val_ks": float(metrics["ks"]),
            "fit_time_seconds": fit_seconds,
            "val_inference_time_seconds": inference_seconds,
            "model_names": json.dumps(model_names),
        })
        checkpoint_artifact = wandb.Artifact(f"mitra-trial-{trial.number}", type="model")
        checkpoint_artifact.add_dir(str(predictor_path))
        run.log_artifact(checkpoint_artifact)

        trial.set_user_attr("predictor_path", _relative(predictor_path))
        trial.set_user_attr("params", model_params)
        trial.set_user_attr("input_dim", input_dim)
        trial.set_user_attr("fit_time_seconds", fit_seconds)
        trial.set_user_attr("inference_time_seconds", inference_seconds)
        for key in ("mse", "binary_cross_entropy"):
            trial.set_user_attr(f"train_{key}", float(train_metrics[key]))
        for key in METRIC_NAMES:
            trial.set_user_attr(f"val_{key}", float(metrics[key]))
        return float(metrics[OBJECTIVE_METRIC])
    finally:
        wandb.finish()


def print_top_trials(study, top_k=5):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_k]
    print(f"\n================ TOP {len(best_trials)} MITRA MODELS (validation) ================\n")
    for rank, trial in enumerate(best_trials, 1):
        print(f"--- Top {rank} (Trial #{trial.number}) ---")
        print(f"Validation KS: {trial.value:.4f}")
        print(f"Parameters: {json.dumps(trial.user_attrs['params'], sort_keys=True)}")
        print(f"Fit time: {trial.user_attrs['fit_time_seconds']:.2f} sec")
        for key in ("mse", "binary_cross_entropy"):
            print(f"Train {key}: {trial.user_attrs[f'train_{key}']:.4f}")
        for key in METRIC_NAMES:
            print(f"Val {key}: {trial.user_attrs[f'val_{key}']:.4f}")
        print(f"Predictor: {trial.user_attrs['predictor_path']}\n")


def main(n_trials=DEFAULT_TRIALS):
    if n_trials < 1:
        raise ValueError("--n-trials must be at least 1")
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        load_if_exists=True,
    )
    existing = study.trials
    # A fresh sampler seed on resume avoids replaying the same startup samples.
    study.sampler = optuna.samplers.TPESampler(seed=42 + len(existing))
    if not existing:
        study.enqueue_trial({"fine_tune_steps": 0, "n_estimators": 1})
        study.enqueue_trial({
            "fine_tune_steps": MAX_FINE_TUNE_EPOCHS, "n_estimators": 1, "lr": 1e-4,
            "warmup_steps": 1000,
            "shuffle_features": False, "shuffle_classes": False,
        })
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, n_trials - completed)
    print(f"Optuna study {STUDY_NAME}: {completed} completed, target {n_trials}, "
          f"running up to {remaining} more trial(s).", flush=True)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    print_top_trials(study)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS,
                        help="Target number of completed trials (default: %(default)s)")
    args = parser.parse_args()
    main(n_trials=args.n_trials)
