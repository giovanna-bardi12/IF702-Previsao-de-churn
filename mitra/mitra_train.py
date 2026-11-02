"""Optuna fine-tuning search for Amazon/AutoGluon MITRA.

Run from the repository root with ``python -m mitra.mitra_train``.
"""
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
    DEVICE, MODEL_DIR, VARIATION, fit_predictor, load_splits,
    positive_probabilities,
)


OBJECTIVE_METRIC = "ks"
N_TRIALS = 30
FIT_TIME_LIMIT_SECONDS = 900
STUDY_NAME = f"mitra-churn-{OBJECTIVE_METRIC}"
STORAGE = f"sqlite:///{MODEL_DIR / 'churn_mitra_optuna.db'}"
WANDB_ENTITY = "zovkar-ufpe"
WANDB_PROJECT = "IF702-Projeto-Final"


def _relative(path):
    return os.path.relpath(path, MODEL_DIR)


def objective(trial):
    fine_tune = trial.suggest_categorical("fine_tune", [False, True])
    fine_tune_steps = trial.suggest_categorical("fine_tune_steps", [10, 25, 50, 100]) if fine_tune else 0
    search_params = {
        "fine_tune": fine_tune,
        "fine_tune_steps": fine_tune_steps,
    }
    model_params = {"fine_tune": fine_tune}
    if fine_tune:
        model_params["fine_tune_steps"] = fine_tune_steps

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
            **search_params,
            "variation": VARIATION,
            "input_dim": input_dim,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "device": DEVICE,
            "implementation": "AutoGluon MITRA",
        },
        dir=str(MODEL_DIR),
        reinit=True,
    )
    # Use the active run directory so each trial keeps a self-contained predictor.
    predictor_path = Path(run.dir) / "mitra_predictor"
    try:
        fit_started = time.perf_counter()
        predictor = fit_predictor(
            train_df, val_df, predictor_path, model_params,
            time_limit=FIT_TIME_LIMIT_SECONDS,
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        probabilities = positive_probabilities(predictor, val_df)
        inference_seconds = time.perf_counter() - prediction_started
        metrics = calculate_metrics(val_df["Churn"].to_numpy(dtype=int), probabilities)
        model_names = predictor.model_names()

        wandb.log({
            "fit_time_seconds": fit_seconds,
            "val_inference_time_seconds": inference_seconds,
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
        trial.set_user_attr("params", search_params)
        trial.set_user_attr("input_dim", input_dim)
        trial.set_user_attr("fit_time_seconds", fit_seconds)
        trial.set_user_attr("inference_time_seconds", inference_seconds)
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
        print(f"Fine-tuning: {trial.params['fine_tune']} | steps: {trial.user_attrs['params']['fine_tune_steps']}")
        print(f"Fit time: {trial.user_attrs['fit_time_seconds']:.2f} sec")
        for key in METRIC_NAMES:
            print(f"Val {key}: {trial.user_attrs[f'val_{key}']:.4f}")
        print(f"Predictor: {trial.user_attrs['predictor_path']}\n")


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)
    print_top_trials(study)
