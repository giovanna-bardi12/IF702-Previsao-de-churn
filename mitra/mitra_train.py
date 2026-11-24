"""Optuna fine-tuning search for Amazon/AutoGluon MITRA.

Run from the repository root with ``python -m mitra.mitra_train``.
"""
import argparse
from contextlib import closing
import fcntl
import json
import os
import secrets
import shutil
import sqlite3
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import optuna
import wandb

from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure, validation_ks_threshold
from src.run_tags import wandb_provenance_tags
from mitra.common import (
    DATA_MODE, DEVICE, EXPERIMENT_ID, MAX_FINE_TUNE_EPOCHS, MODEL_DIR, PATIENCE, VARIATION,
    evaluate_predictor, fit_predictor, load_splits,
    positive_probabilities,
)


OBJECTIVE_METRIC = "ks"
FIT_TIME_LIMIT_SECONDS = None
STORAGE_KIND = os.environ.get("MITRA_OPTUNA_STORAGE", "sqlite")
if STORAGE_KIND not in ("sqlite", "journal"):
    raise ValueError("MITRA_OPTUNA_STORAGE must be 'sqlite' or 'journal'")
ESTIMATOR_MODE = os.environ.get("MITRA_ESTIMATORS", "1,2")
if ESTIMATOR_MODE not in ("1", "1,2"):
    raise ValueError("MITRA_ESTIMATORS must be '1' or '1,2'")
ESTIMATORS = [1] if ESTIMATOR_MODE == "1" else [1, 2]
STUDY_NAME = (f"mitra-churn-v8-{EXPERIMENT_ID}-{OBJECTIVE_METRIC}"
              f"{'-single-estimator' if ESTIMATOR_MODE == '1' else ''}"
              f"{'-journal' if STORAGE_KIND == 'journal' else ''}")
DEFAULT_TRIALS = 20
STUDY_DB_PATH = MODEL_DIR / "churn_mitra_optuna.db"
STORAGE = f"sqlite:///{STUDY_DB_PATH}"
STUDY_STORAGE_PATH = (MODEL_DIR / f"{STUDY_NAME}.journal") if STORAGE_KIND == "journal" else STUDY_DB_PATH
WANDB_ENTITY = "zovkar-ufpe"
WANDB_PROJECT = "IF702-Projeto-Final"


def _relative(path):
    return os.path.relpath(path, MODEL_DIR)


def journal_classes():
    """Support Optuna's old and new public journal APIs."""
    try:
        from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    except ImportError:
        from optuna.storages import JournalFileStorage as JournalFileBackend, JournalFileOpenLock
    return JournalFileBackend, JournalFileOpenLock


def study_storage():
    if STORAGE_KIND == "sqlite":
        return STORAGE
    backend_class, lock_class = journal_classes()
    lock = lock_class(str(STUDY_STORAGE_PATH))
    return optuna.storages.JournalStorage(backend_class(str(STUDY_STORAGE_PATH), lock_obj=lock))


def suggest_model_params(trial):
    """Sample fine-tuned models; zero-shot is run separately."""
    params = {
        "fine_tune": True,
        "n_estimators": trial.suggest_categorical("n_estimators", ESTIMATORS),
    }
    params.update({
        "fine_tune_steps": MAX_FINE_TUNE_EPOCHS,
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "warmup_steps": trial.suggest_categorical("warmup_steps", [5, 10]),
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
            "data_mode": DATA_MODE,
            "input_dim": input_dim,
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "device": DEVICE,
            "implementation": "AutoGluon MITRA",
        },
        dir=str(MODEL_DIR),
    )
    predictor_path = MODEL_DIR / "predictors" / STUDY_NAME / f"trial-{trial.number}"
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
        val_threshold = validation_ks_threshold(val_df["Churn"].to_numpy(dtype=int), probabilities)
        train_metrics = evaluate_predictor(predictor, train_df)
        model_names = predictor.model_names()

        wandb.log({
            "fit_time_seconds": fit_seconds,
            "val_inference_time_seconds": inference_seconds,
            "train_mse": float(train_metrics["mse"]),
            "train_binary_cross_entropy": float(train_metrics["binary_cross_entropy"]),
            "val_ks_threshold": val_threshold,
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
        checkpoint_artifact = wandb.Artifact(f"{STUDY_NAME}-trial-{trial.number}", type="model")
        checkpoint_artifact.add_dir(str(predictor_path))
        run.log_artifact(checkpoint_artifact)

        trial.set_user_attr("predictor_path", _relative(predictor_path))
        trial.set_user_attr("params", model_params)
        trial.set_user_attr("val_ks_threshold", val_threshold)
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


def backup_study(study, trial):
    """Snapshot SQLite after the trial is committed; upload without risking the search."""
    backup_dir = MODEL_DIR / "study_backups"
    extension = "journal" if STORAGE_KIND == "journal" else "db"
    destination = backup_dir / f"{STUDY_NAME}-trial-{trial.number}.{extension}"
    temporary = backup_dir / f"{STUDY_NAME}-trial-{trial.number}.tmp.{extension}"
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        if STORAGE_KIND == "journal":
            _, lock_class = journal_classes()
            lock = lock_class(str(STUDY_STORAGE_PATH))
            lock.acquire()
            try:
                shutil.copyfile(STUDY_STORAGE_PATH, temporary)
            finally:
                lock.release()
        else:
            with closing(sqlite3.connect(STUDY_DB_PATH)) as source, closing(sqlite3.connect(temporary)) as target:
                source.backup(target)
        os.replace(temporary, destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        warnings.warn(f"Could not back up Optuna study: {exc}")
        return
    run = None
    try:
        run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
                         name=f"{STUDY_NAME}-backup", group="mitra_study_backups",
                         job_type="study-backup", dir=str(MODEL_DIR),
                         config={"study_name": STUDY_NAME, "trial": trial.number})
        artifact = wandb.Artifact(f"{STUDY_NAME}-study-trial-{trial.number}", type="optuna-study")
        artifact.add_file(str(destination), name=STUDY_STORAGE_PATH.name)
        run.log_artifact(artifact)
    except Exception as exc:
        warnings.warn(f"Local study backup saved at {destination}, but W&B upload failed: {exc}")
    finally:
        if run is not None:
            try:
                run.finish()
            except Exception as exc:
                warnings.warn(f"Could not finish W&B study backup run: {exc}")


def main(n_trials=DEFAULT_TRIALS, worker_trials=None):
    if n_trials < 1:
        raise ValueError("--n-trials must be at least 1")
    if worker_trials is not None and (worker_trials < 1 or STORAGE_KIND != "journal"):
        raise ValueError("--worker-trials requires a positive count and MITRA_OPTUNA_STORAGE=journal")
    # Serialize initialization and the single enqueued baseline across processes.
    with (MODEL_DIR / ".mitra_study_init.lock").open("w") as init_lock:
        fcntl.flock(init_lock, fcntl.LOCK_EX)
        try:
            study = optuna.create_study(study_name=STUDY_NAME, storage=study_storage(),
                                        direction="maximize", load_if_exists=True)
            existing = study.trials
            if not existing:
                study.enqueue_trial({
                    "n_estimators": 1, "lr": 1e-4, "warmup_steps": 10,
                    "shuffle_features": False, "shuffle_classes": False,
                })
        finally:
            fcntl.flock(init_lock, fcntl.LOCK_UN)
    seed = secrets.randbits(32) if worker_trials is not None else 42 + len(existing)
    study.sampler = optuna.samplers.TPESampler(seed=seed, constant_liar=True)
    completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = worker_trials if worker_trials is not None else max(0, n_trials - completed)
    print(f"Optuna study {STUDY_NAME}: {completed} completed; "
          f"running up to {remaining} {'worker-assigned' if worker_trials is not None else 'remaining'} trial(s).", flush=True)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, callbacks=[backup_study])
    print_top_trials(study)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--n-trials", type=int, default=DEFAULT_TRIALS,
                        help="Target number of completed trials (default: %(default)s)")
    budget.add_argument("--worker-trials", type=int,
                        help="Number of trial attempts for this journal-backed worker")
    args = parser.parse_args()
    main(n_trials=args.n_trials, worker_trials=args.worker_trials)
