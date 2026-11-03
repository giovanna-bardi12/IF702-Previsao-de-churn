"""Optuna search over TabPFN v2 inference hyperparameters (one study per data arm).
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import optuna
import wandb

from src.metrics import METRIC_NAMES, plot_confusion_matrix_figure
from src.run_tags import wandb_provenance_tags
from tabpfn_v2.common import (
    ARMS, DEVICE, MODEL_DIR, SEED, calculate_metrics, effective_params, fit_classifier,
    ks_optimal_threshold, load_arm, positive_probabilities, release_gpu, tabpfn_version,
)


OBJECTIVE_METRIC = "ks"
N_TRIALS = 30
STORAGE = f"sqlite:///{MODEL_DIR / 'churn_tabpfn_optuna.db'}"
WANDB_ENTITY = "zovkar-ufpe"
WANDB_PROJECT = "IF702-Projeto-Final"


def study_name(arm):
    return f"tabpfn-v2-churn-{arm}-{OBJECTIVE_METRIC}"


def suggest_params(trial, arm):
    """Search space. balance_probabilities only matters when the context keeps the natural prior."""
    params = {
        "n_estimators": trial.suggest_categorical("n_estimators", [4, 8, 16, 32]),
        "softmax_temperature": trial.suggest_float("softmax_temperature", 0.5, 1.2),
        "average_before_softmax": trial.suggest_categorical("average_before_softmax", [False, True]),
    }
    if arm == "raw_natural":
        params["balance_probabilities"] = trial.suggest_categorical("balance_probabilities", [False, True])
    return params


def make_objective(arm, data):
    def objective(trial):
        params = suggest_params(trial, arm)
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=f"TabPFNv2_{arm}_trial-{trial.number}",
            group=f"tabpfn_optimization_{arm}",
            tags=[
                "model:tabpfn_v2",
                "task:churn",
                "stage:train",
                "strategy:optuna",
                f"arm:{arm}",
                f"device:{DEVICE}",
                f"objective:{OBJECTIVE_METRIC}",
                f"tabpfn:{tabpfn_version()}",
                *wandb_provenance_tags(),
            ],
            config={
                **params,
                **data.summary(),
                "data_description": data.description,
                "device": DEVICE,
                "implementation": "tabpfn native, ModelVersion.V2",
            },
            dir=str(MODEL_DIR),
            reinit=True,
        )
        classifier = None
        try:
            classifier, fit_seconds = fit_classifier(params, data, seed=SEED)
            probabilities, inference_seconds = positive_probabilities(classifier, data.X_val)
            metrics = calculate_metrics(data.y_val, probabilities)
            ks_threshold = ks_optimal_threshold(data.y_val, probabilities)

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
                "effective_params": json.dumps(effective_params(classifier)),
            })

            trial.set_user_attr("params", params)
            trial.set_user_attr("arm", arm)
            trial.set_user_attr("input_dim", data.n_features)
            trial.set_user_attr("fit_time_seconds", fit_seconds)
            trial.set_user_attr("inference_time_seconds", inference_seconds)
            trial.set_user_attr("val_ks_threshold", ks_threshold)
            for key in METRIC_NAMES:
                trial.set_user_attr(f"val_{key}", float(metrics[key]))
            return float(metrics[OBJECTIVE_METRIC])
        finally:
            del classifier
            release_gpu()
            wandb.finish()

    return objective


def print_top_trials(study, top_k=5):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_k]
    print(f"\n================ TOP {len(best_trials)} TABPFN v2 MODELS (validation) ================\n")
    for rank, trial in enumerate(best_trials, 1):
        print(f"--- Top {rank} (Trial #{trial.number}) ---")
        print(f"Validation KS: {trial.value:.4f}")
        print(f"Params: {trial.user_attrs['params']}")
        print(f"Fit time: {trial.user_attrs['fit_time_seconds']:.2f} sec | "
              f"inference time: {trial.user_attrs['inference_time_seconds']:.2f} sec")
        for key in METRIC_NAMES:
            print(f"Val {key}: {trial.user_attrs[f'val_{key}']:.4f}")
        print()


def main(arm, n_trials=N_TRIALS):
    data = load_arm(arm)
    study = optuna.create_study(
        study_name=study_name(arm),
        storage=STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        load_if_exists=True,
    )
    study.optimize(make_objective(arm, data), n_trials=n_trials)
    print_top_trials(study)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, default="raw")
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    main(args.arm, args.trials)
