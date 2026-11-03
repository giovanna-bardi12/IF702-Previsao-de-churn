import os
import random
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR.parent))

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from torch import nn
import wandb

from models_stab import build_stab, train_epoch, validate_epoch
from src.data_loader import get_dataloaders
from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure
from src.run_tags import wandb_provenance_tags

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VARIATION = "missing_encode_normalize"
MIN_EPOCHS, MAX_EPOCHS, PATIENCE, N_TRIALS = 50, 70, 15, 30
LOSS_MIN_DELTA, KS_MIN_DELTA = 1e-4, 1e-3
WANDB_ENTITY, WANDB_PROJECT = "gmcb", "IF702-Projeto-Final"
STUDY_NAME = "stab-churn-ks"
STORAGE = f"sqlite:///{MODEL_DIR / 'churn_stab_optuna.db'}"
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_run(params, strategy, trial=None):
    set_seed()
    train_loader, val_loader, _ = get_dataloaders(batch_size=params["batch_size"])
    input_dim = train_loader.dataset.tensors[0].shape[1]
    d_model = params["d_model"]
    model = build_stab(
        n_features=input_dim, d_model=d_model, n_heads=params["n_heads"],
        n_layers=params["n_layers"], ffn_multiplier=4, gate_hidden=4 * d_model,
        classifier_hidden=2 * d_model, dropout=params["dropout"],
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    run = wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT,
        name=f"STab_trial-{trial.number}" if trial is not None else "STab_baseline",
        group=f"stab_{strategy}", dir=str(MODEL_DIR), reinit=True,
        tags=["model:stab", "task:churn", "stage:train", f"strategy:{strategy}",
              f"data:{VARIATION}", f"device:{DEVICE.type}", "objective:ks", *wandb_provenance_tags()],
        config={**params, "variation": VARIATION, "input_dim": input_dim,
                "n_params": n_params, "seed": SEED},
    )
    checkpoint_dir = MODEL_DIR / "baseline" if trial is None else Path(run.dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: checkpoint_dir / f"best_{name}.pth" for name in ("loss", "ks")}
    best_loss, best_ks = float("inf"), float("-inf")
    best_epochs, best_metrics = {}, {}
    stale, times = 0, []
    try:
        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_loss, inference_time, labels, probs = validate_epoch(model, val_loader, criterion, DEVICE)
            metrics = calculate_metrics(labels, probs)
            times.append(inference_time)
            loss_improved = val_loss < best_loss - LOSS_MIN_DELTA
            ks_improved = metrics["ks"] > best_ks + KS_MIN_DELTA
            if loss_improved:
                best_loss = val_loss
                best_epochs["loss"] = epoch
                best_metrics["loss"] = metrics
                torch.save(model.state_dict(), paths["loss"])
            if ks_improved:
                best_ks = metrics["ks"]
                best_epochs["ks"] = epoch
                best_metrics["ks"] = metrics
                torch.save(model.state_dict(), paths["ks"])
            stale = 0 if loss_improved or ks_improved else stale + 1
            run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     **{f"val_{k}": float(metrics[k]) for k in METRIC_NAMES},
                     "val_inference_time_batch": inference_time, "epochs_without_improvement": stale})
            if trial is not None:
                trial.report(float(metrics["ks"]), epoch)
                if epoch >= MIN_EPOCHS and trial.should_prune():
                    raise optuna.TrialPruned()
            if epoch >= MIN_EPOCHS and stale >= PATIENCE:
                print(f"Early stopping na época {epoch}; melhor KS na {best_epochs['ks']}")
                break

        fig = plot_confusion_matrix_figure(best_metrics["ks"]["confusion_matrix"])
        run.log({"confusion_matrix_img": wandb.Image(fig)})
        plt.close(fig)
        summary = {"best_loss_epoch": best_epochs["loss"], "best_ks_epoch": best_epochs["ks"],
                   "avg_inference_time": float(np.mean(times)), "n_params": n_params}
        for name in ("loss", "ks"):
            summary.update({f"best_{name}_model/val_{k}": float(best_metrics[name][k]) for k in METRIC_NAMES})
        run.summary.update(summary)
        for path in paths.values():
            wandb.save(str(path), base_path=str(MODEL_DIR))
        return {"params": params, "input_dim": input_dim, "n_params": n_params,
                "epochs_run": epoch, "best_epochs": best_epochs,
                "avg_inference_time": float(np.mean(times)),
                "val_best_ks_model": {k: float(best_metrics["ks"][k]) for k in METRIC_NAMES},
                "val_best_loss_model": {k: float(best_metrics["loss"][k]) for k in METRIC_NAMES},
                "model_paths": {k: os.path.relpath(v, MODEL_DIR) for k, v in paths.items()}}
    finally:
        run.finish()


def objective(trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "d_model": trial.suggest_categorical("d_model", [8, 16, 32]),
        "n_heads": trial.suggest_categorical("n_heads", [1, 2, 4]),
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
    }
    result = train_run(params, "optimization", trial)
    for key in ("input_dim", "n_params", "avg_inference_time", "best_epochs", "model_paths"):
        trial.set_user_attr(key, result[key])
    trial.set_user_attr("val_best_loss_model", result["val_best_loss_model"])
    for key, value in result["val_best_ks_model"].items():
        trial.set_user_attr(f"val_{key}", value)
    return result["val_best_ks_model"]["ks"]


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED), load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)
    completed = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
                       key=lambda t: t.value, reverse=True)
    for rank, trial in enumerate(completed[:5], 1):
        print(f"Top {rank}: trial {trial.number}, KS={trial.value:.4f}, "
              f"checkpoint={trial.user_attrs['model_paths']['ks']}, params={trial.params}")
