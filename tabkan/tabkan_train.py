import os
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.data_loader import get_dataloaders
from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure

# Importa o wrapper e os loops diretamente de models_kan.py
from models_tabkan import TabKANWrapper, train_epoch, validate_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIATION = "missing_encode_normalize"
OBJECTIVE_METRIC = "ks"
MAX_EPOCHS = 1000
PATIENCE = 15
LOSS_MIN_DELTA = 1e-4
KS_MIN_DELTA = 1e-3
N_TRIALS = 100

STUDY_NAME = f"tabkan-mixer-churn-{OBJECTIVE_METRIC}"
STORAGE = f"sqlite:///{MODEL_DIR / 'churn_tabkan_optuna.db'}"


def _rel(path):
    return os.path.relpath(path, MODEL_DIR)


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    num_layers = trial.suggest_categorical("num_layers", [1, 2, 3])
    token_dim = trial.suggest_categorical("token_dim", [8, 16, 32, 64])
    channel_dim = trial.suggest_categorical("channel_dim", [16, 32, 64, 128])
    token_order = trial.suggest_categorical("token_order", [2, 3, 4])
    channel_order = trial.suggest_categorical("channel_order", [2, 3, 4])

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    input_dim = train_loader.dataset.tensors[0].shape[1]

    # Instancia o wrapper importado do models_kan.py
    model = TabKANWrapper(
        num_features=input_dim,
        num_layers=num_layers,
        token_dim=token_dim,
        channel_dim=channel_dim,
        token_order=token_order,
        channel_order=channel_order,
    ).to(device)

    run = wandb.init(
        entity="zovkar-ufpe",
        project="IF702-Projeto-Final",
        name=f"TabKAN_trial-{trial.number}_lr-{lr:.4f}_bs-{batch_size}",
        group="tabkan_optimization",
        config={**trial.params, "variation": VARIATION, "input_dim": input_dim},
        dir=str(MODEL_DIR),
        reinit=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss_path = os.path.join(run.dir, f"best_loss_tabkan_trial_{trial.number}.pth")
    best_ks_path = os.path.join(run.dir, f"best_ks_tabkan_trial_{trial.number}.pth")

    epochs_without_improvement = 0
    inference_times = []
    best_val_loss = float("inf")
    best_ks = float("-inf")
    best_loss_epoch = 0
    best_ks_epoch = 0
    best_loss_metrics, best_ks_metrics = None, None

    for epoch in range(MAX_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, inference_time, labels, probs = validate_epoch(model, val_loader, criterion, device)
        inference_times.append(inference_time)

        metrics = calculate_metrics(labels, probs)

        loss_improved = val_loss < best_val_loss - LOSS_MIN_DELTA
        ks_improved = metrics["ks"] > best_ks + KS_MIN_DELTA

        if loss_improved:
            best_val_loss = val_loss
            best_loss_epoch = epoch + 1
            best_loss_metrics = metrics
            torch.save(model.state_dict(), best_loss_path)

        if ks_improved:
            best_ks = metrics["ks"]
            best_ks_epoch = epoch + 1
            best_ks_metrics = metrics
            torch.save(model.state_dict(), best_ks_path)

        if loss_improved or ks_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": metrics[k] for k in METRIC_NAMES},
            "val_inference_time_batch": inference_time,
            "epochs_without_improvement": epochs_without_improvement,
        })

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping na época {epoch + 1} "
                  f"(melhor loss: época {best_loss_epoch} | melhor KS: época {best_ks_epoch})")
            break

    fig = plot_confusion_matrix_figure(best_ks_metrics["confusion_matrix"])
    wandb.log({"confusion_matrix_img": wandb.Image(fig)})
    plt.close(fig)

    wandb.summary.update({
        "best_loss_epoch": best_loss_epoch,
        "best_ks_epoch": best_ks_epoch,
        **{f"best_ks_model/val_{k}": best_ks_metrics[k] for k in METRIC_NAMES},
        **{f"best_loss_model/val_{k}": best_loss_metrics[k] for k in METRIC_NAMES},
    })
    wandb.save(best_loss_path, base_path=run.dir)
    wandb.save(best_ks_path, base_path=run.dir)

    trial.set_user_attr("avg_inference_time", sum(inference_times) / len(inference_times))
    trial.set_user_attr("model_path", _rel(best_ks_path))
    trial.set_user_attr("best_loss_model_path", _rel(best_loss_path))
    trial.set_user_attr("input_dim", input_dim)
    trial.set_user_attr("best_ks_epoch", best_ks_epoch)
    trial.set_user_attr("best_loss_epoch", best_loss_epoch)
    for k in METRIC_NAMES:
        trial.set_user_attr(f"val_{k}", float(best_ks_metrics[k]))

    wandb.finish()
    return best_ks


if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

    print(f"\n================ TOP {len(top_5)} MELHORES MODELOS TABKAN (validação) ================\n")
    for rank, trial in enumerate(top_5, 1):
        print(f"--- Top {rank} (Trial #{trial.number}) ---")
        print(f"Melhor KS na época {trial.user_attrs['best_ks_epoch']} | "
              f"menor val_loss na época {trial.user_attrs['best_loss_epoch']}")
        for k in METRIC_NAMES:
            print(f"Val {k}: {trial.user_attrs[f'val_{k}']:.4f}")
        print(f"Average Inference Time: {trial.user_attrs['avg_inference_time']:.6f} sec/batch")
        print("Hyperparameters:")
        for param, val in trial.params.items():
            print(f"  - {param}: {val}")
        print("-" * 50 + "\n")