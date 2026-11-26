"""
Uso (na raiz do projeto):
    python tabkan/tabkan_baseline.py

Treina o TabKAN com hiperparâmetros fixos (sem Optuna), usando as mesmas
configurações de treino do tabkan_train.py (early stopping duplo, patience,
min_delta, loss), para a comparação com o modelo otimizado ser justa.
Salva checkpoints e config.json em tabkan/baseline/.
"""
import json
import random
import sys
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent      # .../IF702-Previsao-de-churn/tabkan
PROJECT_ROOT = MODEL_DIR.parent                  # .../IF702-Previsao-de-churn
sys.path.insert(0, str(PROJECT_ROOT))


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.data_loader import get_dataloaders
from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure
from models_tabkan import TabKANWrapper, train_epoch, validate_epoch
# Mesmas configurações de treino do Optuna, para a comparação ser justa
from tabkan_train import KS_MIN_DELTA, LOSS_MIN_DELTA, MAX_EPOCHS, PATIENCE, VARIATION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparâmetros fixos: valores padrão do TabKANWrapper / Adam
# (1 camada mixer, token_dim 16, channel_dim 32, polinômios de ordem 3,
#  lr 1e-3, sem weight decay)
BASELINE_PARAMS = {
    "lr": 1e-3,
    "batch_size": 64,
    "weight_decay": 0.0,
    "num_layers": 1,
    "token_dim": 16,
    "channel_dim": 32,
    "token_order": 3,
    "channel_order": 3,
}
SEED = 42

BASELINE_DIR = MODEL_DIR / "baseline"
BEST_KS_PATH = BASELINE_DIR / "best_ks.pth"
BEST_LOSS_PATH = BASELINE_DIR / "best_loss.pth"
CONFIG_PATH = BASELINE_DIR / "config.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed(SEED)
    BASELINE_DIR.mkdir(exist_ok=True)
    p = BASELINE_PARAMS

    train_loader, val_loader, _ = get_dataloaders(batch_size=p["batch_size"])
    input_dim = train_loader.dataset.tensors[0].shape[1]

    model = TabKANWrapper(
        num_features=input_dim,
        num_layers=p["num_layers"],
        token_dim=p["token_dim"],
        channel_dim=p["channel_dim"],
        token_order=p["token_order"],
        channel_order=p["channel_order"],
    ).to(device)
    n_params = sum(t.numel() for t in model.parameters())

    run = wandb.init(
        entity="zovkar-ufpe",
        project="IF702-Projeto-Final",
        name="TabKAN_baseline",
        group="tabkan_baseline",
        config={**p, "variation": VARIATION, "input_dim": input_dim, "seed": SEED, "n_params": n_params},
        dir=str(MODEL_DIR),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])

    # Early stopping duplo: o contador só zera se a val_loss OU o KS melhorar.
    # Cada métrica guarda o seu próprio checkpoint.
    max_epochs = MAX_EPOCHS
    patience = PATIENCE
    loss_min_delta = LOSS_MIN_DELTA
    ks_min_delta = KS_MIN_DELTA
    epochs_without_improvement = 0
    inference_times = []
    best_val_loss = float("inf")
    best_ks = float("-inf")
    best_loss_epoch = 0
    best_ks_epoch = 0
    best_loss_metrics, best_ks_metrics = None, None

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, inference_time, labels, probs = validate_epoch(model, val_loader, criterion, device)
        inference_times.append(inference_time)

        metrics = calculate_metrics(labels, probs)

        loss_improved = val_loss < best_val_loss - loss_min_delta
        ks_improved = metrics["ks"] > best_ks + ks_min_delta

        if loss_improved:
            best_val_loss = val_loss
            best_loss_epoch = epoch + 1
            best_loss_metrics = metrics
            torch.save(model.state_dict(), BEST_LOSS_PATH)

        if ks_improved:
            best_ks = metrics["ks"]
            best_ks_epoch = epoch + 1
            best_ks_metrics = metrics
            torch.save(model.state_dict(), BEST_KS_PATH)

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

        if epochs_without_improvement >= patience:
            print(f"Early stopping na época {epoch + 1} "
                  f"(melhor loss: época {best_loss_epoch} | melhor KS: época {best_ks_epoch})")
            break

    epochs_run = epoch + 1
    avg_inference_time = sum(inference_times) / len(inference_times)

    # Matriz de confusão do modelo de melhor KS (métrica principal)
    fig = plot_confusion_matrix_figure(best_ks_metrics["confusion_matrix"])
    wandb.log({"confusion_matrix_img": wandb.Image(fig)})
    plt.close(fig)

    wandb.summary.update({
        "best_loss_epoch": best_loss_epoch,
        "best_ks_epoch": best_ks_epoch,
        **{f"best_ks_model/val_{k}": best_ks_metrics[k] for k in METRIC_NAMES},
        **{f"best_loss_model/val_{k}": best_loss_metrics[k] for k in METRIC_NAMES},
    })
    wandb.save(str(BEST_KS_PATH), base_path=str(MODEL_DIR))
    wandb.save(str(BEST_LOSS_PATH), base_path=str(MODEL_DIR))

    # config.json: tudo que o tabkan_test.py precisa para reconstruir e avaliar o baseline
    config = {
        "params": p,
        "seed": SEED,
        "variation": VARIATION,
        "input_dim": input_dim,
        "n_params": n_params,
        "epochs_run": epochs_run,
        "best_ks_epoch": best_ks_epoch,
        "best_loss_epoch": best_loss_epoch,
        "avg_inference_time": avg_inference_time,
        "val_best_ks_model": {k: float(best_ks_metrics[k]) for k in METRIC_NAMES},
        "val_best_loss_model": {k: float(best_loss_metrics[k]) for k in METRIC_NAMES},
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))

    print("\n================ TABKAN BASELINE (validação) ================")
    print(f"Parâmetros treináveis: {n_params}")
    print(f"Épocas: {epochs_run} | melhor KS na época {best_ks_epoch} | "
          f"menor val_loss na época {best_loss_epoch}")
    for k in METRIC_NAMES:
        print(f"Val {k}: {best_ks_metrics[k]:.4f}")
    print(f"Average Inference Time: {avg_inference_time:.6f} sec/batch")
    print(f"\nCheckpoints e config salvos em {BASELINE_DIR}")

    wandb.finish()


if __name__ == "__main__":
    main()