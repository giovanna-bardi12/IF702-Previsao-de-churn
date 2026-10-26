import os
import sys
from pathlib import Path


# kan/ fica dentro do projeto: coloca a raiz no sys.path para o "from src..." funcionar
MODEL_DIR = Path(__file__).resolve().parent      # .../IF702-Previsao-de-churn/kan
PROJECT_ROOT = MODEL_DIR.parent                  # .../IF702-Previsao-de-churn
sys.path.insert(0, str(PROJECT_ROOT))


import matplotlib.pyplot as plt
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from src.data_loader import get_dataloaders
from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure
from models_kan import build_kan, train_epoch, validate_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIATION = "missing_encode_normalize"  # qual conjunto de CSVs do prepare_data usar
OBJECTIVE_METRIC = "ks"                 # métrica principal (validação) que o Optuna maximiza
MAX_EPOCHS = 1000
PATIENCE = 15
LOSS_MIN_DELTA = 1e-4                   # melhora mínima na val_loss (BCE)
KS_MIN_DELTA = 1e-3                     # melhora mínima no KS
N_TRIALS = 100

STUDY_NAME = f"kan-churn-{OBJECTIVE_METRIC}"

STORAGE = f"sqlite:///{MODEL_DIR / 'churn_optuna.db'}"

def _rel(path):
    """Caminho relativo à raiz do projeto. Assim o banco do Optuna continua válido
    se o projeto for copiado para outra pasta (ex.: $TMPDIR do cluster -> home)."""
    return os.path.relpath(path, MODEL_DIR)
 

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    num_hidden_layers = trial.suggest_categorical("num_hidden_layers", [1, 2])
    hidden_dim = trial.suggest_categorical("hidden_dim", [4, 8, 16, 32])
    grid_size = trial.suggest_categorical("grid_size", [3, 5, 8])         # nº de intervalos da spline
    spline_order = trial.suggest_categorical("spline_order", [2, 3])      # grau da spline (3 = cúbica)
    activation_name = trial.suggest_categorical("activation", ["SiLU", "ReLU", "GELU"])  # função base

    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    input_dim = train_loader.dataset.tensors[0].shape[1]

    model = build_kan(
        input_dim=input_dim,
        num_hidden_layers=num_hidden_layers,
        hidden_dim=hidden_dim,
        grid_size=grid_size,
        spline_order=spline_order,
        activation_name=activation_name,
    ).to(device)

    run = wandb.init(
         entity="zovkar-ufpe",
        project="IF702-Projeto-Final",
        name=f"KAN_trial-{trial.number}_lr-{lr:.4f}_bs-{batch_size}",
        group="kan_optimization",
        config={**trial.params, "variation": VARIATION, "input_dim": input_dim},
        dir=str(MODEL_DIR),
        reinit=True,
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss_path = os.path.join(run.dir, f"best_loss_kan_trial_{trial.number}.pth")
    best_ks_path = os.path.join(run.dir, f"best_ks_kan_trial_{trial.number}.pth")

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

        if epochs_without_improvement >= patience:
            print(f"Early stopping na época {epoch + 1} "
                  f"(melhor loss: época {best_loss_epoch} | melhor KS: época {best_ks_epoch})")
            break

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
        direction="maximize", # maximize ks
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top_5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]

    print(f"\n================ TOP {len(top_5)} MELHORES MODELOS (validação) ================\n")
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
