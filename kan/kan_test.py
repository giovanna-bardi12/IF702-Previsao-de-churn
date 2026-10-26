"""
escolhe as melhores trials pela métrica
de validação, recarrega o checkpoint salvo e calcula as métricas no teste.
Os resultados vão para test_results/.

Uso (na raiz do projeto, depois de rodar o train_kan.py):
    python test_kan.py                    # melhor trial, checkpoint de melhor KS
    python test_kan.py --top-k 3          # as 3 melhores trials
    python test_kan.py --checkpoint loss  # usa o checkpoint de menor val_loss
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parent      # .../kan
PROJECT_ROOT = MODEL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))



from src.data_loader import get_dataloaders
from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure
from models_kan import build_kan, validate_epoch
from kan_train import STORAGE, STUDY_NAME, VARIATION,PROJECT_ROOT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = MODEL_DIR / "test_results"


def load_model(trial, checkpoint="ks"):
    """Reconstrói a KAN com os hiperparâmetros da trial e carrega os pesos salvos."""
    p = trial.params
    model = build_kan(
        input_dim=trial.user_attrs["input_dim"],
        num_hidden_layers=p["num_hidden_layers"],
        hidden_dim=p["hidden_dim"],
        grid_size=p["grid_size"],
        spline_order=p["spline_order"],
        activation_name=p["activation"],
    ).to(device)


    rel_path = trial.user_attrs["model_path"] if checkpoint == "ks" else trial.user_attrs["best_loss_model_path"]
    path = PROJECT_ROOT / rel_path
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def evaluate_on_test(trial, checkpoint="ks"):
    model = load_model(trial, checkpoint)
    _, _, test_loader = get_dataloaders(batch_size=trial.params["batch_size"], variation=VARIATION)
    test_loss, inference_time, labels, probs = validate_epoch(model, test_loader, nn.BCEWithLogitsLoss(), device)
    return test_loss, inference_time, calculate_metrics(labels, probs)


def main(top_k=1, checkpoint="ks"):
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("Nenhuma trial concluída. Rode antes: python train_kan.py")
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_k]

    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []

    for rank, trial in enumerate(top_trials, 1):
        test_loss, inference_time, metrics = evaluate_on_test(trial, checkpoint)

        print(f"================ TESTE: Top {rank} (Trial #{trial.number}, checkpoint '{checkpoint}') ================")
        print(f"Val KS (seleção): {trial.value:.4f}")
        print(f"Test loss (BCE): {test_loss:.4f}")
        for k in METRIC_NAMES:
            print(f"Test {k}: {metrics[k]:.4f}")
        print(f"Tempo médio de inferência: {inference_time:.6f} s/batch")
        print("Matriz de confusão:\n", metrics["confusion_matrix"], "\n")

        tag = f"kan_trial{trial.number}_{checkpoint}"
        fig = plot_confusion_matrix_figure(metrics["confusion_matrix"])
        fig.savefig(RESULTS_DIR / f"confusion_matrix_{tag}.png", bbox_inches="tight")
        plt.close(fig)

        rows.append({
            "model": "KAN",
            "rank": rank,
            "trial": trial.number,
            "checkpoint": checkpoint,
            "val_ks": trial.value,
            "test_loss": test_loss,
            **{f"test_{k}": float(metrics[k]) for k in METRIC_NAMES},
            "test_inference_time_batch": inference_time,
            "confusion_matrix": json.dumps(metrics["confusion_matrix"].tolist()),
            "hyperparameters": json.dumps(trial.params),
        })

    out = RESULTS_DIR / f"kan_test_metrics_{checkpoint}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Resultados salvos em {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia no teste as melhores KANs do Optuna")
    parser.add_argument("--top-k", type=int, default=1, help="quantas das melhores trials avaliar")
    parser.add_argument("--checkpoint", choices=["ks", "loss"], default="ks",
                        help="qual checkpoint carregar: melhor KS ou menor val_loss")
    args = parser.parse_args()
    main(top_k=args.top_k, checkpoint=args.checkpoint)