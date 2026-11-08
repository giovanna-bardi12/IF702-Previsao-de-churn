"""
Uso (na raiz do projeto):
    python kan/kan_test.py --baseline         # baseline (checkpoint de melhor KS)
    python kan/kan_test.py                    # melhor trial, checkpoint de melhor KS
    python kan/kan_test.py --top-k 3          # as 3 melhores trials
    python kan/kan_test.py --checkpoint loss  # usa o checkpoint de menor val_loss
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
from kan_train import STORAGE, STUDY_NAME
from kan_train_baseline import BASELINE_DIR, CONFIG_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = MODEL_DIR / "test_results"


def load_model(params, input_dim, weights_path):
    """Reconstrói a KAN com os hiperparâmetros e carrega os pesos salvos."""
    model = build_kan(
        input_dim=input_dim,
        num_hidden_layers=params["num_hidden_layers"],
        hidden_dim=params["hidden_dim"],
        grid_size=params["grid_size"],
        spline_order=params["spline_order"],
        activation_name=params["activation"],
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model

def evaluate_on_test(params, input_dim, weights_path):
    model = load_model(params, input_dim, weights_path)
    _, _, test_loader = get_dataloaders(batch_size=params["batch_size"])
    test_loss, inference_time, labels, probs = validate_epoch(model, test_loader, nn.BCEWithLogitsLoss(), device)
    return test_loss, inference_time, calculate_metrics(labels, probs)


def report(title, val_ks, test_loss, inference_time, metrics, tag):
    print(f"================ TESTE: {title} ================")
    print(f"Val KS (seleção): {val_ks:.4f}")
    print(f"Test loss (BCE): {test_loss:.4f}")
    for k in METRIC_NAMES:
        print(f"Test {k}: {metrics[k]:.4f}")
    print(f"Tempo médio de inferência: {inference_time:.6f} s/batch")
    print("Matriz de confusão:\n", metrics["confusion_matrix"], "\n")

    fig = plot_confusion_matrix_figure(metrics["confusion_matrix"])
    fig.savefig(RESULTS_DIR / f"confusion_matrix_{tag}.png", bbox_inches="tight")
    plt.close(fig)

def make_row(model_name, rank, trial, checkpoint, val_ks, test_loss, inference_time, metrics, params):
    return {
        "model": model_name,
        "rank": rank,
        "trial": trial,
        "checkpoint": checkpoint,
        "val_ks": val_ks,
        "test_loss": test_loss,
        **{f"test_{k}": float(metrics[k]) for k in METRIC_NAMES},
        "test_inference_time_batch": inference_time,
        "confusion_matrix": json.dumps(metrics["confusion_matrix"].tolist()),
        "hyperparameters": json.dumps(params),
    }

def test_baseline(checkpoint="ks"):
    if not CONFIG_PATH.exists():
        raise RuntimeError("Baseline não encontrado. Rode antes: python kan/kan_baseline.py")
    config = json.loads(CONFIG_PATH.read_text())
    weights = BASELINE_DIR / f"best_{checkpoint}.pth"
    val_ks = config["val_best_ks_model" if checkpoint == "ks" else "val_best_loss_model"]["ks"]

    test_loss, inference_time, metrics = evaluate_on_test(config["params"], config["input_dim"], weights)
    report(f"KAN baseline (checkpoint '{checkpoint}')", val_ks, test_loss, inference_time,
           metrics, tag=f"kan_baseline_{checkpoint}")

    rows = [make_row("KAN-baseline", 1, "baseline", checkpoint, val_ks, test_loss,
                     inference_time, metrics, config["params"])]
    return rows, RESULTS_DIR / f"kan_baseline_test_metrics_{checkpoint}.csv"

def test_optuna(top_k=1, checkpoint="ks"):
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("Nenhuma trial concluída. Rode antes: python kan/kan_train.py")
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_k]

    rows = []
    for rank, trial in enumerate(top_trials, 1):
        key = "model_path" if checkpoint == "ks" else "best_loss_model_path"
        weights = MODEL_DIR / trial.user_attrs[key]  # o kan_train salva os caminhos relativos à pasta kan/

        test_loss, inference_time, metrics = evaluate_on_test(trial.params, trial.user_attrs["input_dim"], weights)
        report(f"Top {rank} (Trial #{trial.number}, checkpoint '{checkpoint}')", trial.value,
               test_loss, inference_time, metrics, tag=f"kan_trial{trial.number}_{checkpoint}")
        rows.append(make_row("KAN-optuna", rank, trial.number, checkpoint, trial.value, test_loss,
                             inference_time, metrics, trial.params))

    return rows, RESULTS_DIR / f"kan_test_metrics_{checkpoint}.csv"

def main(top_k=1, checkpoint="ks", baseline=False):
    RESULTS_DIR.mkdir(exist_ok=True)
    rows, out = test_baseline(checkpoint) if baseline else test_optuna(top_k, checkpoint)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Resultados salvos em {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia KANs no conjunto de teste")
    parser.add_argument("--baseline", action="store_true", help="avalia o modelo do kan_baseline.py")
    parser.add_argument("--top-k", type=int, default=1, help="quantas das melhores trials do Optuna avaliar")
    parser.add_argument("--checkpoint", choices=["ks", "loss"], default="ks",
                        help="qual checkpoint carregar: melhor KS ou menor val_loss")
    args = parser.parse_args()
    main(top_k=args.top_k, checkpoint=args.checkpoint, baseline=args.baseline)