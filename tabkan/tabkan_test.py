"""
Usage (from the project root):
    python tabkan/tabkan_test.py                    # best trial, best-KS checkpoint
    python tabkan/tabkan_test.py --top-k 3          # top 3 trials
    python tabkan/tabkan_test.py --checkpoint loss  # use the lowest val_loss checkpoint
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

MODEL_DIR = Path(__file__).resolve().parent      # .../tabkan
PROJECT_ROOT = MODEL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_dataloaders
from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure
from models_tabkan import TabKANWrapper, validate_epoch
from tabkan_train import STORAGE, STUDY_NAME  # adjust if your training file has another name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = MODEL_DIR / "test_results"


def load_model(params, input_dim, weights_path):
    """Rebuilds the TabKAN with the trial hyperparameters and loads the saved weights."""
    model = TabKANWrapper(
        num_features=input_dim,
        num_layers=params["num_layers"],
        token_dim=params["token_dim"],
        channel_dim=params["channel_dim"],
        token_order=params["token_order"],
        channel_order=params["channel_order"],
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def evaluate_on_test(params, input_dim, weights_path):
    model = load_model(params, input_dim, weights_path)
    _, _, test_loader = get_dataloaders(batch_size=params["batch_size"])
    test_loss, inference_time, labels, probs = validate_epoch(
        model, test_loader, nn.BCEWithLogitsLoss(), device
    )
    return test_loss, inference_time, calculate_metrics(labels, probs)


def report(title, val_ks, test_loss, inference_time, metrics, tag):
    print(f"================ TEST: {title} ================")
    print(f"Val KS (selection): {val_ks:.4f}")
    print(f"Test loss (BCE): {test_loss:.4f}")
    for k in METRIC_NAMES:
        print(f"Test {k}: {metrics[k]:.4f}")
    print(f"Average inference time: {inference_time:.6f} s/batch")
    print("Confusion matrix:\n", metrics["confusion_matrix"], "\n")

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


def test_optuna(top_k=1, checkpoint="ks"):
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No completed trials. Run first: python tabkan/tabkan_train.py")
    top_trials = sorted(completed, key=lambda t: t.value, reverse=True)[:top_k]

    rows = []
    for rank, trial in enumerate(top_trials, 1):
        key = "model_path" if checkpoint == "ks" else "best_loss_model_path"
        weights = MODEL_DIR / trial.user_attrs[key]  # training saves paths relative to the tabkan/ folder

        test_loss, inference_time, metrics = evaluate_on_test(
            trial.params, trial.user_attrs["input_dim"], weights
        )
        report(f"Top {rank} (Trial #{trial.number}, checkpoint '{checkpoint}')", trial.value,
               test_loss, inference_time, metrics, tag=f"tabkan_trial{trial.number}_{checkpoint}")
        rows.append(make_row("TabKAN-optuna", rank, trial.number, checkpoint, trial.value,
                             test_loss, inference_time, metrics, trial.params))

    return rows, RESULTS_DIR / f"tabkan_test_metrics_{checkpoint}.csv"


def main(top_k=1, checkpoint="ks"):
    RESULTS_DIR.mkdir(exist_ok=True)
    rows, out = test_optuna(top_k, checkpoint)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates TabKANs on the test set")
    parser.add_argument("--top-k", type=int, default=1, help="how many of the best Optuna trials to evaluate")
    parser.add_argument("--checkpoint", choices=["ks", "loss"], default="ks",
                        help="which checkpoint to load: best KS or lowest val_loss")
    args = parser.parse_args()
    main(top_k=args.top_k, checkpoint=args.checkpoint)