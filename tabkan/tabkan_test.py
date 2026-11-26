"""
Usage (from the project root):
    python tabkan/tabkan_test.py                                # top 1 by val KS, best-KS checkpoint
    python tabkan/tabkan_test.py --top-k 3                      # top 3 by val KS (honest selection)
    python tabkan/tabkan_test.py --top-k 3 --rank-by test       # top 3 by TEST KS (all trials evaluated)
    python tabkan/tabkan_test.py --top-k 3 --rank-by test --pool 20   # only the 20 best val-KS trials compete
    python tabkan/tabkan_test.py --checkpoint loss              # use the lowest val_loss checkpoint
    python tabkan/tabkan_test.py --baseline                     # evaluate the fixed-hyperparameter baseline

Note: --rank-by test picks models by looking at the test set, so the reported
test numbers become optimistic. Use it for exploration, not as the final result.
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
BASELINE_DIR = MODEL_DIR / "baseline"  # created by tabkan_baseline.py


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


def test_baseline(checkpoint="ks"):
    """Evaluates the fixed-hyperparameter baseline saved by tabkan_baseline.py."""
    config_path = BASELINE_DIR / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found. Run first: python tabkan/tabkan_baseline.py")
    config = json.loads(config_path.read_text())

    weights_name, val_key = ("best_ks.pth", "val_best_ks_model") if checkpoint == "ks" \
        else ("best_loss.pth", "val_best_loss_model")
    weights = BASELINE_DIR / weights_name
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found.")

    params = config["params"]
    test_loss, inference_time, metrics = evaluate_on_test(params, config["input_dim"], weights)
    val_ks = config[val_key]["ks"]

    report(f"Baseline (checkpoint '{checkpoint}')", val_ks, test_loss, inference_time, metrics,
           tag=f"tabkan_baseline_{checkpoint}")
    row = make_row("TabKAN-baseline", 1, "baseline", checkpoint, val_ks,
                   test_loss, inference_time, metrics, params)
    row["rank_by"] = "baseline"
    return [row], RESULTS_DIR / f"tabkan_test_metrics_baseline_{checkpoint}.csv"


def test_optuna(top_k=1, checkpoint="ks", rank_by="val", pool=0):
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No completed trials. Run first: python tabkan/tabkan_train.py")
    completed = sorted(completed, key=lambda t: t.value, reverse=True)  # t.value = val KS

    if rank_by == "val":
        candidates = completed[:top_k]
    else:
        candidates = completed[:pool] if pool > 0 else completed
        print(f"[aviso] Ranqueando {len(candidates)} trials pelo KS do TESTE. "
              f"O resultado fica otimista; não use como estimativa final.\n")

    key = "model_path" if checkpoint == "ks" else "best_loss_model_path"
    evaluated = []
    for trial in candidates:
        weights = MODEL_DIR / trial.user_attrs[key]  # training saves paths relative to the tabkan/ folder
        if not weights.exists():
            print(f"[skip] Trial #{trial.number}: checkpoint não encontrado em {weights}")
            continue
        test_loss, inference_time, metrics = evaluate_on_test(
            trial.params, trial.user_attrs["input_dim"], weights
        )
        evaluated.append((trial, test_loss, inference_time, metrics))

    if not evaluated:
        raise RuntimeError("Nenhum checkpoint pôde ser avaliado.")

    if rank_by == "test":
        evaluated.sort(key=lambda item: item[3]["ks"], reverse=True)
    evaluated = evaluated[:top_k]

    rows = []
    for rank, (trial, test_loss, inference_time, metrics) in enumerate(evaluated, 1):
        report(f"Top {rank} by {rank_by} KS (Trial #{trial.number}, checkpoint '{checkpoint}')",
               trial.value, test_loss, inference_time, metrics,
               tag=f"tabkan_trial{trial.number}_{checkpoint}_{rank_by}")
        row = make_row("TabKAN-optuna", rank, trial.number, checkpoint, trial.value,
                       test_loss, inference_time, metrics, trial.params)
        row["rank_by"] = rank_by
        rows.append(row)

    return rows, RESULTS_DIR / f"tabkan_test_metrics_{checkpoint}_{rank_by}.csv"


def main(top_k=1, checkpoint="ks", rank_by="val", pool=0, baseline=False):
    RESULTS_DIR.mkdir(exist_ok=True)
    if baseline:
        rows, out = test_baseline(checkpoint)
    else:
        rows, out = test_optuna(top_k, checkpoint, rank_by, pool)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates TabKANs on the test set")
    parser.add_argument("--top-k", type=int, default=1, help="how many models to report")
    parser.add_argument("--checkpoint", choices=["ks", "loss"], default="ks",
                        help="which checkpoint to load: best KS or lowest val_loss")
    parser.add_argument("--rank-by", choices=["val", "test"], default="val",
                        help="rank trials by validation KS (default) or by test KS")
    parser.add_argument("--pool", type=int, default=0,
                        help="with --rank-by test: only the N best val-KS trials compete (0 = all)")
    parser.add_argument("--baseline", action="store_true",
                        help="evaluate the fixed-hyperparameter baseline (tabkan/baseline/) instead of Optuna trials")
    args = parser.parse_args()
    main(top_k=args.top_k, checkpoint=args.checkpoint, rank_by=args.rank_by,
         pool=args.pool, baseline=args.baseline)