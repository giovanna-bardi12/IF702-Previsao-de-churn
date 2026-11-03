# Avalia STab no teste

import argparse
import json
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_DIR.parent))

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import torch
from torch import nn
import wandb

from models_stab import build_stab, validate_epoch
from stab_train import DEVICE, STORAGE, STUDY_NAME, VARIATION, WANDB_ENTITY, WANDB_PROJECT
from src.data_loader import get_dataloaders
from src.metrics import METRIC_NAMES, calculate_metrics, plot_confusion_matrix_figure
from src.run_tags import wandb_provenance_tags

RESULTS_DIR = MODEL_DIR / "test_results"


def evaluate(params, input_dim, weights_path):
    d_model = params["d_model"]
    model = build_stab(
        n_features=input_dim, d_model=d_model, n_heads=params["n_heads"],
        n_layers=params["n_layers"], ffn_multiplier=4, gate_hidden=4 * d_model,
        classifier_hidden=2 * d_model, dropout=params["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    _, _, test_loader = get_dataloaders(batch_size=params["batch_size"])
    loss, inference_time, labels, probs = validate_epoch(
        model, test_loader, nn.BCEWithLogitsLoss(), DEVICE)
    return loss, inference_time, calculate_metrics(labels, probs)


def selected_models(baseline, top_k, checkpoint):
    if baseline:
        config_path = MODEL_DIR / "baseline" / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        val_ks = config[f"val_best_{checkpoint}_model"]["ks"]
        return [("baseline", None, config["params"], config["input_dim"],
                 MODEL_DIR / config["model_paths"][checkpoint], val_ks)]

    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    completed = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE),
                       key=lambda t: t.value, reverse=True)
    if not completed:
        raise RuntimeError("Nenhum trial STab concluído")
    models = []
    for trial in completed[:top_k]:
        val_ks = (trial.value if checkpoint == "ks"
                  else trial.user_attrs["val_best_loss_model"]["ks"])
        models.append((f"trial_{trial.number}", trial.number, trial.params,
                       trial.user_attrs["input_dim"],
                       MODEL_DIR / trial.user_attrs["model_paths"][checkpoint], val_ks))
    return models


def main(baseline=False, top_k=1, checkpoint="ks"):
   
    models = selected_models(baseline, top_k, checkpoint)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT,
        name=f"STab_test_{'baseline' if baseline else 'top_trials'}_{checkpoint}",
        group="stab_test", dir=str(MODEL_DIR), reinit=True,
        tags=["model:stab", "task:churn", "stage:test",
              f"strategy:{'baseline' if baseline else 'optuna'}", f"checkpoint:{checkpoint}",
              f"data:{VARIATION}", f"device:{DEVICE.type}", *wandb_provenance_tags()],
        config={"baseline": baseline, "top_k": len(models), "checkpoint": checkpoint},
    )
    rows = []
    try:
        for rank, (name, trial_number, params, input_dim, weights, val_ks) in enumerate(models, 1):
            test_loss, inference_time, metrics = evaluate(params, input_dim, weights)
            print(f"{name}: val_KS={val_ks:.4f}, test_KS={metrics['ks']:.4f}, test_BCE={test_loss:.4f}")
            fig = plot_confusion_matrix_figure(metrics["confusion_matrix"])
            image_path = RESULTS_DIR / f"confusion_matrix_{name}_{checkpoint}.png"
            fig.savefig(image_path, bbox_inches="tight")
            run.log({"rank": rank, "trial": trial_number if trial_number is not None else -1,
                     "val_ks": val_ks, "test_loss": test_loss,
                     "test_inference_time_batch": inference_time,
                     **{f"test_{key}": float(metrics[key]) for key in METRIC_NAMES},
                     "test_confusion_matrix": wandb.Image(fig)})
            plt.close(fig)
            wandb.save(str(image_path), base_path=str(MODEL_DIR))
            rows.append({"model": name, "trial": trial_number, "checkpoint": checkpoint,
                         "val_ks": val_ks, "test_loss": test_loss,
                         "test_inference_time_batch": inference_time,
                         **{f"test_{key}": float(metrics[key]) for key in METRIC_NAMES},
                         "confusion_matrix": json.dumps(metrics["confusion_matrix"].tolist()),
                         "hyperparameters": json.dumps(params)})
        output_path = RESULTS_DIR / f"stab_{'baseline_' if baseline else ''}test_metrics_{checkpoint}.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        wandb.save(str(output_path), base_path=str(MODEL_DIR))
        print(f"Resultados salvos em {output_path}")
    finally:
        run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia STab no conjunto de teste")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--checkpoint", choices=("ks", "loss"), default="ks")
    args = parser.parse_args()
    main(args.baseline, args.top_k, args.checkpoint)
