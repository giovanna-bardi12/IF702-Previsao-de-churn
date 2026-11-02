"""Shared data and AutoGluon helpers for Mitra experiments."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor

from src.data_loader import PROCESSED_DIR, TARGET_COLUMN
from src.metrics import calculate_metrics


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
VARIATION = "missing_encode_normalize"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = 1 if DEVICE == "cuda" else 0


def load_splits(data_dir=PROCESSED_DIR):
    """Read the same prepared CSV splits used by the KAN experiments."""
    data_dir = Path(data_dir)
    frames = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}_{VARIATION}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} não existe. Rode antes: python -m src.prepare_data")
        frames[split] = pd.read_csv(path)
        if TARGET_COLUMN not in frames[split]:
            raise ValueError(f"Coluna alvo {TARGET_COLUMN!r} não existe em {path}")
    return frames["train"], frames["val"], frames["test"]


def make_predictor(path, *, verbosity=2):
    """Create the official AutoGluon predictor configured for binary churn."""
    return TabularPredictor(
        label=TARGET_COLUMN,
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(path),
        verbosity=verbosity,
    )


def fit_predictor(train_df, val_df, path, params, *, time_limit=None, verbosity=2):
    """Fit the pretrained MITRA model, optionally fine-tuning its weights."""
    predictor = make_predictor(path, verbosity=verbosity)
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        hyperparameters={"MITRA": dict(params)},
        fit_weighted_ensemble=False,
        num_gpus=NUM_GPUS,
        time_limit=time_limit,
    )
    return predictor


def positive_probabilities(predictor, frame):
    """Return P(Churn=1), handling AutoGluon's Series/DataFrame outputs."""
    features = frame.drop(columns=[TARGET_COLUMN], errors="ignore")
    prediction = predictor.predict_proba(features, model="Mitra")
    if isinstance(prediction, pd.DataFrame):
        column = 1 if 1 in prediction.columns else prediction.columns[-1]
        prediction = prediction[column]
    return np.asarray(prediction, dtype=float).reshape(-1)


def evaluate_predictor(predictor, frame):
    labels = frame[TARGET_COLUMN].to_numpy(dtype=int)
    probabilities = positive_probabilities(predictor, frame)
    return calculate_metrics(labels, probabilities)
