"""Shared data and AutoGluon helpers for Mitra experiments."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split

from src.data_loader import PROCESSED_DIR, RANDOM_SEED, TARGET_COLUMN
from src.metrics import calculate_metrics


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
VARIATION = os.environ.get("MITRA_PREPROCESSING", "missing_encode_normalize")
DATA_MODE = os.environ.get("MITRA_DATA_MODE", "balanced")
if VARIATION not in ("missing", "missing_encode_normalize"):
    raise ValueError("MITRA_PREPROCESSING must be 'missing' or 'missing_encode_normalize'")
if DATA_MODE not in ("balanced", "natural"):
    raise ValueError("MITRA_DATA_MODE must be 'balanced' or 'natural'")
EXPERIMENT_ID = f"{DATA_MODE}-{VARIATION}"
MITRA_DATA_DIR = PROCESSED_DIR / "mitra_natural"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = 1 if DEVICE == "cuda" else 0
PATIENCE = 10
MAX_FINE_TUNE_EPOCHS = 10_000


def load_splits(data_dir=MITRA_DATA_DIR):
    """Read independent splits; balance validation only for the balanced protocol."""
    data_dir = Path(data_dir)
    frames = {}
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}_{VARIATION}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist. Run: python -m mitra.prepare_data")
        frames[split] = pd.read_csv(path)
        if TARGET_COLUMN not in frames[split]:
            raise ValueError(f"Coluna alvo {TARGET_COLUMN!r} não existe em {path}")
    if DATA_MODE == "balanced":
        frames["val"] = balance_frame(frames["val"])
    return frames["train"], frames["val"], frames["test"]


def balance_frame(frame):
    """Repeat minority rows to balance classes without crossing split boundaries."""
    counts = frame[TARGET_COLUMN].value_counts()
    if len(counts) != 2:
        raise ValueError("Both churn classes are required for oversampling")
    target = counts.max()
    groups = [group if len(group) == target else pd.concat([
        group, group.sample(n=target - len(group), replace=True, random_state=RANDOM_SEED)
    ]) for _, group in frame.groupby(TARGET_COLUMN)]
    return pd.concat(groups).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def make_predictor(path, *, verbosity=2):
    """Create the official AutoGluon predictor configured for binary churn."""
    return TabularPredictor(
        label=TARGET_COLUMN,
        problem_type="binary",
        eval_metric="roc_auc",
        path=str(path),
        verbosity=verbosity,
    )


def fit_predictor(train_df, path, params, *, time_limit=None, verbosity=3):
    """Fit MITRA using an internal holdout for early stopping, not model selection."""
    if params.get("fine_tune", False):
        fit_df, early_stop_df = train_test_split(
            train_df, test_size=0.15, random_state=RANDOM_SEED,
            stratify=train_df[TARGET_COLUMN],
        )
    else:
        # AutoGluon otherwise reserves an automatic holdout, shrinking support.
        # This internal score is in-sample; model selection uses external validation.
        fit_df, early_stop_df = train_df, train_df
    if DATA_MODE == "balanced":
        fit_df = balance_frame(fit_df)
    print(f"MITRA data mode={DATA_MODE}, preprocessing={VARIATION}; "
          f"fit class counts={fit_df[TARGET_COLUMN].value_counts().to_dict()}, "
          f"internal validation class counts={early_stop_df[TARGET_COLUMN].value_counts().to_dict()}",
          flush=True)
    predictor = make_predictor(path, verbosity=verbosity)
    predictor.fit(
        train_data=fit_df,
        tuning_data=early_stop_df,
        hyperparameters={"MITRA": dict(params)},
        presets="medium",
        fit_weighted_ensemble=False,
        num_gpus=NUM_GPUS,
        time_limit=time_limit,
    )
    return predictor


def positive_probabilities(predictor, frame):
    """Return P(Churn=1), handling AutoGluon's Series/DataFrame outputs."""
    features = frame.drop(columns=[TARGET_COLUMN], errors="ignore")
    prediction = predictor.predict_proba(features)
    if isinstance(prediction, pd.DataFrame):
        if 1 not in prediction.columns:
            raise ValueError(f"Expected churn class 1 in probability columns: {prediction.columns.tolist()}")
        prediction = prediction[1]
    return np.asarray(prediction, dtype=float).reshape(-1)


def evaluate_predictor(predictor, frame):
    labels = frame[TARGET_COLUMN].to_numpy(dtype=int)
    probabilities = positive_probabilities(predictor, frame)
    return calculate_metrics(labels, probabilities)
