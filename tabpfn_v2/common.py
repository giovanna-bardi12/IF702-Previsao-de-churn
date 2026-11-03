"""Shared data and TabPFN v2 helpers for the TabPFN experiments.

Three data "arms" are supported:

* ``raw``         - oversampled train/val (same rows as KAN/Mitra), raw features.
* ``processed``   - oversampled train/val, ``missing_encode_normalize`` features
                    (exact parity with the KAN and Mitra experiments).
* ``raw_natural`` - train/val WITHOUT repetition (natural ~27% churn), raw features.

The test split is the same in every arm: natural distribution, no repetition.
"""
import time
import warnings
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_curve

from src.data_loader import PROCESSED_DIR, RANDOM_SEED, TARGET_COLUMN, load_data, split_data
from src.metrics import calculate_metrics, ks_statistic
from src.prepare_data import encode_target


MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = RANDOM_SEED

VARIATION = "missing_encode_normalize"
ARMS = ("raw", "processed", "raw_natural")
ID_COLUMNS = ("customerID", "CustomerID", "customer_id")

# Hyperparameters that this project varies (names of TabPFNClassifier arguments).
TUNABLE_PARAMS = ("n_estimators", "softmax_temperature", "balance_probabilities", "average_before_softmax")


def tabpfn_version():
    try:
        return pkg_version("tabpfn")
    except PackageNotFoundError:
        return "unknown"


# --------------------------------------------------------------------------- data
@dataclass
class ArmData:
    arm: str
    description: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list
    categorical_indices: list = field(default_factory=list)

    @property
    def n_features(self):
        return self.X_train.shape[1]

    def summary(self):
        """Row counts / unique rows / churn rate, for W&B config and logs."""
        def rate(y):
            return float(np.mean(y))
        return {
            "arm": self.arm,
            "input_dim": self.n_features,
            "train_rows": len(self.y_train),
            "validation_rows": len(self.y_val),
            "test_rows": len(self.y_test),
            "train_churn_rate": rate(self.y_train),
            "val_churn_rate": rate(self.y_val),
            "test_churn_rate": rate(self.y_test),
            "n_categorical_features": len(self.categorical_indices),
        }


def _blank_to_nan(series):
    return series.replace(r"^\s*$", np.nan, regex=True)


def _prepare_raw_features(x_train, x_val, x_test):
    """Minimal conversion for the raw arms: NO scaling, NO one-hot, NO imputation.

    * identifier columns are dropped;
    * text columns whose values are all numbers (e.g. TotalCharges with blanks) become
      numeric, blanks become NaN (TabPFN handles missing values natively);
    * remaining text columns become integer codes learned on train; unseen values -> NaN.

    Returns float DataFrames and the positional indices of the categorical columns.
    """
    frames = []
    for x in (x_train, x_val, x_test):
        x = x.drop(columns=[c for c in ID_COLUMNS if c in x.columns]).copy()
        frames.append(x)
    x_train, x_val, x_test = frames

    categorical = []
    for column in list(x_train.columns):
        if is_numeric_dtype(x_train[column]):
            continue
        blank_free = _blank_to_nan(x_train[column])
        numeric_train = pd.to_numeric(blank_free, errors="coerce")
        if numeric_train.notna().sum() == blank_free.notna().sum():
            for x in (x_train, x_val, x_test):
                x[column] = pd.to_numeric(_blank_to_nan(x[column]), errors="coerce")
            continue
        categories = sorted(x_train[column].dropna().unique(), key=str)
        mapping = {value: code for code, value in enumerate(categories)}
        for x in (x_train, x_val, x_test):
            x[column] = x[column].astype(object).map(mapping).astype("float64")
        categorical.append(column)

    columns = list(x_train.columns)
    for x in (x_train, x_val, x_test):
        for column in columns:
            x[column] = x[column].astype("float64")
    return x_train, x_val, x_test, [columns.index(c) for c in categorical]


def _read_split_csvs(suffix, data_dir):
    frames = {}
    for split in ("train", "val", "test"):
        path = Path(data_dir) / f"{split}_{suffix}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} não existe. Rode antes: python -m src.prepare_data")
        frames[split] = pd.read_csv(path)
        if TARGET_COLUMN not in frames[split]:
            raise ValueError(f"Coluna alvo {TARGET_COLUMN!r} não existe em {path}")
    return frames["train"], frames["val"], frames["test"]


def _split_xy(df):
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN].to_numpy(dtype=int)


def _arm_from_raw_frames(arm, description, x_train, x_val, x_test, y_train, y_val, y_test):
    px_train, px_val, px_test, cat_idx = _prepare_raw_features(x_train, x_val, x_test)
    return ArmData(
        arm=arm,
        description=description,
        X_train=px_train.to_numpy(dtype=float), y_train=np.asarray(y_train, dtype=int),
        X_val=px_val.to_numpy(dtype=float), y_val=np.asarray(y_val, dtype=int),
        X_test=px_test.to_numpy(dtype=float), y_test=np.asarray(y_test, dtype=int),
        feature_names=list(px_train.columns),
        categorical_indices=cat_idx,
    )


def load_arm(arm, data_dir=PROCESSED_DIR, seed=SEED):
    """Load one experimental arm. Row membership of train/val/test never changes between arms."""
    if arm not in ARMS:
        raise ValueError(f"Arm desconhecido {arm!r}; escolha entre {ARMS}")

    if arm == "processed":
        train_df, val_df, test_df = _read_split_csvs(VARIATION, data_dir)
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = map(_split_xy, (train_df, val_df, test_df))
        return ArmData(
            arm=arm,
            description=f"oversampled train/val, {VARIATION} features (parity with KAN/Mitra)",
            X_train=x_tr.to_numpy(dtype=float), y_train=y_tr,
            X_val=x_va.to_numpy(dtype=float), y_val=y_va,
            X_test=x_te.to_numpy(dtype=float), y_test=y_te,
            feature_names=list(x_tr.columns),
            categorical_indices=[],
        )

    if arm == "raw":
        train_df, val_df, test_df = _read_split_csvs("raw", data_dir)
        (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = map(_split_xy, (train_df, val_df, test_df))
        return _arm_from_raw_frames(
            arm, "oversampled train/val, raw features", x_tr, x_va, x_te, y_tr, y_va, y_te
        )

    # raw_natural: same split, but train/val keep the natural class distribution.
    x_tr, x_va, x_te, y_tr, y_va, y_te = split_data(
        load_data(), target_column=TARGET_COLUMN, seed=seed, oversample=False
    )
    data = _arm_from_raw_frames(
        arm, "natural train/val (no repetition), raw features",
        x_tr, x_va, x_te, encode_target(y_tr), encode_target(y_va), encode_target(y_te),
    )
    _warn_if_test_differs(data, data_dir)
    return data


def _warn_if_test_differs(data, data_dir):
    """The test split must be identical across arms; warn loudly if the shared CSVs disagree."""
    try:
        reference = load_arm("raw", data_dir)
    except FileNotFoundError:
        return
    same = (
        np.array_equal(reference.y_test, data.y_test)
        and reference.X_test.shape == data.X_test.shape
        and np.allclose(reference.X_test, data.X_test, equal_nan=True)
    )
    if not same:
        warnings.warn(
            "O conjunto de teste de 'raw_natural' difere de 'raw' (test_raw.csv). "
            "Regenere os CSVs com python -m src.prepare_data antes de comparar os arms.",
            stacklevel=2,
        )


# ------------------------------------------------------------------------ TabPFN
def make_classifier(params, categorical_indices=None, seed=SEED):
    """TabPFN v2 (native package, v2 weights) with the given hyperparameter overrides."""
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    classifier = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
    overrides = dict(params)
    overrides["device"] = DEVICE
    overrides["random_state"] = seed
    if categorical_indices:
        overrides["categorical_features_indices"] = list(categorical_indices)

    unknown = sorted(set(overrides) - set(classifier.get_params()))
    if unknown:
        raise ValueError(
            f"A versão instalada do tabpfn ({tabpfn_version()}) não aceita: {unknown}. "
            f"Parâmetros disponíveis: {sorted(classifier.get_params())}"
        )
    classifier.set_params(**overrides)
    return classifier


def effective_params(classifier, keys=TUNABLE_PARAMS):
    """Values the model will actually use for the tuned hyperparameters (useful for the baseline)."""
    current = classifier.get_params()
    return {key: current[key] for key in keys if key in current}


def fit_classifier(params, data, seed=SEED, X=None, y=None):
    """'Fit' = store the context. Returns (classifier, seconds). Defaults to the train split."""
    X = data.X_train if X is None else X
    y = data.y_train if y is None else y
    classifier = make_classifier(params, data.categorical_indices, seed)
    started = time.perf_counter()
    classifier.fit(X, y)
    return classifier, time.perf_counter() - started


def positive_probabilities(classifier, X):
    """P(Churn=1) for every row of X, in a single batch. Returns (probabilities, seconds)."""
    started = time.perf_counter()
    proba = np.asarray(classifier.predict_proba(X), dtype=float)
    seconds = time.perf_counter() - started
    column = list(classifier.classes_).index(1)
    return proba[:, column], seconds


def release_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------- metrics
def ks_optimal_threshold(labels, probs):
    """Threshold that maximises TPR-FPR (= the KS statistic). Choose it on validation only."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def bootstrap_ci(labels, probs, metric_fn, n_boot=1000, alpha=0.05, seed=SEED):
    """Percentile bootstrap CI of a threshold-free metric on the test set."""
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if labels[idx].min() == labels[idx].max():
            continue
        values.append(metric_fn(labels[idx], probs[idx]))
    low, high = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


__all__ = [
    "ARMS", "ArmData", "DEVICE", "MODEL_DIR", "SEED", "TARGET_COLUMN", "TUNABLE_PARAMS", "VARIATION",
    "bootstrap_ci", "calculate_metrics", "effective_params", "fit_classifier", "ks_optimal_threshold",
    "ks_statistic", "load_arm", "make_classifier", "positive_probabilities", "release_gpu",
    "tabpfn_version",
]
