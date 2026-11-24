import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

METRIC_NAMES = ["ks", "roc_auc", "mse", "binary_cross_entropy", "accuracy", "precision", "recall", "f1"]


def ks_statistic(labels, probs):
    """
    Vai de 0 (o modelo não separa as classes) a 1 (separação perfeita).
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    return ks_2samp(probs[labels == 1], probs[labels == 0]).statistic


def calculate_metrics(labels, probs, threshold=0.5):
    """Métricas de classificação binária.

    labels: 0/1 | probs: P(churn), saída do sigmoid.
    KS, AUROC, MSE e entropia cruzada usam as probabilidades (sem limiar);
    matriz de confusão, acurácia, precision, recall e F1 usam probs >= threshold.
    """
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    return {
        "ks": ks_statistic(labels, probs),
        "roc_auc": roc_auc_score(labels, probs),
        "mse": mean_squared_error(labels, probs),
        "binary_cross_entropy": log_loss(labels, probs, labels=[0, 1]),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]),
    }


def validation_ks_threshold(labels, probs):
    """Choose the finite threshold maximizing TPR - FPR on validation only."""
    fpr, tpr, thresholds = roc_curve(labels, probs, drop_intermediate=False)
    valid = np.isfinite(thresholds)
    if not valid.any():
        raise ValueError("No finite validation threshold available")
    indices = np.flatnonzero(valid)
    return float(thresholds[indices[np.argmax((tpr - fpr)[valid])]])


def plot_roc_figure(labels, probs):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(labels, probs):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_ks_figure(labels, probs):
    labels, probs = np.asarray(labels), np.asarray(probs)
    scores = np.unique(probs)
    cdf0 = np.searchsorted(np.sort(probs[labels == 0]), scores, side="right") / np.sum(labels == 0)
    cdf1 = np.searchsorted(np.sort(probs[labels == 1]), scores, side="right") / np.sum(labels == 1)
    index = int(np.argmax(np.abs(cdf0 - cdf1)))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.step(scores, cdf0, where="post", label="No churn")
    ax.step(scores, cdf1, where="post", label="Churn")
    ax.vlines(scores[index], cdf0[index], cdf1[index], color="black", linestyle="--",
              label=f"KS = {abs(cdf0[index] - cdf1[index]):.4f}")
    ax.set(xlabel="Predicted churn score", ylabel="Empirical CDF", title="KS separation")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confusion_matrix_figure(cm, class_names=("Não churn", "Churn")):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Matriz de confusão")
    fig.tight_layout()
    return fig
