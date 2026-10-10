"""
Evaluator module for Brain Tumor Detection and Classification.

Computes metrics, generates confusion matrix and training curve plots,
and builds a cross-model comparison table.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import CLASS_NAMES


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    model_name: str,
    class_names: list = None,
    results_dir: str = "results",
) -> dict:
    """
    Evaluate *model* on *test_ds* and save artifacts.

    Computes accuracy, per-class precision/recall/F1, and macro averages.
    Saves a confusion matrix PNG and a metrics JSON file.

    Parameters
    ----------
    model:
        Trained Keras model.
    test_ds:
        Ordered, non-shuffled test ``tf.data.Dataset`` yielding
        ``(images, one_hot_labels)``.
    model_name:
        Architecture identifier used to name output files.
    class_names:
        List of class label strings.  Defaults to ``CLASS_NAMES``.
    results_dir:
        Root results directory.  Artifacts are saved under
        ``{results_dir}/{model_name}/``.

    Returns
    -------
    dict
        Metrics dictionary with keys: ``accuracy``, ``macro_precision``,
        ``macro_recall``, ``macro_f1``, and ``per_class`` (dict of
        ``{class_name: {precision, recall, f1}}``).
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Collect predictions and ground-truth labels
    all_preds = []
    all_labels = []
    for images, one_hot_labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1).tolist())
        all_labels.extend(np.argmax(one_hot_labels.numpy(), axis=1).tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    accuracy = float(accuracy_score(all_labels, all_preds))
    macro_precision = float(precision_score(all_labels, all_preds, average="macro", zero_division=0))
    macro_recall = float(recall_score(all_labels, all_preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))

    per_class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_class_precision[i]),
            "recall": float(per_class_recall[i]),
            "f1": float(per_class_f1[i]),
        }

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }

    # Save artifacts
    out_dir = os.path.join(results_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    _save_confusion_matrix(cm, class_names, model_name, out_dir)

    # Metrics JSON
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Evaluator] Metrics saved to: {metrics_path}")

    # Print classification report
    print(f"\n[Evaluator] Classification report for {model_name}:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    return metrics


def _save_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    model_name: str,
    out_dir: str,
) -> None:
    """Save a heatmap of *cm* as a PNG file."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name.upper()}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Evaluator] Confusion matrix saved to: {path}")


def plot_training_curves(
    history: tf.keras.callbacks.History,
    model_name: str,
    results_dir: str = "results",
) -> None:
    """
    Plot and save training/validation loss and accuracy curves.

    Parameters
    ----------
    history:
        Keras ``History`` object returned by ``model.fit()``.
    model_name:
        Architecture identifier used for the output filename and plot title.
    results_dir:
        Root results directory.
    """
    out_dir = os.path.join(results_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    epochs_range = range(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs_range, history.history["loss"], label="Train Loss")
    axes[0].plot(epochs_range, history.history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{model_name.upper()} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs_range, history.history["accuracy"], label="Train Accuracy")
    axes[1].plot(epochs_range, history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title(f"{model_name.upper()} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.suptitle(f"Training Curves — {model_name.upper()}", fontsize=14)
    plt.tight_layout()

    path = os.path.join(out_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Evaluator] Training curves saved to: {path}")


def build_comparison_table(
    metrics: dict,
    results_dir: str = "results",
) -> pd.DataFrame:
    """
    Build a cross-model comparison DataFrame and save it as CSV.

    Parameters
    ----------
    metrics:
        Dict of ``{model_name: metrics_dict}`` for all evaluated models.
    results_dir:
        Root results directory.  CSV is saved to
        ``{results_dir}/comparison_table.csv``.

    Returns
    -------
    pd.DataFrame
        Columns: Model, Accuracy, Macro-P, Macro-R, Macro-F1.
    """
    rows = []
    for model_name, m in metrics.items():
        rows.append({
            "Model": model_name.upper(),
            "Accuracy": round(m["accuracy"], 4),
            "Macro-P": round(m["macro_precision"], 4),
            "Macro-R": round(m["macro_recall"], 4),
            "Macro-F1": round(m["macro_f1"], 4),
        })

    df = pd.DataFrame(rows)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Evaluator] Comparison table saved to: {csv_path}")
    print(df.to_string(index=False))
    return df
