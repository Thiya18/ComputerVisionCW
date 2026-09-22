"""
evaluate.py
===========
Model evaluation script for the Diabetic Retinopathy Stage Detection project.

Produces:
    - Classification report (precision, recall, F1-score per class)
    - Confusion matrix heatmap
    - Accuracy & loss curves (training vs. validation)
    - ROC curves (one-vs-rest, per class)
    - Per-class accuracy bar chart

All figures are saved to reports/screenshots/evaluation/ as PNG files.

Usage:
    python src/evaluate.py [--model checkpoints/best_model.keras]
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Dict, List, Tuple

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES: List[str] = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]
NUM_CLASSES: int = len(CLASS_NAMES)
IMAGE_SIZE:  Tuple[int, int] = (224, 224)
BATCH_SIZE:  int = 32
RANDOM_STATE: int = 42

DATA_AUG_DIR   = Path("data/augmented")
CHECKPOINT_DIR = Path("checkpoints")
HISTORY_DIR    = Path("reports")
EVAL_OUT_DIR   = Path("reports/screenshots/evaluation")
TRAIN_OUT_DIR  = Path("reports/screenshots/training")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Data loader ───────────────────────────────────────────────────────────────

def load_test_data(
    data_dir: Path = DATA_AUG_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    test_split: float = 0.15,
) -> tf.keras.preprocessing.image.DirectoryIterator:
    """Create a test-set generator (stratified, no shuffle).

    Uses a fixed validation_split of *test_split* as a proxy for a held-out
    test set (consistent with the split used in train.py).

    Args:
        data_dir:   Root directory with per-class subfolders.
        image_size: (H, W) target size.
        batch_size: Mini-batch size.
        test_split: Fraction to hold out for testing.

    Returns:
        A :class:`DirectoryIterator` over the test subset.
    """
    def preprocess_fn(image):
        image = tf.cast(image, tf.float32) / 255.0
        mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
        std  = tf.constant(IMAGENET_STD,  dtype=tf.float32)
        return (image - mean) / std

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        validation_split=test_split,
    )
    return datagen.flow_from_directory(
        str(data_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
        seed=RANDOM_STATE,
    )


# ── Prediction helpers ────────────────────────────────────────────────────────

def get_predictions(
    model: tf.keras.Model,
    generator: tf.keras.preprocessing.image.DirectoryIterator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over the full test generator.

    Args:
        model:     Loaded Keras model.
        generator: Test-set :class:`DirectoryIterator`.

    Returns:
        Tuple of (y_true, y_pred_labels, y_pred_probs) as numpy arrays.
    """
    y_true: List[int] = []
    y_pred_probs_list: List[np.ndarray] = []

    steps = len(generator)
    for step in range(steps):
        x, y = generator[step]
        probs = model.predict(x, verbose=0)
        y_true.extend(y.astype(int).tolist())
        y_pred_probs_list.append(probs)

    y_pred_probs = np.concatenate(y_pred_probs_list, axis=0)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    return np.array(y_true), y_pred_labels, y_pred_probs


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    out_dir: Path = EVAL_OUT_DIR,
) -> None:
    """Plot and save a normalised confusion matrix heatmap.

    Args:
        y_true:      Ground-truth class indices.
        y_pred:      Predicted class indices.
        class_names: List of class name strings.
        out_dir:     Directory where the figure is saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"],
        ["d", ".2f"],
    ):
        sns.heatmap(
            data,
            annot=True, fmt=fmt,
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    save_path = out_dir / "confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(
    history_paths: List[Path],
    out_dir: Path = TRAIN_OUT_DIR,
) -> None:
    """Plot accuracy and loss curves from saved JSON history files.

    Args:
        history_paths: Paths to ``training_history.json`` files
                       (stage1 first, then stage2).
        out_dir:       Directory where the figure is saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    combined: Dict[str, List[float]] = {
        "accuracy": [], "val_accuracy": [],
        "loss":     [], "val_loss":     [],
    }
    for path in history_paths:
        with open(path) as f:
            h = json.load(f)
        for key in combined:
            combined[key].extend(h.get(key, []))

    epochs = range(1, len(combined["accuracy"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, combined["accuracy"],     label="Train Accuracy", linewidth=2)
    ax1.plot(epochs, combined["val_accuracy"], label="Val Accuracy",   linewidth=2, linestyle="--")
    ax1.set_title("Accuracy over Epochs", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(True, alpha=0.4)

    # Loss
    ax2.plot(epochs, combined["loss"],     label="Train Loss", linewidth=2)
    ax2.plot(epochs, combined["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
    ax2.set_title("Loss over Epochs", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    save_path = out_dir / "training_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    out_dir: Path = EVAL_OUT_DIR,
) -> None:
    """Plot one-vs-rest ROC curves for each class.

    Args:
        y_true:        Ground-truth class indices.
        y_pred_probs:  Softmax probability matrix, shape (N, num_classes).
        class_names:   List of class name strings.
        out_dir:       Directory where the figure is saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    colours = sns.color_palette("husl", n_classes)
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (name, colour) in enumerate(zip(class_names, colours)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colour, linewidth=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_title("ROC Curves — One-vs-Rest", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    save_path = out_dir / "roc_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    out_dir: Path = EVAL_OUT_DIR,
) -> None:
    """Print and save a full sklearn classification report.

    Args:
        y_true:      Ground-truth class indices.
        y_pred:      Predicted class indices.
        class_names: List of class name strings.
        out_dir:     Directory where the text report is saved.
    """
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:\n")
    print(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")


# ── Main evaluation routine ───────────────────────────────────────────────────

def evaluate_model(model_path: Path = CHECKPOINT_DIR / "best_model.keras") -> None:
    """Run the full evaluation pipeline on the test set.

    Args:
        model_path: Path to the saved Keras model (.keras file).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train the model first: python src/train.py"
        )

    print(f"Loading model from {model_path} …")
    model = tf.keras.models.load_model(str(model_path))

    print("Loading test data …")
    test_gen = load_test_data()

    print("Running inference …")
    y_true, y_pred, y_probs = get_predictions(model, test_gen)

    # ── Evaluation outputs ─────────────────────────────────────────────────────
    print_classification_report(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
    plot_roc_curves(y_true, y_probs)

    # ── Training curves (from saved history JSON) ──────────────────────────────
    history_paths = [
        HISTORY_DIR / "stage1" / "training_history.json",
        HISTORY_DIR / "stage2" / "training_history.json",
    ]
    existing = [p for p in history_paths if p.exists()]
    if existing:
        plot_training_curves(existing)
    else:
        print("No training history JSON found — skipping curve plots.")

    print("\nEvaluation complete. All figures saved to reports/screenshots/evaluation/")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the DR stage detection model."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=CHECKPOINT_DIR / "best_model.keras",
        help="Path to the saved Keras model file.",
    )
    args = parser.parse_args()
    evaluate_model(model_path=args.model)
