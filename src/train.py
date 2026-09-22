"""
train.py
========
Training script for the Diabetic Retinopathy Stage Detection model.

Two-stage training strategy:
    Stage 1 — Frozen base:  Train only the classification head (high LR, fast convergence).
    Stage 2 — Fine-tuning:  Unfreeze top 30 layers of EfficientNetB3 (low LR, precision).

Usage:
    python src/train.py [--epochs 30] [--batch-size 32] [--finetune-epochs 20]
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Dict, Tuple

from model import build_model, unfreeze_top_layers, get_callbacks

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_AUG_DIR   = Path("data/augmented")
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR        = Path("logs")
HISTORY_DIR    = Path("reports")

# ── Defaults ──────────────────────────────────────────────────────────────────
IMAGE_SIZE    = (224, 224)
BATCH_SIZE    = 32
EPOCHS_FROZEN = 30
EPOCHS_FINETUNE = 20
VALIDATION_SPLIT = 0.15
NUM_CLASSES   = 5

# ImageNet normalisation (must match preprocessing.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    data_dir: Path = DATA_AUG_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
    val_split: float = VALIDATION_SPLIT,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray]:
    """Create train and validation :class:`tf.data.Dataset` objects.

    Uses :class:`ImageDataGenerator` with the same normalisation applied
    in preprocessing.py (ImageNet mean/std) and no extra augmentation
    (augmentation was done offline in augmentation.py).

    Args:
        data_dir:   Root directory with per-class subfolders.
        image_size: (H, W) tuple for input images.
        batch_size: Mini-batch size.
        val_split:  Fraction of data to use for validation.

    Returns:
        Tuple of (train_generator, val_generator, class_labels_array).
    """
    def preprocess_fn(image):
        """Rescale to [0,1] then apply ImageNet normalisation."""
        image = tf.cast(image, tf.float32) / 255.0
        mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
        std  = tf.constant(IMAGENET_STD,  dtype=tf.float32)
        return (image - mean) / std

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        validation_split=val_split,
    )

    train_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="training",
        shuffle=True,
        seed=RANDOM_STATE,
    )

    val_gen = datagen.flow_from_directory(
        str(data_dir),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
        seed=RANDOM_STATE,
    )

    labels = train_gen.classes
    return train_gen, val_gen, labels


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights to handle class imbalance.

    Args:
        labels: 1-D array of integer class labels from the training generator.

    Returns:
        Dict mapping class index → weight scalar.
    """
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels,
    )
    class_weight_dict = dict(zip(classes.tolist(), weights.tolist()))
    print("Class weights:", {k: round(v, 4) for k, v in class_weight_dict.items()})
    return class_weight_dict


def save_history(history: tf.keras.callbacks.History, out_dir: Path = HISTORY_DIR) -> None:
    """Save training history as a JSON file for later plotting.

    Args:
        history: Keras History object returned by :meth:`model.fit`.
        out_dir: Directory where ``training_history.json`` is saved.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    path = out_dir / "training_history.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Training history saved to {path}")


# ── Main training routine ─────────────────────────────────────────────────────

def train(
    epochs_frozen: int = EPOCHS_FROZEN,
    epochs_finetune: int = EPOCHS_FINETUNE,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Run the two-stage training pipeline.

    Stage 1: Train classification head with frozen base.
    Stage 2: Fine-tune top 30 layers of EfficientNetB3.

    Args:
        epochs_frozen:    Max epochs for stage-1 (frozen base).
        epochs_finetune:  Max epochs for stage-2 (fine-tuning).
        batch_size:       Mini-batch size.
    """
    # ── Load data ──────────────────────────────────────────────────────────────
    if not DATA_AUG_DIR.exists():
        raise RuntimeError(
            f"Augmented data directory not found: {DATA_AUG_DIR}\n"
            "Run src/augmentation.py first."
        )

    train_gen, val_gen, labels = load_data(batch_size=batch_size)
    class_weights = compute_class_weights(labels)

    # ── Stage 1: Frozen base ───────────────────────────────────────────────────
    print("\n── Stage 1: Training classification head (frozen base) ──")
    model = build_model(freeze_base=True)
    callbacks = get_callbacks(CHECKPOINT_DIR, LOG_DIR)

    history_1 = model.fit(
        train_gen,
        epochs=epochs_frozen,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks,
    )
    save_history(history_1, HISTORY_DIR / "stage1")

    # ── Stage 2: Fine-tuning ───────────────────────────────────────────────────
    print("\n── Stage 2: Fine-tuning top layers of EfficientNetB3 ──")
    model = unfreeze_top_layers(model, n_layers=30)
    callbacks_ft = get_callbacks(CHECKPOINT_DIR / "finetune", LOG_DIR / "finetune")

    history_2 = model.fit(
        train_gen,
        epochs=epochs_finetune,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=callbacks_ft,
    )
    save_history(history_2, HISTORY_DIR / "stage2")

    # ── Save final model ───────────────────────────────────────────────────────
    final_path = CHECKPOINT_DIR / "final_model.keras"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the DR stage detection model (EfficientNetB3)."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS_FROZEN,
                        help="Max epochs for stage-1 frozen training.")
    parser.add_argument("--finetune-epochs", type=int, default=EPOCHS_FINETUNE,
                        help="Max epochs for stage-2 fine-tuning.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Mini-batch size.")
    args = parser.parse_args()

    train(
        epochs_frozen=args.epochs,
        epochs_finetune=args.finetune_epochs,
        batch_size=args.batch_size,
    )
