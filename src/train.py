"""
train.py
========
Training script for the Diabetic Retinopathy Stage Detection model.

Dataset layout (APTOS 2019 Blindness Detection):
    Reads from ``data/augmented/<label>/`` — the balanced, per-class
    subfolder structure written by augmentation.py.
    Labels originate from ``train.csv`` (via preprocessing.py → augmentation.py).

Two-stage training strategy:
    Stage 1 — Frozen base:  Train only the classification head (high LR).
    Stage 2 — Fine-tuning:  Unfreeze top 30 layers of EfficientNetB3 (low LR).

Usage:
    python src/train.py [--epochs 30] [--batch-size 32] [--finetune-epochs 20]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import json
import argparse
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from typing import Dict, Tuple

import config as cfg
from model import build_model, unfreeze_top_layers, get_callbacks

# ── Reproducibility ───────────────────────────────────────────────────────────
tf.random.set_seed(cfg.RANDOM_STATE)
np.random.seed(cfg.RANDOM_STATE)

# ── Path aliases ──────────────────────────────────────────────────────────────
DATA_AUG_DIR   = cfg.DATA_AUG_DIR
CHECKPOINT_DIR = cfg.CHECKPOINT_DIR
LOG_DIR        = cfg.LOG_DIR
HISTORY_DIR    = cfg.HISTORY_DIR


# ── Data loading ──────────────────────────────────────────────────────────────

def _preprocess_fn(image: tf.Tensor) -> tf.Tensor:
    """Rescale uint8 [0, 255] to float32 then apply ImageNet normalisation."""
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant(cfg.IMAGENET_MEAN, dtype=tf.float32)
    std  = tf.constant(cfg.IMAGENET_STD,  dtype=tf.float32)
    return (image - mean) / std


def load_data(
    data_dir: Path = DATA_AUG_DIR,
    image_size: Tuple[int, int] = cfg.IMAGE_SIZE,
    batch_size: int = cfg.BATCH_SIZE,
) -> Tuple:
    """Create train and validation generators from the augmented dataset.

    Reads from ``data/augmented/train/`` and ``data/augmented/val/``.

    Args:
        data_dir:   Root directory containing train, val, and test subfolders.
        image_size: (H, W) tuple for input images.
        batch_size: Mini-batch size.

    Returns:
        Tuple of (train_generator, val_generator, train_class_labels_array).
    """
    datagen = ImageDataGenerator(preprocessing_function=_preprocess_fn)

    train_gen = datagen.flow_from_directory(
        str(data_dir / "train"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=True,
        seed=cfg.RANDOM_STATE,
    )

    val_gen = datagen.flow_from_directory(
        str(data_dir / "val"),
        target_size=image_size,
        batch_size=batch_size,
        class_mode="sparse",
        shuffle=False,
        seed=cfg.RANDOM_STATE,
    )

    return train_gen, val_gen, train_gen.classes


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights to handle residual class imbalance.

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


def save_history(
    history: tf.keras.callbacks.History,
    out_dir: Path = HISTORY_DIR,
) -> None:
    """Save training history dict as a JSON file for later plotting.

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
    epochs_frozen: int = cfg.EPOCHS_FROZEN,
    epochs_finetune: int = cfg.EPOCHS_FINETUNE,
    batch_size: int = cfg.BATCH_SIZE,
) -> None:
    """Run the two-stage training pipeline.

    Stage 1: Train classification head with frozen EfficientNetB3 base.
    Stage 2: Fine-tune top 30 layers with a 100x lower learning rate.

    Args:
        epochs_frozen:    Max epochs for stage-1 (frozen base).
        epochs_finetune:  Max epochs for stage-2 (fine-tuning).
        batch_size:       Mini-batch size.
    """
    if not DATA_AUG_DIR.exists() or not any(DATA_AUG_DIR.iterdir()):
        raise RuntimeError(
            f"Augmented data not found at {DATA_AUG_DIR}.\n"
            "Run src/augmentation.py first."
        )

    # ── Load data ──────────────────────────────────────────────────────────────
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
    callbacks_ft = get_callbacks(
        CHECKPOINT_DIR / "finetune", LOG_DIR / "finetune"
    )

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
    parser.add_argument(
        "--epochs", type=int, default=cfg.EPOCHS_FROZEN,
        help="Max epochs for stage-1 frozen training.",
    )
    parser.add_argument(
        "--finetune-epochs", type=int, default=cfg.EPOCHS_FINETUNE,
        help="Max epochs for stage-2 fine-tuning.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=cfg.BATCH_SIZE,
        help="Mini-batch size.",
    )
    args = parser.parse_args()
    train(
        epochs_frozen=args.epochs,
        epochs_finetune=args.finetune_epochs,
        batch_size=args.batch_size,
    )
