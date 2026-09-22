"""
model.py
========
CNN model definition with transfer learning for Diabetic Retinopathy Stage Detection.

Architecture:
    Base:       EfficientNetB3 pretrained on ImageNet (feature extractor)
    Head:       GlobalAveragePooling → BatchNorm → Dropout(0.4)
                → Dense(256, relu) → BatchNorm → Dropout(0.3)
                → Dense(5, softmax)
    Fine-tune:  Unfreeze top N layers of the base after initial frozen training.

Usage:
    from src.model import build_model, get_callbacks
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
)
from pathlib import Path
from typing import List, Tuple

# ── Constants ────────────────────────────────────────────────────────────────
NUM_CLASSES: int    = 5       # No DR, Mild, Moderate, Severe, Proliferative DR
IMAGE_SIZE: Tuple[int, int, int] = (224, 224, 3)
DROPOUT_RATE_1: float = 0.4
DROPOUT_RATE_2: float = 0.3
DENSE_UNITS: int      = 256

CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR        = Path("logs")


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(
    num_classes: int = NUM_CLASSES,
    image_size: Tuple[int, int, int] = IMAGE_SIZE,
    learning_rate: float = 1e-3,
    freeze_base: bool = True,
) -> Model:
    """Build the EfficientNetB3 transfer-learning model.

    Args:
        num_classes:    Number of output classes (default 5 for DR stages).
        image_size:     Input shape as (H, W, C).
        learning_rate:  Initial learning rate for the Adam optimiser.
        freeze_base:    If True, freeze all base-model weights for stage-1
                        training; set False for full fine-tuning.

    Returns:
        Compiled :class:`keras.Model` ready for training.
    """
    # ── Base model (ImageNet weights, no top classifier) ──────────────────────
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=image_size,
    )
    base_model.trainable = not freeze_base

    # ── Custom classification head ─────────────────────────────────────────────
    inputs = keras.Input(shape=image_size, name="input_image")
    x = base_model(inputs, training=not freeze_base)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(DROPOUT_RATE_1, name="drop_1")(x)
    x = layers.Dense(DENSE_UNITS, activation="relu", name="fc_256")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(DROPOUT_RATE_2, name="drop_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="DR_EfficientNetB3")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_top_layers(model: Model, n_layers: int = 30) -> Model:
    """Unfreeze the top *n_layers* of the EfficientNetB3 base for fine-tuning.

    Call this after stage-1 training (frozen base) to fine-tune with a
    lower learning rate.

    Args:
        model:    The compiled model returned by :func:`build_model`.
        n_layers: Number of layers (from the end of the base) to unfreeze.

    Returns:
        Re-compiled model with unfrozen layers and a lower learning rate.
    """
    base_model: keras.layers.Layer = model.get_layer("efficientnetb3")
    base_model.trainable = True
    for layer in base_model.layers[:-n_layers]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Unfrozen top {n_layers} layers of EfficientNetB3 for fine-tuning.")
    return model


def print_model_summary(model: Model) -> None:
    """Print a human-readable model summary.

    Args:
        model: A compiled Keras model.
    """
    model.summary(line_length=90)
    total   = model.count_params()
    trainable = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"\nTotal params:     {total:,}")
    print(f"Trainable params: {trainable:,}")
    print(f"Frozen params:    {total - trainable:,}")


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(
    checkpoint_dir: Path = CHECKPOINT_DIR,
    log_dir: Path = LOG_DIR,
    monitor: str = "val_accuracy",
    patience: int = 10,
) -> List[keras.callbacks.Callback]:
    """Build the standard set of Keras training callbacks.

    Callbacks included:
        - :class:`ModelCheckpoint` — save best weights by *monitor* metric
        - :class:`EarlyStopping`   — stop when *monitor* plateaus
        - :class:`ReduceLROnPlateau` — halve LR when *monitor* stalls
        - :class:`TensorBoard`     — log metrics for visualisation
        - :class:`CSVLogger`       — save per-epoch metrics to CSV

    Args:
        checkpoint_dir: Directory to save model checkpoints.
        log_dir:        Directory for TensorBoard logs.
        monitor:        Metric to monitor (default ``"val_accuracy"``).
        patience:       Early-stopping patience in epochs.

    Returns:
        List of configured callback objects.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / "best_model.keras"

    return [
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir=str(log_dir), histogram_freq=1),
        CSVLogger(str(log_dir / "training_log.csv"), append=False),
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model(freeze_base=True)
    print_model_summary(model)
