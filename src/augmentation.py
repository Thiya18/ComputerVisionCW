"""
augmentation.py
===============
Data augmentation pipeline for the Diabetic Retinopathy Stage Detection project.

Uses the `albumentations` library for fast, composable, reproducible augmentation.
Addresses class imbalance by oversampling minority classes with augmented images.

Augmentation strategies:
    - Geometric: horizontal/vertical flip, rotation, zoom (scale), shift
    - Photometric: brightness/contrast jitter, hue/saturation shift, gamma
    - Blur/noise: Gaussian blur, ISO noise
    - Advanced: GridDistortion (mimics retinal imaging artefacts)

Usage:
    python src/augmentation.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
IMAGE_SIZE: Tuple[int, int] = (224, 224)

DATA_PROC_DIR   = Path("data/processed")
DATA_AUG_DIR    = Path("data/augmented")
SCREENSHOTS_DIR = Path("reports/screenshots/augmentation")

# DR class labels (Kaggle APTOS convention)
CLASS_LABELS: Dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}


# ── Augmentation pipeline ────────────────────────────────────────────────────

def get_train_augmentation_pipeline() -> A.Compose:
    """Return the training augmentation pipeline (strong augmentation).

    Returns:
        An :class:`albumentations.Compose` transform object.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.6),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10,
                             sat_shift_limit=20,
                             val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ISONoise(p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    ])


def get_val_augmentation_pipeline() -> A.Compose:
    """Return the validation/test augmentation pipeline (resize only).

    Returns:
        An :class:`albumentations.Compose` transform with only resizing.
    """
    return A.Compose([
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    ])


def augment_image(image: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """Apply an albumentations pipeline to a single image.

    Args:
        image:    RGB image, uint8, shape (H, W, 3).
        pipeline: An :class:`albumentations.Compose` transform.

    Returns:
        Augmented RGB image (uint8).
    """
    result = pipeline(image=image)
    return result["image"]


# ── Class-imbalance handling ─────────────────────────────────────────────────

def count_class_samples(data_dir: Path) -> Dict[int, int]:
    """Count the number of image samples per class in *data_dir*.

    Expects one subfolder per class named by its integer label (0–4).

    Args:
        data_dir: Root directory with per-class subfolders.

    Returns:
        Dict mapping class index → sample count.
    """
    image_extensions = {".jpg", ".jpeg", ".png"}
    counts: Dict[int, int] = {}
    for class_dir in sorted(data_dir.iterdir()):
        if class_dir.is_dir() and class_dir.name.isdigit():
            label = int(class_dir.name)
            counts[label] = sum(
                1 for f in class_dir.iterdir()
                if f.suffix.lower() in image_extensions
            )
    return counts


def augment_dataset(
    proc_dir: Path = DATA_PROC_DIR,
    out_dir: Path = DATA_AUG_DIR,
    target_per_class: int = 3000,
    pipeline: A.Compose | None = None,
) -> None:
    """Oversample minority classes using augmentation until *target_per_class*.

    Copies majority-class images as-is; augments minority classes to reach
    *target_per_class* samples each.

    Args:
        proc_dir:          Root dir with per-class preprocessed images.
        out_dir:           Root dir where balanced images are saved.
        target_per_class:  Desired number of samples per class after balancing.
        pipeline:          Albumentations pipeline (defaults to training pipeline).
    """
    if pipeline is None:
        pipeline = get_train_augmentation_pipeline()

    image_extensions = {".jpg", ".jpeg", ".png"}
    out_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(proc_dir.iterdir()):
        if not (class_dir.is_dir() and class_dir.name.isdigit()):
            continue
        label = class_dir.name
        out_class_dir = out_dir / label
        out_class_dir.mkdir(parents=True, exist_ok=True)

        images = [f for f in class_dir.iterdir()
                  if f.suffix.lower() in image_extensions]
        current_count = len(images)
        print(f"Class {label} ({CLASS_LABELS.get(int(label), '?')}): "
              f"{current_count} → {target_per_class}")

        # Copy originals
        for img_path in images:
            dest = out_class_dir / img_path.name
            if not dest.exists():
                cv2.imwrite(
                    str(dest),
                    cv2.imread(str(img_path)),
                )

        # Augment up to target
        needed = max(0, target_per_class - current_count)
        np.random.seed(RANDOM_STATE)
        for i in tqdm(range(needed), desc=f"Augmenting class {label}"):
            src_path = images[i % current_count]
            img = cv2.cvtColor(cv2.imread(str(src_path)), cv2.COLOR_BGR2RGB)
            aug = augment_image(img, pipeline)
            out_path = out_class_dir / f"aug_{i:05d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))

    print(f"Augmented dataset saved to {out_dir}")


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise_augmentations(
    image_path: str | Path,
    n_variants: int = 8,
    pipeline: A.Compose | None = None,
    out_dir: Path = SCREENSHOTS_DIR,
) -> None:
    """Plot a grid of augmented variants of a single image and save as PNG.

    Args:
        image_path: Path to a raw/preprocessed source image.
        n_variants: Number of augmented variants to display.
        pipeline:   Albumentations pipeline (defaults to training pipeline).
        out_dir:    Directory where the figure is saved.
    """
    if pipeline is None:
        pipeline = get_train_augmentation_pipeline()

    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    cols = 4
    rows = (n_variants + 1 + cols - 1) // cols  # +1 for original
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    np.random.seed(RANDOM_STATE)
    for idx in range(1, n_variants + 1):
        aug = augment_image(img.copy(), pipeline)
        axes[idx].imshow(aug)
        axes[idx].set_title(f"Variant {idx}", fontsize=9)
        axes[idx].axis("off")

    for ax in axes[n_variants + 1:]:
        ax.axis("off")

    plt.suptitle("Augmentation Variants", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = out_dir / "augmentation_grid.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_class_distribution(
    data_dir: Path,
    title: str = "Class Distribution",
    out_path: Path = Path("reports/screenshots/dataset/class_distribution.png"),
) -> None:
    """Plot and save a bar chart of per-class sample counts.

    Args:
        data_dir:  Root directory with per-class subfolders.
        title:     Chart title.
        out_path:  Destination path for the saved PNG.
    """
    import seaborn as sns

    counts = count_class_samples(data_dir)
    labels = [CLASS_LABELS.get(k, str(k)) for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]
    colours = sns.color_palette("husl", len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor="black", linewidth=0.5)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Diabetic Retinopathy Stage", fontsize=11)
    ax.set_ylabel("Number of Images", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    augment_dataset()
