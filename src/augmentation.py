"""
augmentation.py
===============
Data augmentation pipeline for the Diabetic Retinopathy Stage Detection project.

Dataset layout (APTOS 2019 Blindness Detection):
    Labels are read from ``data/processed/processed.csv`` (written by preprocessing.py),
    which has columns ``id_code``, ``diagnosis``, ``filepath``.

Augmentation strategies:
    - Geometric: horizontal/vertical flip, rotation, zoom (scale), shift
    - Photometric: brightness/contrast jitter, hue/saturation shift, gamma
    - Blur/noise: Gaussian blur, ISO noise
    - Advanced: GridDistortion (mimics retinal imaging artefacts)

Augmented images are written to ``data/augmented/<label>/`` so that
train.py can use ``flow_from_directory()`` on the balanced dataset.

Usage:
    python src/augmentation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import albumentations as A
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import config as cfg

# ── Constants ────────────────────────────────────────────────────────────────
RANDOM_STATE    = cfg.RANDOM_STATE
IMAGE_SIZE      = cfg.IMAGE_SIZE
CLASS_NAMES     = cfg.CLASS_NAMES
DATA_PROC_DIR   = cfg.DATA_PROC_DIR
DATA_AUG_DIR    = cfg.DATA_AUG_DIR
SCREENSHOTS_DIR = cfg.SS_AUGMENTATION
PROC_CSV_PATH   = cfg.DATA_PROC_DIR / "processed.csv"


# ── Augmentation pipelines ────────────────────────────────────────────────────

def get_train_augmentation_pipeline() -> A.Compose:
    """Return the training augmentation pipeline (strong augmentation).

    Returns:
        An :class:`albumentations.Compose` transform object.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.6),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ISONoise(p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    ])


def get_val_augmentation_pipeline() -> A.Compose:
    """Return the validation/test augmentation pipeline (resize only).

    Returns:
        An :class:`albumentations.Compose` with only resizing.
    """
    return A.Compose([A.Resize(IMAGE_SIZE[0], IMAGE_SIZE[1])])


def augment_image(image: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """Apply an albumentations pipeline to a single image.

    Args:
        image:    RGB image, uint8, shape (H, W, 3).
        pipeline: An :class:`albumentations.Compose` transform.

    Returns:
        Augmented RGB image (uint8).
    """
    return pipeline(image=image)["image"]


# ── CSV-based class counting ──────────────────────────────────────────────────

def count_class_samples_from_df(df: pd.DataFrame) -> Dict[int, int]:
    """Count samples per class from a DataFrame.

    Args:
        df: DataFrame with a ``diagnosis`` column (integer 0–4).

    Returns:
        Dict mapping class index → sample count.
    """
    return df["diagnosis"].value_counts().sort_index().to_dict()


# ── Dataset augmentation ──────────────────────────────────────────────────────

def augment_dataset(
    splits_csv: Path = DATA_PROC_DIR / "splits.csv",
    out_dir: Path = DATA_AUG_DIR,
    target_per_class: int = 1000,
    pipeline: Optional[A.Compose] = None,
) -> None:
    """Balance the training set to *target_per_class* images.

    Reads labels and splits from ``splits.csv``.
    - Train split: undersamples Class 0, keeps Class 2, oversamples others to target.
      Dropped images from undersampling are saved to ``<out_dir>/dropped/<label>/``.
    - Val/Test splits: copied exactly as they are (real-world distribution).
    
    Output is organised into ``<out_dir>/<split>/<label>/`` subfolders.

    Args:
        splits_csv:        Path to ``data/processed/splits.csv``.
        out_dir:           Root directory for augmented output.
        target_per_class:  Desired samples per class in the training set.
        pipeline:          Albumentations pipeline (defaults to training pipeline).
    """
    if not splits_csv.exists():
        raise FileNotFoundError(
            f"Splits CSV not found: {splits_csv}\n"
            "Run src/preprocessing.py and the EDA script first."
        )

    if pipeline is None:
        pipeline = get_train_augmentation_pipeline()

    df = pd.read_csv(splits_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name]
        
        for label in sorted(split_df["diagnosis"].unique()):
            class_df = split_df[split_df["diagnosis"] == label].reset_index(drop=True)
            class_name = CLASS_NAMES[label].replace(" ", "_")
            current_count = len(class_df)

            out_class_dir = out_dir / split_name / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)
            
            if split_name != "train":
                print(f"[{split_name.upper()}] Class {label} ({class_name}): {current_count} (no augmentation)")
                for _, row in class_df.iterrows():
                    src = Path(row["filepath"])
                    dest = out_class_dir / src.name
                    if not dest.exists() and src.exists():
                        cv2.imwrite(str(dest), cv2.imread(str(src)))
                continue

            # Train split logic
            print(f"[TRAIN] Class {label} ({class_name}): {current_count} -> {target_per_class}")
            
            # Undersampling (Class 0)
            if current_count > target_per_class:
                np.random.seed(RANDOM_STATE)
                keep_indices = np.random.choice(current_count, target_per_class, replace=False)
                keep_df = class_df.iloc[keep_indices]
                drop_df = class_df.drop(keep_indices)
                
                for _, row in keep_df.iterrows():
                    src = Path(row["filepath"])
                    dest = out_class_dir / src.name
                    if not dest.exists() and src.exists():
                        cv2.imwrite(str(dest), cv2.imread(str(src)))
                        
                drop_dir = out_dir / "dropped" / class_name
                drop_dir.mkdir(parents=True, exist_ok=True)
                for _, row in drop_df.iterrows():
                    src = Path(row["filepath"])
                    dest = drop_dir / src.name
                    if not dest.exists() and src.exists():
                        cv2.imwrite(str(dest), cv2.imread(str(src)))
                        
            else:
                for _, row in class_df.iterrows():
                    src = Path(row["filepath"])
                    dest = out_class_dir / src.name
                    if not dest.exists() and src.exists():
                        cv2.imwrite(str(dest), cv2.imread(str(src)))
                        
                # Oversampling
                needed = target_per_class - current_count
                if needed > 0:
                    np.random.seed(RANDOM_STATE)
                    for i in tqdm(range(needed), desc=f"Augmenting class {label}"):
                        row = class_df.iloc[i % current_count]
                        src_path = Path(row["filepath"])
                        if not src_path.exists():
                            continue
                        img = cv2.cvtColor(cv2.imread(str(src_path)), cv2.COLOR_BGR2RGB)
                        aug = augment_image(img, pipeline)
                        out_path = out_class_dir / f"aug_{i:05d}.png"
                        cv2.imwrite(str(out_path), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))

    print(f"Dataset saved to {out_dir}")


# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise_augmentations(
    image_path: str | Path,
    n_variants: int = 8,
    pipeline: Optional[A.Compose] = None,
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
    rows = (n_variants + 1 + cols - 1) // cols
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
    df: pd.DataFrame,
    title: str = "Class Distribution (APTOS 2019)",
    out_path: Path = cfg.SS_DATASET / "class_distribution.png",
) -> None:
    """Plot and save a bar chart of per-class sample counts from a DataFrame.

    Args:
        df:       DataFrame with a ``diagnosis`` column (integer 0–4).
        title:    Chart title.
        out_path: Destination path for the saved PNG.
    """
    counts = count_class_samples_from_df(df)
    labels = [CLASS_NAMES[k] for k in sorted(counts)]
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
