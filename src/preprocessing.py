"""
preprocessing.py
================
Image preprocessing pipeline for the Diabetic Retinopathy Stage Detection project.

Steps applied per image:
    1. Load image (BGR → RGB)
    2. Resize to target dimensions
    3. CLAHE contrast enhancement (applied to L channel in LAB colour space)
    4. Gaussian blur for noise removal
    5. Unsharp mask for edge enhancement
    6. Normalise to [0, 1] using ImageNet mean/std

Usage:
    python src/preprocessing.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional

# ── Constants ────────────────────────────────────────────────────────────────
IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DATA_RAW_DIR  = Path("data/raw")
DATA_PROC_DIR = Path("data/processed")
SCREENSHOTS_DIR = Path("reports/screenshots/preprocessing")


# ── Core functions ────────────────────────────────────────────────────────────

def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk and convert BGR → RGB.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        RGB image as a uint8 numpy array of shape (H, W, 3).

    Raises:
        FileNotFoundError: If the image cannot be read from *path*.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Operates on the L (lightness) channel in LAB colour space to avoid
    colour distortion.

    Args:
        image:          RGB image, uint8, shape (H, W, 3).
        clip_limit:     Threshold for contrast limiting (default 2.0).
        tile_grid_size: Size of the grid for histogram equalisation.

    Returns:
        Contrast-enhanced RGB image (uint8).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def remove_noise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Gaussian blur for noise reduction.

    Args:
        image:       RGB image, uint8.
        kernel_size: Size of the Gaussian kernel (must be odd).

    Returns:
        Noise-reduced RGB image (uint8).
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=0)


def sharpen_edges(image: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """Enhance edges via unsharp masking.

    Args:
        image:  RGB image, uint8.
        amount: Strength of sharpening (default 1.5).

    Returns:
        Edge-enhanced RGB image (uint8).
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return sharpened


def resize_image(image: np.ndarray, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize image to the target dimensions using Lanczos interpolation.

    Args:
        image: RGB image, uint8.
        size:  Target (width, height) tuple.

    Returns:
        Resized RGB image (uint8).
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)


def normalise(image: np.ndarray) -> np.ndarray:
    """Normalise pixel values using ImageNet mean and std.

    Converts uint8 [0, 255] → float32 [0, 1] then applies
    per-channel standardisation.

    Args:
        image: RGB image, uint8, shape (H, W, 3).

    Returns:
        Normalised float32 array of shape (H, W, 3).
    """
    img = image.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img


def preprocess_image(path: str | Path, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Full preprocessing pipeline for a single image.

    Pipeline: load → resize → CLAHE → denoise → sharpen → normalise

    Args:
        path: Path to the input image.
        size: Target (width, height).

    Returns:
        Preprocessed float32 numpy array of shape (*size, 3).
    """
    img = load_image(path)
    img = resize_image(img, size)
    img = apply_clahe(img)
    img = remove_noise(img)
    img = sharpen_edges(img)
    img = normalise(img)
    return img


def save_before_after(
    original_path: str | Path,
    out_dir: Path = SCREENSHOTS_DIR,
    filename: Optional[str] = None,
) -> None:
    """Save a side-by-side before/after preprocessing figure as PNG.

    Args:
        original_path: Path to the raw input image.
        out_dir:       Directory where the figure is saved.
        filename:      Output filename (defaults to stem of *original_path*).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    original = load_image(original_path)
    original_resized = resize_image(original)
    enhanced = apply_clahe(original_resized)
    enhanced = remove_noise(enhanced)
    enhanced = sharpen_edges(enhanced)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_resized)
    axes[0].set_title("Original (resized)")
    axes[0].axis("off")
    axes[1].imshow(enhanced)
    axes[1].set_title("After CLAHE + denoise + sharpen")
    axes[1].axis("off")
    plt.suptitle("Preprocessing Pipeline", fontsize=14, fontweight="bold")
    plt.tight_layout()

    stem = filename or Path(original_path).stem
    save_path = out_dir / f"{stem}_before_after.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def preprocess_dataset(
    raw_dir: Path = DATA_RAW_DIR,
    out_dir: Path = DATA_PROC_DIR,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> None:
    """Preprocess all images in *raw_dir* and save to *out_dir*.

    Preserves the subdirectory structure (one subfolder per class label).

    Args:
        raw_dir: Root directory containing per-class subfolders of raw images.
        out_dir: Root directory where preprocessed images are saved.
        size:    Target image size.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    paths = [
        p for p in raw_dir.rglob("*")
        if p.suffix.lower() in image_extensions
    ]

    if not paths:
        print(f"No images found in {raw_dir}. "
              "Download the dataset first: kaggle datasets download ...")
        return

    for path in tqdm(paths, desc="Preprocessing"):
        relative = path.relative_to(raw_dir)
        dest = out_dir / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            processed = preprocess_image(path, size)
            # De-normalise for saving (save as uint8 PNG)
            img_uint8 = ((processed * IMAGENET_STD + IMAGENET_MEAN) * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(dest.with_suffix(".png")), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            print(f"Skipping {path}: {exc}")

    print(f"Done. Preprocessed images saved to {out_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocess_dataset()
