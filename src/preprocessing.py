"""
preprocessing.py
================
Image preprocessing pipeline for the Diabetic Retinopathy Stage Detection project.

Dataset layout (APTOS 2019 Blindness Detection):
    <IMAGES_DIR>/         — flat folder of all retinal images
    <CSV_PATH>            — CSV with columns `id_code` and `diagnosis` (0-4)

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

import sys
from pathlib import Path

# Allow sibling imports when running as __main__
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Tuple

import config as cfg

# ── Constants (re-exported from config for backward compatibility) ─────────────
IMAGE_SIZE      = cfg.IMAGE_SIZE
IMAGENET_MEAN   = np.array(cfg.IMAGENET_MEAN, dtype=np.float32)
IMAGENET_STD    = np.array(cfg.IMAGENET_STD,  dtype=np.float32)
CSV_PATH        = cfg.CSV_PATH
IMAGES_DIR      = cfg.IMAGES_DIR
DATA_PROC_DIR   = cfg.DATA_PROC_DIR
SCREENSHOTS_DIR = cfg.SS_PREPROCESSING


# ── CSV / DataFrame helpers ───────────────────────────────────────────────────

def load_dataframe(
    csv_path: Path = CSV_PATH,
    images_dir: Path = IMAGES_DIR,
    ext: Optional[str] = None,
) -> pd.DataFrame:
    """Load ``train.csv`` and attach a resolved ``filepath`` column.

    Args:
        csv_path:   Path to the CSV file (columns: ``id_code``, ``diagnosis``).
        images_dir: Directory containing the flat image files.
        ext:        Image extension including the dot (e.g. ``".png"``).
                    Auto-detected from *images_dir* when ``None``.

    Returns:
        DataFrame with columns: ``id_code``, ``diagnosis``, ``filepath``,
        ``label`` (string class name).

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Download APTOS 2019 from Kaggle and place train.csv at that path."
        )

    if ext is None:
        ext = cfg.detect_image_extension(images_dir)

    df = pd.read_csv(csv_path)
    df.rename(columns={"diagnosis": "diagnosis"}, inplace=True)  # normalise
    df["filepath"] = df["id_code"].apply(
        lambda code: str(images_dir / f"{code}{ext}")
    )
    df["label"] = df["diagnosis"].map(
        {i: name for i, name in enumerate(cfg.CLASS_NAMES)}
    )
    return df


# ── Core image functions ──────────────────────────────────────────────────────

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
    colour distortion while enhancing contrast in fundus photographs.

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
    return cv2.addWeighted(image, 1 + amount, blurred, -amount, 0)


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
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


def preprocess_image(
    path: str | Path,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
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


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_before_after(
    original_path: str | Path,
    out_dir: Path = SCREENSHOTS_DIR,
    filename: Optional[str] = None,
) -> None:
    """Save a side-by-side before/after preprocessing figure as PNG.

    Args:
        original_path: Path to the raw input image.
        out_dir:       Directory where the figure is saved.
        filename:      Output filename stem (defaults to stem of *original_path*).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    original = load_image(original_path)
    original_resized = resize_image(original)
    enhanced = sharpen_edges(remove_noise(apply_clahe(original_resized)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_resized)
    axes[0].set_title("Original (resized)", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(enhanced)
    axes[1].set_title("After CLAHE + denoise + sharpen", fontsize=11)
    axes[1].axis("off")
    plt.suptitle("Preprocessing Pipeline", fontsize=14, fontweight="bold")
    plt.tight_layout()

    stem = filename or Path(original_path).stem
    save_path = out_dir / f"{stem}_before_after.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Dataset-level processing ──────────────────────────────────────────────────

def preprocess_dataset(
    csv_path: Path = CSV_PATH,
    images_dir: Path = IMAGES_DIR,
    out_dir: Path = DATA_PROC_DIR,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> None:
    """Preprocess every image listed in *csv_path* and save to *out_dir*.

    Reads labels from ``train.csv`` (APTOS 2019 layout) rather than
    inferring them from sub-directory names.  Saves preprocessed images
    as ``<out_dir>/<id_code>.png`` and writes a ``processed.csv`` with
    columns ``id_code``, ``diagnosis``, ``filepath`` for downstream use.

    Args:
        csv_path:   Path to the CSV file.
        images_dir: Flat folder containing raw retinal images.
        out_dir:    Directory where preprocessed images are saved.
        size:       Target (width, height) for resizing.
    """
    df = load_dataframe(csv_path, images_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
        src_path = Path(row["filepath"])
        dest_path = out_dir / f"{row['id_code']}.png"

        try:
            processed = preprocess_image(src_path, size)
            # De-normalise back to uint8 for disk storage
            img_uint8 = (
                (processed * IMAGENET_STD + IMAGENET_MEAN) * 255
            ).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                str(dest_path),
                cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
            )
            processed_rows.append({
                "id_code":   row["id_code"],
                "diagnosis": row["diagnosis"],
                "filepath":  str(dest_path),
            })
        except Exception as exc:
            print(f"Skipping {src_path.name}: {exc}")

    # Save a processed.csv so augmentation.py can pick up labels directly
    out_csv = out_dir / "processed.csv"
    pd.DataFrame(processed_rows).to_csv(out_csv, index=False)
    print(f"Done. {len(processed_rows)} images saved to {out_dir}")
    print(f"Label CSV written to {out_csv}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    preprocess_dataset()
