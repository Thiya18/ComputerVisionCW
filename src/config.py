"""
config.py
=========
Central configuration for all paths and shared hyperparameters.

Edit this file to point to your local dataset location.
All other modules import their paths and constants from here.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Dataset paths  ← EDIT THESE if your files live elsewhere
# ─────────────────────────────────────────────────────────────────────────────

# Root folder of the APTOS 2019 dataset (flat structure from Kaggle download)
DATASET_ROOT: Path = Path("data/raw")

# CSV file: columns `id_code` and `diagnosis`
CSV_PATH: Path = DATASET_ROOT / "train.csv"

# Flat folder of all raw retinal images (no sub-directories)
IMAGES_DIR: Path = DATASET_ROOT / "train_images"

# ─────────────────────────────────────────────────────────────────────────────
# Derived / intermediate paths  (inside the project — gitignored)
# ─────────────────────────────────────────────────────────────────────────────

DATA_PROC_DIR:  Path = Path("data/processed")   # preprocessed images
DATA_AUG_DIR:   Path = Path("data/augmented")   # augmented + balanced images
CHECKPOINT_DIR: Path = Path("checkpoints")
FINAL_MODEL_PATH: Path = CHECKPOINT_DIR / "final_model.keras"
LOG_DIR:        Path = Path("logs")
HISTORY_DIR:    Path = Path("reports")

# ─────────────────────────────────────────────────────────────────────────────
# Screenshot output directories
# ─────────────────────────────────────────────────────────────────────────────

SS_DATASET:       Path = Path("reports/screenshots/dataset")
SS_PREPROCESSING: Path = Path("reports/screenshots/preprocessing")
SS_AUGMENTATION:  Path = Path("reports/screenshots/augmentation")
SS_TRAINING:      Path = Path("reports/screenshots/training")
SS_EVALUATION:    Path = Path("reports/screenshots/evaluation")

# ─────────────────────────────────────────────────────────────────────────────
# Image settings
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_SIZE: tuple[int, int] = (224, 224)       # (height, width) for the model

def detect_image_extension(images_dir: Path = IMAGES_DIR) -> str:
    """Auto-detect the file extension used in *images_dir*.

    Checks for the first file with a recognised image extension.
    Falls back to ``.png`` (APTOS 2019 default) if the directory is not
    yet available (e.g. before extraction).

    Args:
        images_dir: Directory containing the flat image files.

    Returns:
        Extension string including the dot, e.g. ``".png"`` or ``".jpeg"``.
    """
    recognised = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    if images_dir.exists():
        for f in images_dir.iterdir():
            if f.suffix.lower() in recognised:
                return f.suffix.lower()
    # APTOS 2019 Blindness Detection uses PNG
    return ".png"

IMAGE_EXT: str = detect_image_extension()

# ─────────────────────────────────────────────────────────────────────────────
# Dataset split ratios
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_RATIO: float = 0.70
VAL_RATIO:   float = 0.15
TEST_RATIO:  float = 0.15    # 1 - TRAIN - VAL

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE:       int   = 32
EPOCHS_FROZEN:    int   = 30
EPOCHS_FINETUNE:  int   = 20
RANDOM_STATE:     int   = 42

# ─────────────────────────────────────────────────────────────────────────────
# ImageNet normalisation constants
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────────────────────────────────────
# Class metadata
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES: int = 5
CLASS_NAMES: list[str] = [
    "No DR",          # 0
    "Mild",           # 1
    "Moderate",       # 2
    "Severe",         # 3
    "Proliferative DR",  # 4
]
