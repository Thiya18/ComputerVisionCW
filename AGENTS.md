# AGENTS.md — AI Coding Assistant Context

This file provides structured context for AI assistants (GitHub Copilot, Antigravity, etc.)
working on this codebase.

---

## Project Overview

**Title:** Diabetic Retinopathy Stage Detection
**Module:** Computer Vision CW1 — BSc (Hons) Computer Science, BSCCOMP24.2P
**Deadline:** 2026-10-03 (confirm with lecturer)
**Submission:** PDF report (≤ 20 pages) + codebase via Turnitin

The goal is to build a **5-class CNN classifier** (No DR / Mild / Moderate / Severe / Proliferative DR)
using transfer learning on the APTOS 2019 Blindness Detection dataset.

---

## Dataset Layout (APTOS 2019 Blindness Detection)

> **Source:** https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
> (Manual download — NOT via the Kaggle CLI datasets command)

| File / Folder | Location | Description |
|---|---|---|
| `train.csv` | `C:\Users\User\Downloads\train.csv` | Columns: `id_code` (filename stem), `diagnosis` (0–4) |
| `train_images/` | `C:\Users\User\Downloads\train_images\` | **Flat** folder of PNG fundus images — no sub-directories |

- **Classes:** 0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR
- **Split:** 70% train / 15% validation / 15% test (stratified by label)
- **Images:** Flat structure — labels come from CSV only, NOT from folder names

> ⚠️ **The `data/` folder is gitignored.** Never commit images or CSV files into the repo.
> All paths are configured in `src/config.py`.

---

## Rubric Summary (100 marks)

| # | Component | Marks | Key requirement |
|---|---|---|---|
| 1 | Problem Understanding & Dataset Justification | 10 | Explain DR, classes, splits, ethical concerns |
| 2 | Data Preprocessing | 10 | CLAHE, resize, normalise, edge enhance — all justified |
| 3 | Data Augmentation & Balancing | 10 | Rotation, flip, zoom, brightness; handle class imbalance |
| 4 | CNN + Transfer Learning | **20** | ResNet/EfficientNet/MobileNet; fine-tuning; hyperparameter tuning |
| 5 | Training Strategy | 10 | Callbacks, early stopping, LR scheduling, validation splits |
| 6 | Model Evaluation | 15 | Accuracy, precision, recall, F1, confusion matrix, curves |
| 7 | Code Quality | 10 | Modular, commented, reproducible |
| 8 | Report Quality | 10 | ≤ 20 pages, graphs, screenshots, hosted video URL |
| 9 | Innovation & Impact | 5 | Real-world impact, limitations, future work |

---

## Architecture Decisions

- **Framework:** TensorFlow / Keras
- **Base model:** EfficientNetB3 (best accuracy/size trade-off for medical imaging)
- **Preprocessing:** CLAHE → BGR→RGB → resize 224×224 → ImageNet normalisation
- **Augmentation library:** `albumentations` (fast, composable, reproducible)
- **Class imbalance strategy:** Class weights + augmentation of minority classes to 3000 samples each
- **Training:** Adam optimiser, ReduceLROnPlateau, EarlyStopping (patience=10), ModelCheckpoint
- **Evaluation:** sklearn classification_report, seaborn heatmap confusion matrix

---

## Module Map

| File | Responsibility |
|---|---|
| `src/config.py` | **Single source of truth** for all paths and hyperparameters |
| `src/preprocessing.py` | `load_dataframe()`, `apply_clahe()`, `preprocess_image()`, `preprocess_dataset()` |
| `src/augmentation.py` | `get_train_augmentation_pipeline()`, `augment_dataset()`, `plot_class_distribution()` |
| `src/model.py` | `build_model()`, `unfreeze_top_layers()`, `get_callbacks()` |
| `src/train.py` | `load_data()`, `compute_class_weights()`, `train()`, `save_history()` |
| `src/evaluate.py` | `evaluate_model()`, `plot_curves()`, `plot_confusion_matrix()`, `plot_roc_curves()` |
| `app/` | Gradio/Streamlit demo UI — to be built after training |
| `notebooks/` | Step-by-step EDA and experiment notebooks |

---

## Data Flow

```
train.csv + train_images/  (C:\Users\User\Downloads\)
        │
        ▼
src/preprocessing.py   →  data/processed/*.png  +  data/processed/processed.csv
        │
        ▼
src/augmentation.py    →  data/augmented/<label>/*.png   (balanced, per-class subfolders)
        │
        ▼
src/train.py           →  checkpoints/best_model.keras
        │
        ▼
src/evaluate.py        →  reports/screenshots/evaluation/*.png
```

---

## Coding Standards

- Python 3.10+
- Type hints on all public functions
- Google-style docstrings on every function
- `random_state=42` for all stochastic operations (see `config.RANDOM_STATE`)
- Import `config as cfg` at the top of every module — never hardcode paths
- Save all figures to `reports/screenshots/<subfolder>/` as PNG
- Log training progress with `tqdm`

---

## Report Evidence Checklist

Screenshots must be saved to `reports/screenshots/` subfolders:

- [ ] `dataset/` — class distribution bar chart (from `train.csv`), sample images per class
- [ ] `preprocessing/` — before/after CLAHE, before/after normalisation
- [ ] `augmentation/` — grid of augmented samples
- [ ] `training/` — accuracy curve, loss curve, LR schedule
- [ ] `evaluation/` — confusion matrix, classification report, ROC curves
