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
using transfer learning on a Kaggle fundus photography dataset.

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

- **Framework:** TensorFlow / Keras (preferred for deployment simplicity)
- **Base model:** EfficientNetB3 (best accuracy/size trade-off for medical imaging)
- **Preprocessing:** CLAHE contrast enhancement → BGR→RGB → resize to 224×224 → ImageNet normalisation
- **Augmentation library:** `albumentations` (fast, composable, reproducible)
- **Class imbalance strategy:** Class weights + augmentation of minority classes
- **Training:** Adam optimiser, ReduceLROnPlateau, EarlyStopping (patience=10), ModelCheckpoint
- **Evaluation:** sklearn classification_report, seaborn heatmap confusion matrix

---

## Module Map

| File | Responsibility |
|---|---|
| `src/preprocessing.py` | `load_image()`, `apply_clahe()`, `resize_and_normalise()`, `preprocess_dataset()` |
| `src/augmentation.py` | `get_augmentation_pipeline()`, `augment_dataset()`, `visualise_augmentations()` |
| `src/model.py` | `build_model()`, `unfreeze_top_layers()`, `get_callbacks()` |
| `src/train.py` | `load_data()`, `compute_class_weights()`, `train()`, `save_history()` |
| `src/evaluate.py` | `evaluate_model()`, `plot_curves()`, `plot_confusion_matrix()`, `generate_report()` |
| `app/` | Gradio/Streamlit demo UI — to be built after training |
| `notebooks/` | Step-by-step EDA and experiment notebooks |

---

## Dataset

- **Source:** Kaggle — APTOS 2019 Blindness Detection or Diabetic Retinopathy Resized
- **Classes:** 0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR
- **Split:** 70% train / 15% validation / 15% test (stratified)
- **Location:** `data/raw/` (gitignored — never commit images)

---

## Coding Standards

- Python 3.10+
- Type hints on all public functions
- Google-style docstrings on every function
- `random_state=42` for all stochastic operations (reproducibility)
- Save all figures to `reports/screenshots/<subfolder>/` as PNG
- Log training progress with `tqdm`

---

## Report Evidence Checklist

Screenshots must be saved to `reports/screenshots/` subfolders:

- [ ] `dataset/` — class distribution bar chart, sample images per class
- [ ] `preprocessing/` — before/after CLAHE, before/after normalisation
- [ ] `augmentation/` — grid of augmented samples
- [ ] `training/` — accuracy curve, loss curve, LR schedule
- [ ] `evaluation/` — confusion matrix, classification report, ROC curves
