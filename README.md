---
title: DR Stage Detection
emoji: 👁️
colorFrom: blue
colorTo: indigo
sdk: streamlit
app_file: app/streamlit_app.py
pinned: false
---

# Diabetic Retinopathy Stage Detection
### Computer Vision CW1 — BSc (Hons) Computer Science, BSCCOMP24.2P

A deep learning pipeline that classifies **diabetic retinopathy severity** (5-class) from
fundus photographs using transfer learning on the APTOS 2019 Blindness Detection dataset.

---

## Project Structure

```
ComputerVisionCW/
├── data/                        # ← gitignored; never committed
│   ├── processed/               # Preprocessed images + processed.csv
│   └── augmented/               # Balanced augmented images (per-class subfolders)
├── notebooks/                   # Exploratory analysis & step-by-step demos
├── src/                         # Production-quality Python modules
│   ├── config.py                # ← All paths & hyperparameters (edit here first)
│   ├── preprocessing.py         # Reads train.csv; CLAHE → denoise → sharpen → normalise
│   ├── augmentation.py          # Reads processed.csv; albumentations balancing
│   ├── model.py                 # EfficientNetB3 + custom head + callbacks
│   ├── train.py                 # Two-stage training (frozen → fine-tune)
│   └── evaluate.py              # Metrics, confusion matrix, ROC curves
├── app/                         # Gradio / Streamlit demo UI (Week 2+)
├── reports/
│   └── screenshots/             # Evidence for the PDF report
│       ├── dataset/             # Class distribution chart
│       ├── preprocessing/       # Before/after CLAHE figures
│       ├── augmentation/        # Augmentation grids
│       ├── training/            # Accuracy & loss curves
│       └── evaluation/          # Confusion matrix, ROC curves, classification report
├── docs/
│   ├── assignment_brief.md      # Full rubric & module info
│   └── assignment_brief.docx
├── AGENTS.md                    # Context for AI coding assistants
├── requirements.txt
└── README.md
```

---

## Dataset

**APTOS 2019 Blindness Detection** (manual Kaggle download)
