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

| File | Description |
|---|---|
| `train.csv` | `id_code` (filename stem) + `diagnosis` (label 0–4) |
| `train_images/` | Flat folder of PNG retinal fundus photographs |

> **Important:** These files live outside the repo (e.g. `C:\Users\User\Downloads\`).
> Update `src/config.py` → `DATASET_ROOT` if your path differs.
> **Never copy images into the repo** — `data/` is gitignored.

### DR Class Labels

| Code | Stage |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/Thiya18/ComputerVisionCW.git
cd ComputerVisionCW
pip install -r requirements.txt

# 2. Download dataset from Kaggle
#    https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
#    → Download train.csv and train_images.zip
#    → Extract train_images.zip so train_images/ is a flat folder of PNGs

# 3. Edit config.py if your download path differs from the default
#    DATASET_ROOT = Path(r"C:\Users\User\Downloads")

# 4. Preprocess  (reads train.csv, writes data/processed/ + processed.csv)
python src/preprocessing.py

# 5. Augment & balance  (reads processed.csv, writes data/augmented/<label>/)
python src/augmentation.py

# 6. Train
python src/train.py

# 7. Evaluate
python src/evaluate.py
```

---

## Rubric at a Glance

| Component | Marks |
|---|---|
| Problem Understanding & Dataset Justification | 10 |
| Data Preprocessing Techniques | 10 |
| Data Augmentation & Dataset Balancing | 10 |
| CNN Architecture & Transfer Learning | **20** |
| Training Strategy & Experimental Design | 10 |
| Model Evaluation & Performance Analysis | 15 |
| Code Quality & Documentation | 10 |
| Report Quality & Presentation | 10 |
| Innovation & Practical Impact | 5 |
| **Total** | **100** |

See [`docs/assignment_brief.md`](docs/assignment_brief.md) for the full rubric.

---

## Submission

- **Deadline:** 2026-10-03 (confirm with lecturer)
- **Format:** PDF report (≤ 20 pages) + codebase via Turnitin
- Video demo hosted and URL included in the report
