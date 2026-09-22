# Diabetic Retinopathy Stage Detection
### Computer Vision CW1 — BSc (Hons) Computer Science, BSCCOMP24.2P

A deep learning pipeline that classifies **diabetic retinopathy severity** (5-class) from
fundus photographs using transfer learning on a Kaggle dataset.

---

## Project Structure

```
ComputerVisionCW/
├── data/                        # ← gitignored; never committed
│   ├── raw/                     # Original Kaggle download
│   └── processed/               # Preprocessed & augmented images
├── notebooks/                   # Exploratory analysis & step-by-step demos
├── src/                         # Production-quality Python modules
│   ├── preprocessing.py         # Contrast, resizing, normalisation, edge enhancement
│   ├── augmentation.py          # Rotation, flip, zoom, brightness augmentation
│   ├── model.py                 # CNN + transfer learning architecture
│   ├── train.py                 # Training loop, callbacks, LR scheduling
│   └── evaluate.py              # Metrics, confusion matrix, curves
├── app/                         # Gradio / Streamlit demo UI (Week 2+)
├── reports/
│   └── screenshots/             # Evidence for the PDF report
│       ├── dataset/
│       ├── preprocessing/
│       ├── augmentation/
│       ├── training/
│       └── evaluation/
├── docs/
│   ├── assignment_brief.md      # Full rubric & module info
│   └── assignment_brief.docx
├── AGENTS.md                    # Context for AI coding assistants
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/Thiya18/ComputerVisionCW.git
cd ComputerVisionCW
pip install -r requirements.txt

# 2. Download dataset (requires Kaggle API key)
kaggle datasets download -d tanlikesmath/diabetic-retinopathy-resized
unzip diabetic-retinopathy-resized.zip -d data/raw/

# 3. Preprocess
python src/preprocessing.py

# 4. Train
python src/train.py

# 5. Evaluate
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
