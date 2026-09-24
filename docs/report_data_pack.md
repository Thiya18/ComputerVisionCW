# Report Data Pack — Diabetic Retinopathy Stage Detection
> Reference material only. Raw facts, numbers, parameters, paths.
> Do NOT copy as report prose — write in your own words.

---

## 1. Dataset

### Source
- **Name:** APTOS 2019 Blindness Detection
- **Host:** Kaggle competition
- **URL:** https://www.kaggle.com/competitions/aptos2019-blindness-detection
- **Format:** Flat folder of PNG retinal fundus images + `train.csv`
- **CSV columns:** `id_code` (filename stem), `diagnosis` (integer 0–4)
- **Image structure:** Single flat directory — no class subfolders; labels come from CSV only

### Class Definitions
| Label | Class Name | Diagnosis |
|---|---|---|
| 0 | No DR | No diabetic retinopathy |
| 1 | Mild | Mild NPDR |
| 2 | Moderate | Moderate NPDR |
| 3 | Severe | Severe NPDR |
| 4 | Proliferative DR | PDR |

### Raw Class Distribution (full dataset, n=3662)
| Class | Count | % of total |
|---|---|---|
| No DR | 1805 | 49.3% |
| Moderate | 999 | 27.3% |
| Mild | 370 | 10.1% |
| Proliferative DR | 295 | 8.1% |
| Severe | 193 | 5.3% |
| **Total** | **3662** | 100% |

- Severe class is ~9× underrepresented vs No DR
- No DR is ~49% of dataset (majority class)

### Train / Val / Test Split
- **Ratios:** 70% train / 15% val / 15% test (configured in `src/config.py`)
- **Method:** Stratified by label (preserves class proportions in each split)
- **Random seed:** 42
- **Approximate counts** (from 3662 total):

| Split | Approx. count |
|---|---|
| Train | ~2563 |
| Val | ~549 |
| Test | **550** (exact — confirmed from classification report support column) |

- Test set support per class (confirmed from `classification_report.txt`):

| Class | Test support |
|---|---|
| Mild | 56 |
| Moderate | 150 |
| No_DR | 271 |
| Proliferative_DR | 44 |
| Severe | 29 |
| **Total** | **550** |

### Ethical Considerations (evidence points)
- Medical imaging data — patient privacy concern
- Class imbalance could lead to biased predictions against minority classes (Severe, Proliferative)
- Model output is NOT a medical diagnosis — educational tool only (enforced in UI)
- Dataset is anonymised (Kaggle competition data, no patient identifiers)

---

## 2. Preprocessing

### Pipeline Order (defined in `src/preprocessing.py`, `preprocess_image()`)
1. Load image → convert BGR → RGB (`cv2.imread` + `cv2.cvtColor`)
2. Resize to 224×224 (`cv2.resize`, interpolation: `INTER_LANCZOS4`)
3. CLAHE contrast enhancement (`apply_clahe()`)
4. Gaussian blur / noise removal (`remove_noise()`)
5. Unsharp masking / edge sharpening (`sharpen_edges()`)
6. ImageNet normalisation (`normalise()`)

### Step-by-Step Parameters (exact values from source)

**CLAHE** (`apply_clahe()`, `src/preprocessing.py` lines 109–132):
- Colour space: RGB → LAB, operates on **L channel only** (avoids colour distortion)
- `clip_limit = 2.0`
- `tile_grid_size = (8, 8)`
- Output: converted back to RGB

**Resize** (`resize_image()`, lines 162–175):
- Target: `(224, 224)` pixels — matches EfficientNetB3 input requirement
- Interpolation: `cv2.INTER_LANCZOS4` (high-quality downsampling)

**Noise removal** (`remove_noise()`, lines 135–145):
- Method: Gaussian blur
- `kernel_size = 3` (3×3 kernel)
- `sigmaX = 0` (auto-computed from kernel size)

**Edge sharpening** (`sharpen_edges()`, lines 148–159):
- Method: Unsharp masking
- Blur for mask: `cv2.GaussianBlur(image, (0,0), sigmaX=3)`
- `amount = 1.5`
- Formula: `addWeighted(image, 1+1.5, blurred, -1.5, 0)` i.e. `2.5 × original − 1.5 × blurred`

**Normalisation** (`normalise()`, lines 178–192):
- Scale: uint8 [0,255] → float32 [0,1] (`/ 255.0`)
- Per-channel mean subtraction + std division using **ImageNet statistics**:
  - Mean: `[0.485, 0.456, 0.406]` (R, G, B)
  - Std:  `[0.229, 0.224, 0.225]` (R, G, B)
- Output dtype: `float32`

### Justification for Each Step
- **CLAHE:** Fundus images often have uneven illumination; CLAHE improves local contrast without over-brightening (clip limit prevents noise amplification)
- **Resize 224×224:** Required input size for EfficientNetB3
- **Lanczos4:** Preferred over bilinear for medical images — preserves fine vessel detail during downsampling
- **Gaussian blur:** Reduces sensor/JPEG noise before feature extraction
- **Unsharp mask:** Enhances microaneurysm and vessel boundaries — clinically relevant features
- **ImageNet normalisation:** EfficientNetB3 was pretrained with ImageNet stats; consistent normalisation is required for transfer learning to work correctly

### Before/After Screenshot Paths
| File | Path |
|---|---|
| Sample 1 | `reports/screenshots/preprocessing/000c1434d8d7_before_after.png` |
| Sample 2 | `reports/screenshots/preprocessing/001639a390f0_before_after.png` |
| Sample 3 | `reports/screenshots/preprocessing/0024cdab0c1e_before_after.png` |
| Sample 4 | `reports/screenshots/preprocessing/002c21358ce6_before_after.png` |
| Sample 5 | `reports/screenshots/preprocessing/0104b032c141_before_after.png` |

- Each image: side-by-side pair — left = original resized, right = after CLAHE + denoise + sharpen
- DPI: 150, saved as PNG

---

## 3. Augmentation & Class Balancing

### Strategy
- **Target:** 1000 images per class in the **training split only**
- **No DR (class 0):** Undersample from ~1263 train images → keep 1000 (random, seed=42; dropped images saved to `data/augmented/dropped/No_DR/`)
- **Moderate (class 2):** ~699 train images — copy all, augment remainder to reach 1000
- **Mild, Severe, Proliferative DR:** Significantly underrepresented → oversample by generating augmented variants (cycle through existing images, apply random transform per image)
- **Val and Test splits:** Copied exactly as-is (no augmentation, real-world distribution preserved)

### Augmentation Techniques (from `src/augmentation.py`, `get_train_augmentation_pipeline()`)
Using **albumentations** library:

| Transform | Parameters | Probability |
|---|---|---|
| `HorizontalFlip` | — | p=0.5 |
| `VerticalFlip` | — | p=0.3 |
| `Rotate` | limit=±30° | p=0.6 |
| `ShiftScaleRotate` | shift=0.05, scale=0.1, rotate=±15° | p=0.5 |
| `RandomBrightnessContrast` | brightness±0.2, contrast±0.2 | p=0.5 |
| `HueSaturationValue` | hue±10, sat±20, val±10 | p=0.3 |
| `GaussianBlur` | blur kernel (3,5) | p=0.2 |
| `ISONoise` | default params | p=0.2 |
| `GridDistortion` | num_steps=5, distort_limit=0.1 | p=0.2 |
| `RandomGamma` | gamma (80,120) | p=0.3 |
| `Resize` | 224×224 (final resize to ensure consistent output) | always |

- Val/test pipeline: `Resize(224, 224)` only
- Random seed: `42` for reproducibility (`np.random.seed(42)`)

### Justification
- Horizontal/vertical flip: retinal images have no anatomical orientation constraint
- Rotation ±30°: camera angle variation in clinical imaging
- Brightness/contrast jitter: mimics variation in fundus camera illumination
- GridDistortion: simulates optical distortion in imaging equipment
- ISONoise: mimics low-light sensor noise
- Hue/saturation: accounts for different imaging devices and dye contrast variation

### Output Directory Structure
```
data/augmented/
  train/
    Mild/             (1000 images)
    Moderate/         (1000 images)
    No_DR/            (1000 images)
    Proliferative_DR/ (1000 images)
    Severe/           (1000 images)
  val/
    (real distribution, no augmentation)
  test/
    (real distribution, 550 total)
  dropped/
    No_DR/            (excess No_DR samples removed from train)
```

### Screenshot Paths
| File | Path |
|---|---|
| Class distribution bar chart | `reports/screenshots/dataset/class_distribution.png` |
| Augmentation variant grid | `reports/screenshots/augmentation/augmentation_grid.png` |

---

## 4. Model Architecture

### Base Model
- **Name:** EfficientNetB3
- **Pretrained weights:** ImageNet
- **include_top:** False (original classifier removed)
- **Input shape:** (224, 224, 3)
- **Total base params:** ~10.7M (frozen in Stage 1)
- **EfficientNetB3 chosen over alternatives because:**
  - Better accuracy/parameter trade-off than ResNet50 and MobileNetV2
  - Compound scaling (depth + width + resolution) suited to fine-grained medical image features
  - Proven in retinal imaging literature

### Custom Classification Head (added on top of base)
```
EfficientNetB3 (base, frozen in Stage 1)
  └─ GlobalAveragePooling2D        (name: "gap")
  └─ BatchNormalization            (name: "bn_1")
  └─ Dropout(0.4)                  (name: "drop_1")
  └─ Dense(256, activation="relu") (name: "fc_256")
  └─ BatchNormalization            (name: "bn_2")
  └─ Dropout(0.3)                  (name: "drop_2")
  └─ Dense(5, activation="softmax")(name: "predictions")
```

- **Dropout rates:** 0.4 (first), 0.3 (second) — reduces overfitting
- **Dense units:** 256
- **Output:** 5-class softmax
- **Loss function:** `sparse_categorical_crossentropy`
- **Metrics:** accuracy
- **Optimiser (Stage 1):** Adam, lr=1e-3
- **Optimiser (Stage 2):** Adam, lr=1e-5 (100× lower for fine-tuning)
- **Full model name:** `DR_EfficientNetB3`

### Source
- Defined in: `src/model.py`, `build_model()` function (lines 44–88)
- Fine-tune logic: `unfreeze_top_layers()` (lines 91–115)

---

## 5. Training Strategy

### Two-Stage Approach
| Stage | Description | Max Epochs | LR |
|---|---|---|---|
| Stage 1 | Frozen base — train head only | 30 | 1e-3 |
| Stage 2 | Fine-tune top 30 EfficientNetB3 layers | 20 | 1e-5 |

- **Stage 1 rationale:** Frozen base prevents destroying ImageNet features before head is trained
- **Stage 2 rationale:** Lower LR fine-tunes task-specific features without catastrophic forgetting
- **Layers unfrozen in Stage 2:** Top 30 layers of EfficientNetB3

### Hyperparameters (from `src/config.py`)
| Parameter | Value |
|---|---|
| `BATCH_SIZE` | 32 |
| `EPOCHS_FROZEN` | 30 |
| `EPOCHS_FINETUNE` | 20 |
| `RANDOM_STATE` | 42 |
| `IMAGE_SIZE` | (224, 224) |
| `IMAGENET_MEAN` | [0.485, 0.456, 0.406] |
| `IMAGENET_STD` | [0.229, 0.224, 0.225] |

### Callbacks (from `src/model.py`, `get_callbacks()`)
| Callback | Config |
|---|---|
| `ModelCheckpoint` | Monitor: `val_accuracy`; save best only; path: `checkpoints/best_model.keras` |
| `EarlyStopping` | Monitor: `val_accuracy`; patience=10; restore best weights |
| `ReduceLROnPlateau` | Monitor: `val_loss`; factor=0.5; patience=4; min_lr=1e-7 |
| `TensorBoard` | histogram_freq=1; log_dir: `logs/` |
| `CSVLogger` | Saves per-epoch metrics to `logs/training_log.csv` |

### Class Weights
- Computed using `sklearn.utils.class_weight.compute_class_weight(class_weight="balanced")`
- Applied during `model.fit()` via `class_weight=class_weight_dict`
- Purpose: penalises misclassification of minority classes more heavily
- Applied to training split only

### Data Loading
- `ImageDataGenerator.flow_from_directory()` on `data/augmented/train/` and `data/augmented/val/`
- Class mode: `"sparse"` (integer labels)
- Shuffle: True for train, False for val/test

### Training Environment
- Trained on Google Colab (T4 GPU)
- TensorFlow / Keras framework
- `colab_train.ipynb` used for execution

### History Files
| Stage | Path |
|---|---|
| Stage 1 history | `reports/stage1/training_history.json` |
| Stage 2 history | `reports/stage2/training_history.json` |

---

## 6. Evaluation Results

### Test Set
- **Total samples:** 550
- **Split:** held-out 15% test split (real-world class distribution, no augmentation)
- **Evaluation script:** `src/evaluate.py`

### Stage 1 vs Stage 2 — Full Comparison

**Overall Metrics:**
| Metric | Stage 1 (`best_model.keras`) | Stage 2 (`finetune_best_model.keras`) |
|---|---|---|
| Accuracy | **76.36%** | 75.82% |
| Macro avg F1 | **0.5664** | 0.5586 |
| Weighted avg F1 | **0.7568** | 0.7466 |
| Weighted avg precision | **0.7553** | 0.7419 |
| Weighted avg recall | 0.7636 | 0.7582 |

**Per-Class F1 Scores:**
| Class | Stage 1 F1 | Stage 2 F1 | Support |
|---|---|---|---|
| Mild | 0.4655 | 0.4505 | 56 |
| Moderate | **0.7093** | 0.6877 | 150 |
| No_DR | 0.9524 | **0.9527** | 271 |
| Proliferative_DR | **0.3478** | 0.2462 | 44 |
| Severe | 0.3571 | **0.4561** | 29 |

**Per-Class Precision:**
| Class | Stage 1 | Stage 2 |
|---|---|---|
| Mild | 0.4500 | 0.4545 |
| Moderate | 0.6810 | 0.6527 |
| No_DR | 0.9455 | 0.9391 |
| Proliferative_DR | **0.4800** | 0.3810 |
| Severe | 0.3704 | **0.4643** |

**Per-Class Recall:**
| Class | Stage 1 | Stage 2 |
|---|---|---|
| Mild | 0.4821 | 0.4464 |
| Moderate | **0.7400** | 0.7267 |
| No_DR | 0.9594 | **0.9668** |
| Proliferative_DR | **0.2727** | 0.1818 |
| Severe | 0.3448 | **0.4483** |

### Final Model Selection Decision
- **Selected: Stage 1 (`best_model.keras`) → renamed to `checkpoints/final_model.keras`**
- **Reasons:**
  1. Higher overall accuracy (+0.54pp)
  2. Higher weighted F1 (+0.0102)
  3. Proliferative DR F1: 0.3478 vs 0.2462 — Stage 2 degraded the most clinically critical class by **0.1016 F1 points**
  4. Stage 2 improved Severe recall (0.45 vs 0.34) but this gain was outweighed by the Proliferative DR collapse
  5. Frozen-base model generalises better — fine-tuning overfitted or disrupted Proliferative features

### Evaluation Output Files
| File | Path |
|---|---|
| Final model classification report | `reports/screenshots/evaluation/final_model/classification_report.txt` |
| Final model confusion matrix | `reports/screenshots/evaluation/final_model/confusion_matrix.png` |
| Final model ROC curves | `reports/screenshots/evaluation/final_model/roc_curves.png` |
| Stage 1 classification report | `reports/screenshots/evaluation/best_model/classification_report.txt` |
| Stage 1 confusion matrix | `reports/screenshots/evaluation/best_model/confusion_matrix.png` |
| Stage 1 ROC curves | `reports/screenshots/evaluation/best_model/roc_curves.png` |
| Stage 2 classification report | `reports/screenshots/evaluation/finetune_best_model/classification_report.txt` |
| Stage 2 confusion matrix | `reports/screenshots/evaluation/finetune_best_model/confusion_matrix.png` |
| Stage 2 ROC curves | `reports/screenshots/evaluation/finetune_best_model/roc_curves.png` |
| Training curves (accuracy + loss, both stages) | `reports/screenshots/training/training_curves.png` |

---

## 7. UI & Chatbot (app/streamlit_app.py)

### Technology
- **Framework:** Streamlit
- **Runs locally:** `streamlit run app/streamlit_app.py`
- **Default port:** 8501

### Features
- Image upload (JPG/PNG) via `st.file_uploader`
- Preprocessing: reuses `preprocess_image()` from `src/preprocessing.py` directly (no duplication)
- Model loading: `tf.keras.models.load_model(cfg.FINAL_MODEL_PATH)` — cached with `@st.cache_resource`
- Prediction: softmax probability for all 5 classes
- Results display:
  - Predicted class name (human-readable)
  - `st.progress()` bar per class showing probability %
- **Chatbot section:**
  - 3 quick-question buttons: "What does this stage mean?", "What should I do next?", "How serious is this?"
  - Responses are class-specific (15 total: 5 classes × 3 questions)
  - Escalating urgency: No DR → routine checkups; Proliferative → urgent specialist referral
  - Disclaimer: "Educational purposes only, not a substitute for professional medical diagnosis"
  - Chat history persists within session via `st.session_state`
  - Rendered as `st.chat_message` bubbles

### Class Mapping in UI
- `MODEL_CLASS_NAMES = ["Mild", "Moderate", "No_DR", "Proliferative_DR", "Severe"]`
- Alphabetical order — matches `flow_from_directory` Keras assignment

### UI Screenshots
- No screenshots currently exist in `reports/screenshots/ui/` (folder not yet created)
- To generate: run app, upload test image, capture manually

---

## 8. Code Quality & Repository

### GitHub Repository
- **URL:** https://github.com/Thiya18/ComputerVisionCW
- **Branch:** `main`

### Repository Structure
```
ComputerVisionCW/
├── AGENTS.md
├── README.md
├── requirements.txt
├── colab_train.ipynb
├── app/
│   └── streamlit_app.py
├── src/
│   ├── config.py          # Single source of truth — all paths & hyperparams
│   ├── preprocessing.py   # CLAHE, resize, denoise, sharpen, normalise
│   ├── augmentation.py    # Albumentations pipeline, balancing logic
│   ├── model.py           # EfficientNetB3 architecture, callbacks
│   ├── train.py           # Two-stage training pipeline
│   └── evaluate.py        # Classification report, confusion matrix, ROC curves
├── checkpoints/
│   ├── final_model.keras  # Selected final model (Stage 1)
│   └── best_model.keras
├── reports/
│   ├── stage1/training_history.json
│   ├── stage2/training_history.json
│   └── screenshots/
├── docs/
│   └── report_data_pack.md
├── notebooks/
└── data/                  # gitignored
```

### Code Standards Applied
- Python 3.10+
- Type hints on all public functions
- Google-style docstrings on every function
- `random_state=42` for all stochastic operations
- All modules import `config as cfg` — no hardcoded paths
- Figures saved to `reports/screenshots/<subfolder>/` as PNG at 150 DPI
- Training progress tracked with `tqdm`

---

## 9. Complete Screenshot Inventory

### `reports/screenshots/dataset/`
| Filename | Description |
|---|---|
| `class_distribution.png` | Bar chart: raw APTOS 2019 class counts (all 3662 images) |

### `reports/screenshots/preprocessing/`
| Filename | Description |
|---|---|
| `000c1434d8d7_before_after.png` | Before/after: original vs CLAHE+denoise+sharpen (sample 1) |
| `001639a390f0_before_after.png` | Before/after: original vs CLAHE+denoise+sharpen (sample 2) |
| `0024cdab0c1e_before_after.png` | Before/after: original vs CLAHE+denoise+sharpen (sample 3) |
| `002c21358ce6_before_after.png` | Before/after: original vs CLAHE+denoise+sharpen (sample 4) |
| `0104b032c141_before_after.png` | Before/after: original vs CLAHE+denoise+sharpen (sample 5) |

### `reports/screenshots/augmentation/`
| Filename | Description |
|---|---|
| `augmentation_grid.png` | Grid: 1 original + 8 augmented variants of one training image |

### `reports/screenshots/training/`
| Filename | Description |
|---|---|
| `training_curves.png` | Accuracy + loss curves across both training stages (concatenated) |

### `reports/screenshots/evaluation/final_model/`
| Filename | Description |
|---|---|
| `classification_report.txt` | Full sklearn report — Stage 1 / final model |
| `confusion_matrix.png` | Side-by-side: raw counts + normalised confusion matrix |
| `roc_curves.png` | One-vs-rest ROC curves with AUC per class |

### `reports/screenshots/evaluation/best_model/`
*(Identical results to final_model — same Stage 1 weights)*
| Filename | Description |
|---|---|
| `classification_report.txt` | Classification report (Stage 1) |
| `confusion_matrix.png` | Confusion matrix (Stage 1) |
| `roc_curves.png` | ROC curves (Stage 1) |

### `reports/screenshots/evaluation/finetune_best_model/`
*(Stage 2 — NOT selected as final)*
| Filename | Description |
|---|---|
| `classification_report.txt` | Classification report (Stage 2) |
| `confusion_matrix.png` | Confusion matrix (Stage 2) |
| `roc_curves.png` | ROC curves (Stage 2) |

---

*Compiled from: `src/config.py`, `src/preprocessing.py`, `src/augmentation.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`, `app/streamlit_app.py`, and `reports/screenshots/evaluation/*/classification_report.txt`*
