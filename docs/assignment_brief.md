# Computer Vision — Course Work Descriptor

## Module Info

| Field | Detail |
|---|---|
| **Course** | BSc (Hons) Computer Science |
| **Module** | Computer Vision |
| **Batch** | BSCCOMP24.2P |
| **CW No.** | 1 |
| **Assessment mode** | Individual |
| **Assessment type** | Project Report with Video demonstration |
| **Hand out date** | 2026-10-03 |
| **Hand in date** | 2026-10-03 *(confirm exact date with your lecturer)* |
| **Submission** | Via Turnitin on the module's VLE page |
| **Total marks** | 100 |

---

## Learning Outcomes Covered

1. Evaluate the suitability of image models and representations commonly used in image and computer vision.
2. Critically discuss a variety of hardware and software technologies used in acquisition, display and transmission of image data.
3. Select, apply and critically evaluate a variety of algorithms typically used in the processing and analysis of images, including techniques for filtering, segmentation, labelling and feature extraction.
4. Evaluate the suitability of concepts, representations and techniques used in object and pattern recognition.

---

## Assignment Description

**Title: Diabetic Retinopathy Stage Detection**

- Use basic image preprocessing (e.g., contrast adjustment, edge enhancement) to organise the dataset acquired from the Kaggle platform.
- Use data augmentation techniques as required to enhance the diversity of the dataset.
- Use a suitable CNN architecture with transfer learning to classify diabetic retinopathy, as well as the stage of the disease.
- Evaluate the effectiveness of the model via plotting accuracy and loss curves, along with precision, recall, and F1-score parameters.

---

## Deliverables

- **Codebase** with inline comments.
- **One PDF report**, not exceeding 20 pages, including:
  - Graphical evidence with supportive explanations on all steps followed above.
  - A record of the prototype in operation, including the hosted video URL in the report.

---

## Learning Outcome Mapping

| Assignment Part | LOs Mapped | Justification |
|---|---|---|
| Complete assignment | LO1, LO2, LO3, LO4 | Use of image models, processing techniques, and classifiers in a diabetic retinopathy detection system |

---

## Notes

- Plagiarism will result in disqualification.
- All code must be original.
- All submissions must be original work; plagiarism is penalised per university regulations.

---

## Marking Rubric (100 Marks Total)

### 1. Problem Understanding & Dataset Justification — 10 marks

| Grade | Criteria |
|---|---|
| **Excellent** | Clearly explains diabetic retinopathy and its medical significance; dataset from Kaggle is highly relevant; detailed description of classes, image distribution, train/validation/test splits, ethical concerns, and dataset limitations. |
| **Very Good** | Strong understanding of the problem and dataset; most dataset characteristics explained with minor omissions. |
| **Good** | Basic explanation of the problem and dataset; limited discussion of dataset properties or challenges. |
| **Needs Work** | Weak articulation of the problem; dataset usage not well justified. |
| **Poor** | Problem not properly explained; dataset inappropriate or poorly described. |

---

### 2. Data Preprocessing Techniques — 10 marks

| Grade | Criteria |
|---|---|
| **Excellent** | Comprehensive preprocessing pipeline implemented and justified (contrast enhancement, resizing, normalisation, noise removal, edge enhancement, etc.); preprocessing is reproducible and improves image quality significantly. |
| **Very Good** | Appropriate preprocessing applied with good explanations; minor missing details or justification gaps. |
| **Good** | Basic preprocessing implemented correctly but lacks depth or proper reasoning. |
| **Needs Work** | Incomplete or inconsistent preprocessing; weak technical implementation. |
| **Poor** | Minimal or incorrect preprocessing; introduces errors or data leakage. |

---

### 3. Data Augmentation & Dataset Balancing — 10 marks

| Grade | Criteria |
|---|---|
| **Excellent** | Effective augmentation techniques used (rotation, flipping, zooming, brightness adjustments, etc.) with strong justification; class imbalance handled appropriately. |
| **Very Good** | Good augmentation strategy with minor limitations or missing analysis. |
| **Good** | Standard augmentation methods applied but without detailed reasoning. |
| **Needs Work** | Limited augmentation; imbalance issues not properly addressed. |
| **Poor** | No augmentation or incorrect application leading to poor dataset quality. |

---

### 4. CNN Architecture & Transfer Learning Implementation — 20 marks

| Grade | Criteria |
|---|---|
| **Excellent** | Appropriate CNN architecture selected (e.g., ResNet, EfficientNet, VGG, MobileNet); transfer learning correctly implemented and optimised; architecture design is well justified with hyperparameter tuning. |
| **Very Good** | Strong implementation of CNN and transfer learning with minor optimisation issues. |
| **Good** | Functional model with acceptable architecture choice; limited tuning or explanation. |
| **Needs Work** | Weak model selection or implementation; poor understanding of transfer learning concepts. |
| **Poor** | Model implementation incorrect, incomplete, or non-functional. |

---

### 5. Training Strategy & Experimental Design — 10 marks

| Grade | Criteria |
|---|---|
| **Excellent** | Robust training strategy includes validation methods, callbacks, early stopping, learning rate scheduling, and overfitting prevention techniques; experiments well organised. |
| **Very Good** | Good training strategy with minor missing optimisation techniques. |
| **Good** | Basic training process implemented correctly but lacks advanced strategies. |
| **Needs Work** | Weak experimental setup; limited control of overfitting or validation process. |
| **Poor** | Poor or invalid training methodology. |

---

### 6. Model Evaluation & Performance Analysis — 15 marks

| Grade | Criteria |
|---|---|
| **Excellent** | Comprehensive evaluation using accuracy, precision, recall, F1-score, confusion matrix, and loss/accuracy curves; insightful interpretation of results and error analysis included. |
| **Very Good** | Strong evaluation with most required metrics and reasonable interpretation. |
| **Good** | Acceptable evaluation with standard metrics but limited analytical discussion. |
| **Needs Work** | Incomplete evaluation; missing important metrics or graphs. |
| **Poor** | Evaluation missing, incorrect, or unsupported. |

---

### 7. Code Quality & Documentation — 10 marks

| Grade | Criteria |
|---|---|
| **Excellent** | Well-structured, modular, and fully commented original code; easy to reproduce and execute; follows good coding practices. |
| **Very Good** | Mostly clean and documented code with minor clarity issues. |
| **Good** | Functional code with basic comments and organisation. |
| **Needs Work** | Poorly organised or partially documented code. |
| **Poor** | Incomplete, copied, or non-functional code. |

---

### 8. Report Quality & Presentation — 10 marks

| Grade | Criteria |
|---|---|
| **Excellent** | Professional report structure within page limit; excellent explanations, visuals, graphs, screenshots, and workflow documentation; hosted prototype video link included and clearly explained. |
| **Very Good** | Clear and well-organised report with good visuals and explanations. |
| **Good** | Understandable report with moderate organisation and sufficient details. |
| **Needs Work** | Weak structure, unclear explanations, or missing important visuals/evidence. |
| **Poor** | Poorly written, incomplete, or missing major sections/video evidence. |

---

### 9. Innovation, Practical Impact & Critical Discussion — 5 marks

| Grade | Criteria |
|---|---|
| **Excellent** | Demonstrates originality, discusses real-world healthcare impact, deployment feasibility, limitations, ethical concerns, and future improvements comprehensively. |
| **Very Good** | Good discussion of practical applications and limitations with some originality. |
| **Good** | Basic impact discussion with limited depth or innovation. |
| **Needs Work** | Minimal discussion of real-world relevance or future scope. |
| **Poor** | No meaningful discussion of impact or innovation. |

---

## Mark Allocation Summary

| Component | Marks |
|---|---|
| Problem Understanding & Dataset Justification | 10 |
| Data Preprocessing Techniques | 10 |
| Data Augmentation & Dataset Balancing | 10 |
| CNN Architecture & Transfer Learning | 20 |
| Training Strategy & Experimental Design | 10 |
| Model Evaluation & Performance Analysis | 15 |
| Code Quality & Documentation | 10 |
| Report Quality & Presentation | 10 |
| Innovation & Practical Impact | 5 |
| **Total** | **100** |
