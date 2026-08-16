# Group03_PRBD_CSE475

**Course:** CSE 475 – Machine Learning  
**Semester:** Summer 2026, Section: 06  
**Group:** 03  
**Track:** Track 3 – CNN with Attention  

## Group Members
| Name | ID |
|---|---|
| Israt Sultana | 2022-2-60-050 |
| MD. Rejaul Hoq | 2023-1-60-124 |
| Istiaque Ahmed | 2022-3-60-143 |

## Dataset
**PRBD (Processed Rice Bangladesh Dataset)** — obtained from Mendeley Data  
(data.mendeley.com/datasets/sfp9s96prh/1)

- 10 processed rice varieties from Bangladesh: Aush, Beroi, BR-28, BR-29, Chinigura, Ghee Bhog, Katari Najir, Katari Siddho, Miniket, Swarna
- `Original_Images/`: 2000 high-resolution microscopic images (200 per class), JPG format
- `Augmented_images/`: 8000 pre-augmented images (rotation, flipping, scaling)
- Only `Original_Images` is used for EDA and train/val/test splitting to avoid data leakage from near-duplicate augmented copies

## Project Title
**Rice Variety Classification from Microscopic Images using CNN with Attention**

## Repository Structure

```
Group03_PRBD_CSE475/
├── README.md
├── report/
│   ├── task1/
│   │   └── Group03_PRBD_task1_report.pdf
│   └── task2/
│       └── Group03_PRBD_task2_report.pdf
├── code/
│   ├── task1/
│   │   └── Group03_PRBD_task1_eda.ipynb
│   └── task2/
│       ├── Group03_PRBD_task2_baselines.ipynb
│       └── Group03_PRBD_task2_proposed_model.ipynb
└── related_work/
    ├── Group03_PRBD_related_work_table.pdf
    └── papers/
        ├── Enhancing agricultural research with an Attention-Based Hybrid Model.pdf
        ├── Identification of Maize Seed Varieties Using MobileNetV2 with Improved Attention Mechanism CBAM.pdf
        ├── Multi-class rice seed recognition based on deep space attention.pdf
        ├── Real-time segmentation and phenotypic analysis of rice.pdf
        └── RiceSeedNet.pdf
```

## Task 1 Summary - Exploratory Data Analysis
- Confirmed 2000 images across 10 balanced classes (200 each, 1:1 ratio)
- Uniform image resolution: 640×480 pixels (no distortion risk when resized to 224×224)
- Brightness/colour distributions overlap heavily across classes → colour is not a usable shortcut feature, supporting the use of an attention mechanism
- No corrupt files or exact duplicates found; a few blurry images flagged (Katari Najir, BR-29, Swarna) for manual review
- Visual similarity noted between Beroi, Katari Najir, and Katari Siddho (likely confusion pairs); Beroi and Aush show a reddish/brown husk residue that may act as an unintended shortcut, to be checked with Grad-CAM in Task 3

## Task 2 Summary - Baselines + Proposed Attention Model
- Leakage-free, class-stratified 70/10/20 train/val/test split built once on `Original_Images` only (400 held-out test images, 40 per class) and reused identically across both notebooks
- 4 representative baselines trained and evaluated: Simple CNN (from scratch), ResNet50, MobileNetV2, EfficientNet-B0 — full metric set reported (accuracy, macro/weighted precision-recall-F1, macro ROC-AUC, confusion matrix, ROC/PR curves, training time, inference time, parameter count, model size)
- **ResNet50** is the strongest baseline (Macro-F1 = 0.9625), narrowly ahead of MobileNetV2 (0.9622); both clearly outperform the from-scratch Simple CNN (0.8898)
- Proposed model: **ResNet50 + CBAM** (hybrid channel + spatial attention), inserted after every residual stage, built on the strongest baseline backbone to isolate attention's effect
- **First result:** attention did not improve the strongest backbone in this run (Macro-F1 = 0.9548 vs. 0.9625, Δ = −0.0077) — reported honestly rather than hidden; this negative delta will be investigated in Task 3 via ablation (CBAM placement, channel-only vs. spatial-only, fine-tuning budget)

## How to Run

### Task 1 - EDA
1. Clone the repository
2. Open `code/task1/Group03_PRBD_task1_eda.ipynb` in Jupyter/Colab
3. Download the PRBD dataset from Mendeley Data and update the dataset path in the notebook
4. Run all cells to reproduce the EDA (class balance, sample grid, resolution check, pixel/colour analysis, image quality check)

### Task 2 - Baselines + Proposed Model
1. Download the PRBD dataset from Mendeley Data and update the dataset path
2. Open and run `code/task2/Group03_PRBD_task2_baselines.ipynb` first — this builds and saves the leakage-free 70/10/20 split, then trains and evaluates all 4 baselines
3. Open and run `code/task2/Group03_PRBD_task2_proposed_model.ipynb` — this loads the same saved split and trains/evaluates the proposed ResNet50+CBAM model
4. Run all cells in order in both notebooks to reproduce the reported metrics, plots, and confusion matrices

## Results

### Task 1
- Dataset confirmed clean, balanced, and uniformly sized
- Key findings guide Task 2 modeling decisions: use of macro-F1 alongside accuracy, CNN + attention architecture choice, and a leakage-safe original-images-only split strategy

### Task 2
| Model | Accuracy | Macro-F1 | Params | Size (MB) |
|---|---|---|---|---|
| ResNet50 (baseline) | 0.9625 | 0.9625 | 14,985,226 | 90.06 |
| MobileNetV2 (baseline) | 0.9625 | 0.9622 | 1,538,890 | 8.77 |
| **ResNet50 + CBAM (proposed)** | 0.9550 | 0.9548 | 22,780,306 | 92.72 |
| EfficientNet-B0 (baseline) | 0.9525 | 0.9524 | 3,168,550 | 15.62 |
| Simple CNN (baseline) | 0.8900 | 0.8898 | 423,562 | 1.63 |

Full metrics, confusion matrices, ROC/PR curves, architecture diagram, and CBAM attention-map visualization are in `report/task2/Group03_PRBD_task2_report.pdf`.
