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
│   ├── task2/
│   │   └── Group03_PRBD_task2_report.pdf
│   └── task3/
│   │    └── Group03_PRBD_task3_report.pdf
│   └──  Final Report/
│   │     └── Final Report.pdf
├── code/
│   ├── task1/
│   │   └── Group03_PRBD_task1_eda.ipynb
│   ├── task2/
│   │   ├── Group03_PRBD_task2_baselines.ipynb
│   │   └── Group03_PRBD_task2_proposed_model.ipynb
│   ├── task3/
│   │   ├── Group03_PRBD_task3_improvement_ablation.ipynb
│   │   └── Group03_PRBD_task3_explainability.ipynb
│   └── extra_gnn/
│       └── Group03_PRBD_GNN.ipynb                 
├── models/
│   ├── Group03_PRBD_final_model_best.pth
│   └── label_map.json
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
- Visual similarity noted between Beroi, Katari Najir, and Katari Siddho (likely confusion pairs); Beroi and Aush show a reddish/brown husk residue that may act as an unintended shortcut, checked with Grad-CAM in Task 3

## Task 2 Summary - Baselines + Proposed Attention Model
- Leakage-free, class-stratified 70/10/20 train/val/test split built once on `Original_Images` only (400 held-out test images, 40 per class) and reused identically across Task 2 and Task 3
- 4 representative baselines trained and evaluated: Simple CNN (from scratch), ResNet50, MobileNetV2, EfficientNet-B0 — full metric set reported (accuracy, macro/weighted precision-recall-F1, macro ROC-AUC, confusion matrix, ROC/PR curves, training time, inference time, parameter count, model size)
- **ResNet50** is the strongest baseline (Macro-F1 = 0.9625), narrowly ahead of MobileNetV2 (0.9622); both clearly outperform the from-scratch Simple CNN (0.8898)
- Proposed model: **ResNet50 + CBAM** (hybrid channel + spatial attention), inserted after every residual stage, built on the strongest baseline backbone to isolate attention's effect
- **First result:** attention did not improve the strongest backbone in this run (Macro-F1 = 0.9548 vs. 0.9625, Δ = −0.0077) — reported honestly rather than hidden; investigated further in Task 3 via ablation (CBAM placement, channel-only vs. spatial-only, fine-tuning budget)

## Task 3 Summary - Improvement, Ablation, CV + Significance, Explainability

Task 3 continues directly from the Task 2 proposed model (ResNet50 + CBAM). The same
leakage-free 70/10/20 split is reused; all model-selection decisions are made on the
validation set only, and the test set is touched exactly once per reported model.

### Improvement (Stage 1–2)
Twelve one-factor-at-a-time variants of the Task 2 CBAM model were evaluated on the
validation set — CBAM placement, fine-tuning depth, dropout, optimizer, LR schedule,
augmentation strength, and input size. The changes that helped were combined into a
final configuration: CBAM placed only after the last residual stage (instead of every
stage), a deeper fine-tuning budget (layer2–4 unfrozen), AdamW + cosine LR schedule,
stronger augmentation, and a larger 256×256 input.

### Final performance (Stage 2 + 5)
| Model | Accuracy | Macro-F1 | Params | Size (MB) |
|---|---|---|---|---|
| ResNet50 (Task 2 best baseline) | – | 0.9625 | 14,985,226 | 90.06 |
| ResNet50 (baseline, reproduced under Task 3 engine) | 0.9550 | 0.9547 | 14,985,226 | 90.06 |
| ResNet50 + CBAM (Task 2 first proposed model) | – | 0.9548 | 22,780,306 | 92.72 |
| **ResNet50 + CBAM (Task 3 final improved model)** | **0.9650** | **0.9650** | 22,780,306 | 92.72 |
| Prova (2025) — Attention CNN+CBAM, Bangladesh rice | – | 0.9192 | related work |
| Rajalakshmi et al. (2024) — RiceSeedNet | – | 0.9700 | related work |
| Ma et al. (2023) — I-CBAM-MobileNetV2, maize | 0.9821 | – | related work |

The final improved model recovers and surpasses the Task 2 negative delta, and sits
close to — though not above — the strongest related-work numbers (Pillar A: **Match**,
not Beat, on the strictest comparison).

### 5-fold CV + significance test (Stage 4)
- 5-fold stratified CV over the train+val pool only (test set excluded from every fold)
- **Final improved model:** 0.9697 ± 0.0113 (Macro-F1, mean ± std across folds)
- **Best baseline (reproduced, identical folds):** 0.9418 ± 0.0104
- **Paired t-test:** t = 4.2378, p = 0.0133 → statistically significant improvement at α = 0.05
- Wilcoxon signed-rank reported as a companion (n = 5 → floor p = 0.0625, cannot reach
  p < 0.05 by construction); Cohen's d = 1.895 (large effect)
- McNemar's exact test on the held-out test split corroborates the direction of the result

### Ablation (Stage 3A + 3B)
- Attention removed vs. channel-only vs. spatial-only vs. full CBAM vs. placement — full
  CBAM at the last stage was the best configuration
- Dropout on/off, augmentation on/off, fine-tuning depth — each varied independently
- On a from-scratch CNN family (to isolate architecture-level factors that a fixed
  pretrained ResNet50 cannot vary): conv-block count (2/3/4/5), filter width
  (16/32/64), kernel size (3/5/7), batch-norm on/off, CBAM on/off

### Explainability (Grad-CAM + LIME)
- Grad-CAM computed at `layer4` (before the final CBAM block) for one correct and one
  wrong prediction; for the wrong prediction, Grad-CAM is also computed for the *true*
  class to distinguish "wrong region" vs. "right region, wrong texture read" failure modes
- LIME (superpixel-based, model-agnostic) run alongside Grad-CAM as an independent check;
  agreement between the two (IoU) reported for each case
- CBAM spatial-attention overlay included to check whether the attention module agrees
  with the explanation methods
- Combined 2×4 panel (`combined_gradcam_lime_correct_vs_wrong.png`) and an auto-written
  findings file (`task3_explainability_findings.md`) produced for the report

## Graph Neural Networks (GCN / GATv2)

*This section is exploratory work carried out on the course instructor's suggestion.
It is not part of the graded Track 3 (CNN + Attention) deliverable — the official
Track 3 track carries no graph bonus — and is reported here separately for completeness.*

Each rice image is converted into a superpixel graph (SLIC segmentation), with 154-dim
node features per superpixel: 13 handcrafted colour/edge/geometry features, a 10-dim LBP
micro-texture histogram, 3 shape descriptors, and a 128-dim deep-texture embedding pooled
from a frozen, never-fine-tuned ImageNet-pretrained ResNet18 (leakage-safe). A GCN and a
GATv2 (edge-attributed) model were trained and compared against a ResNet50 CNN baseline
trained under the identical split.

An earlier version of this pipeline showed a large accuracy gap versus the CNN baseline.
Diagnosis (loss-curve behaviour, not diverging) pointed to an information bottleneck
rather than overfitting; the root cause was traced to weak node features (mostly colour,
one crude texture scalar) rather than superpixel granularity. Enriching the node features
(LBP + shape descriptors + frozen-CNN embedding) closed nearly the entire gap.

### Results (held-out test set)
| Model | Accuracy | Macro-F1 | Params | Inference (ms/img) |
|---|---|---|---|---|
| **GATv2** | 97.26% | 0.9726 | 108,170 | 0.15 |
| ResNet50 (CNN baseline, same split) | 97.01% | 0.9702 | 23,528,522 | 6.05 |
| GCN | 96.51% | 0.9651 | — | — |

- **5-fold CV:** ResNet50 0.9580 ± 0.0075, GATv2 0.9731 ± 0.0045, GCN 0.9355 ± 0.0170
- **Significance:** paired Wilcoxon across the 5 CV folds gives p = 0.0625 for every
  pairwise comparison (the smallest attainable value at n = 5); the direction is
  consistent (GATv2 ahead of both other models in every fold) but this does **not**
  reach the conventional p < 0.05 threshold, so GATv2 is reported as **numerically
  ahead of, and competitive with, the CNN baseline — not a statistically proven win**
- **Efficiency:** GATv2 uses ~217× fewer parameters than ResNet50 and infers ~40× faster,
  while matching or slightly exceeding it on accuracy — an efficiency gain with no
  accuracy penalty on this dataset
- An ablation over segmentation granularity (`n_segments` ∈ {60, 90, 150}) confirmed
  node-feature richness, not superpixel count, was the original bottleneck

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

### Task 3 - Improvement, Ablation, CV + Significance, Explainability
1. Ensure Task 2's split (`splits/group03_prbd_task2_split.csv`) and saved artifacts are available
2. Open and run `code/task3/Group03_PRBD_task3_improvement_ablation.ipynb` — Stage 1 (improvement search), Stage 2 (final model + reproduced baseline), Stage 3 (ablation), Stage 4 (5-fold CV + significance), Stage 5 (final comparison vs. related work). Download the artifact bundle at the end of the notebook
3. Open and run `code/task3/Group03_PRBD_task3_explainability.ipynb` — upload the artifact bundle from step 2 when prompted, then run all cells to reproduce the Grad-CAM + LIME figures

### Extra - GNN Exploration (optional, not required for grading)
1. Open `code/extra_gnn/Group03_PRBD_GNN.ipynb` on a GPU runtime (Kaggle/Colab)
2. Update the dataset path if needed; the notebook reuses Task 2's split when available
3. Run all cells to reproduce the superpixel-graph construction, GCN/GATv2 training, 5-fold CV, and the final comparison against the CNN baseline

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

### Task 3
| Model | Accuracy | Macro-F1 |
|---|---|---|
| ResNet50 baseline (reproduced) | 0.9550 | 0.9547 |
| ResNet50 + CBAM (Task 2 first model) | – | 0.9548 |
| **ResNet50 + CBAM (Task 3 final model)** | **0.9650** | **0.9650** |

5-fold CV: 0.9697 ± 0.0113 vs. baseline 0.9418 ± 0.0104 (paired t-test p = 0.0133,
significant). Full ablation tables, CV/significance results, Grad-CAM and LIME figures
are in `report/task3/Group03_PRBD_task3_report.pdf`.

### GNN Exploration
GATv2 reached 0.9726 test Macro-F1 vs. 0.9702 for the ResNet50 CNN baseline - numerically
ahead but not statistically significant (Wilcoxon p = 0.0625) - with ~217× fewer
parameters and ~40× faster inference. See `code/extra_gnn/Group03_PRBD_GNN.ipynb` for
the full pipeline and honest-reading discussion.
