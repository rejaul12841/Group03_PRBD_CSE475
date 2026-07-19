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
│   └── task1/
│       └── Group03_PRBD_task1_report.pdf
├── code/
│   └── task1/
│       └── Group03_PRBD_task1_eda.ipynb
└── related_work/
    ├── Group03_PRBD_related_work_table.pdf
    └── papers/
        ├── Enhancing agricultural research with an Attention-Based Hybrid Model.pdf
        ├── Identification of Maize Seed Varieties Using MobileNetV2 with Improved Attention Mechanism CBAM.pdf
        ├── Multi-class rice seed recognition based on deep space attention.pdf
        ├── Real-time segmentation and phenotypic analysis of rice.pdf
        └── RiceSeedNet.pdf
```

## Task 1 Summary — Exploratory Data Analysis
- Confirmed 2000 images across 10 balanced classes (200 each, 1:1 ratio)
- Uniform image resolution: 640×480 pixels (no distortion risk when resized to 224×224)
- Brightness/colour distributions overlap heavily across classes → colour is not a usable shortcut feature, supporting the use of an attention mechanism
- No corrupt files or exact duplicates found; a few blurry images flagged (Katari Najir, BR-29, Swarna) for manual review
- Visual similarity noted between Beroi, Katari Najir, and Katari Siddho (likely confusion pairs); Beroi and Aush show a reddish/brown husk residue that may act as an unintended shortcut, to be checked with Grad-CAM in Task 3

## How to Run
1. Clone the repository
2. Open `code/task1/Group03_PRBD_task1_eda.ipynb` in Jupyter/Colab
3. Download the PRBD dataset from Mendeley Data and update the dataset path in the notebook
4. Run all cells to reproduce the EDA (class balance, sample grid, resolution check, pixel/colour analysis, image quality check)

## Results (Task 1)
- Dataset confirmed clean, balanced, and uniformly sized
- Key findings guide Task 2 modeling decisions: use of macro-F1 alongside accuracy, CNN + attention architecture choice, and a leakage-safe original-images-only split strategy
