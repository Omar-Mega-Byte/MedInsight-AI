# 🧠 Brain Tumor MRI Classifier

> A classical computer vision pipeline for multi-class brain tumor detection from MRI scans — no deep learning, fully explainable.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Overview

This project implements a complete, end-to-end computer vision pipeline for classifying brain MRI images into four categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. The entire pipeline — from raw image loading through inference — relies exclusively on classical CV and machine learning techniques, making every decision traceable and interpretable.

The project was developed as a group assignment for a Computer Vision course and covers the full spectrum from low-level image processing to multi-class probabilistic classification.

---

## Features

- **Noise reduction** via Gaussian, Median, and Bilateral filtering with quantitative MSE/PSNR evaluation
- **Contrast enhancement** using CLAHE (Contrast Limited Adaptive Histogram Equalization) on the LAB color space
- **Feature detection** with Harris Corner Detector and multi-threshold analysis
- **Scale-space analysis** using Gaussian and Laplacian image pyramids (5 levels) with LoG blob detection
- **Local feature matching** via SIFT keypoints, FLANN-based matching, and RANSAC homography verification
- **Segmentation** with Otsu thresholding, K-Means clustering, and Watershed algorithm, evaluated via IoU
- **Rich feature engineering**: color histograms, HOG, LBP texture, gradient statistics, and optional Gabor filters
- **Dimensionality reduction** via PCA with cumulative variance analysis
- **Class imbalance handling** using SMOTE oversampling
- **Multi-classifier comparison**: Naive Bayes, SVM (RBF), AdaBoost, Gradient Boosting, Random Forest
- **Full evaluation suite**: confusion matrix, per-class metrics, ROC curves (OvR), calibration curves, AUC scores
- **End-to-end inference pipeline** that runs the complete chain on a single image path

---

## Dataset

| Property | Value |
|---|---|
| Source | [masoudnickparvar/brain-tumor-mri-dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle) |
| Classes | `glioma`, `meningioma`, `notumor`, `pituitary` |
| Split | `Training/` and `Testing/` subfolders |
| Input size | Resized to 224×224 (96×96 for feature extraction) |

**Setup**: Place the dataset at the project root so the loader can find `./Training/` and `./Testing/`. Alternatively, configure your Kaggle API credentials and the notebook will attempt an automatic download.

```
project-root/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── ...
```

---

## Pipeline

```
Raw MRI Image
     │
     ▼
1. Preprocessing        — Gaussian / Median / Bilateral / CLAHE
     │
     ▼
2. Feature Detection    — Harris corners, multi-scale pyramids, LoG blobs
     │
     ▼
3. SIFT Matching        — Keypoint extraction, FLANN matching, RANSAC verification
     │
     ▼
4. Segmentation         — Otsu, K-Means (k=4), Watershed → IoU evaluation
     │
     ▼
5. Feature Engineering  — Color histograms + HOG + LBP + gradient stats (+ Gabor optional)
     │
     ▼
6. Preprocessing        — StandardScaler → PCA (80 components) → SMOTE
     │
     ▼
7. Classification       — NB / SVM / AdaBoost / GradientBoosting / RandomForest
     │
     ▼
8. Evaluation           — Accuracy, F1, ROC-AUC, calibration curves
     │
     ▼
9. Inference            — full_pipeline(img_path) → prediction + visualizations
```

### Feature vector breakdown

| Group | Dimensions |
|---|---|
| Color histograms (3 channels × 24 bins) | 72 |
| Intensity statistics | 6 |
| HOG (9 orientations, 16×16 cells) | ~144 |
| LBP texture (P=16, R=2, uniform) | 20 |
| Gradient statistics (Sobel) | 3 |
| Gabor (optional, off by default) | 12 |
| **Total** | **~245** |

---

## Getting Started

### Requirements

```bash
pip install numpy pandas opencv-contrib-python scikit-image scikit-learn \
            matplotlib seaborn pillow imbalanced-learn
```

> `opencv-contrib-python` is required for SIFT. Do **not** install both `opencv-python` and `opencv-contrib-python` simultaneously.

### Run

Open and run `cv-project-v1-1-0.ipynb` top to bottom in Jupyter or any compatible environment. All cells are self-contained and execute sequentially.

```bash
jupyter notebook cv-project-v1-1-0.ipynb
```

### Inference on a single image

```python
result = full_pipeline("path/to/mri.jpg")
print(result["prediction"])   # e.g. "glioma"
```

---

## Results

All metrics are computed on the held-out test set (25% stratified split).

| Model | Accuracy | F1 Macro | CV F1 (mean ± std) |
|---|---|---|---|
| Naive Bayes | 76.06% | 75.91% | 76.73% ± 0.97% |
| **SVM (RBF)** | **94.72%** | **94.71%** | **92.77% ± 0.29%** |
| AdaBoost | 75.22% | 75.03% | 73.16% ± 1.07% |
| Gradient Boosting | 88.50% | 88.47% | 87.95% ± 0.55% |
| Random Forest | 90.89% | 90.86% | 88.97% ± 1.00% |


**Feature importance** (Gradient Boosting, projected through PCA): HOG and LBP texture features consistently dominate, followed by color histogram contributions.

---

## Project Structure

```
cv-project-v1-1-0.ipynb   # Main notebook (all sections)
README.md
```

---

---

## License

This project is released under the Apache License 2.0  
Dataset is provided by [masoudnickparvar](https://www.kaggle.com/masoudnickparvar) on Kaggle under its original terms.
