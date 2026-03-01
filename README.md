---
title: HydraLANet
emoji: 👁️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# **Enhancing Microaneurysm Segmentation in Retinal Fundus Imaging with HydraLA-Net**

This project investigates techniques for improving microaneurysm segmentation in diabetic retinopathy fundus imaging, where lesions are extremely small and often low-contrast relative to surrounding tissue. We experiment on an adapted version of a previously established segmentation model for Diabetic Retinopathy, LA-Net, into HydraLA-Net.

![Model Architecture](readme_images/HydraLA-Net.png)
*Model Architecture Hand-Drawn by Michael Liu using Goodnotes and Inkscape*

### Project Contributors
**Jessian Yuan**
- Technical Project Manager

**Michael Liu**
- Defined and Finalized the Primary Research Objective
- Model Architecture and Model Modifications
- Model Training (WATGPU via SSH)
- Loss Functions and Class-Imbalance Strategy
- Dataset Curation, Preprocessing & Augmentations
- Metrics Design & Evalutation Protocols
- Documentation (README.md)
- Drafting of the Final Published Research Paper
- Consulted on Dataset Clarifications with University of Waterloo School of Optometry Faculty Members

**Andrew Yang**
- Defined and Finalized the Primary Research Objective
- Model Training (WATGPU via SSH)
- Model Architecture Selection
- Model Architecture Visualization and Figures for Research Paper
- Literature Review and Research on Related Works
- Dataset Curation
- Drafting of the Final Published Research Paper

**Christopher Risi**
- Technical Support

**William Chiu**

**Tom Almog**

**Sidharth Shah**

**Apisan Kaneshan**

### Overview

This repository contains work in progress on the **semantic segmentation of microaneurysms, hemorrhages, soft exudates, and hard exudates** (lesions resulting from Diabetic Retinopathy) from fundus images. In addition to building the full segmentation pipeline, the project also conducts experimentation on techniques for enchancing the detection of microaneurysms. 

![Fundus Example](readme_images/sample4.png)

---

### Research Focus
Our research primarily focuses on:
- **Contrast enhancement strategies** to improve lesion visibility, including channel-aware selective preprocessing (CASP) and local contrast normalization (CLAHE).
- **Training-time techniques** to improve sensitivity to small structures, including loss functions designed to emphasize microaneurysm recall.

![CLAHE Demo](readme_images/clahe_demo2.png)

---

### Project Status/Progress (as of Feb 11 2026)
- [x] Dataset Curation 
- [x] Preprocessing and Augmentation Pipelines
- [x] Loss Function Development 
- [x] Data Visualizations
- [x] HydraLA-Net Implementation in PyTorch
- [x] Training Scripts 
- [x] Part 1: Baseline Training
- [x] Part 2: Preprocessing Variation Analysis
- [x] Part 3: Class Imbalance Aware Loss Function Analaysis
- [x] Part 4: Individual Dataset Analysis
- [x] Get results on testing sets
- [ ] Publish research paper!

---

### Model
The segmentation architecture used in this project is based directly on the original research paper that introduced it. 
* **LANet-DR** Paper: [Lesion-Aware Network for Diabetic Retinopathy Diagnosis](https://arxiv.org/abs/2408.07264) 

The GitHub repository containing the experiment conducted in the paper above can be found at:
[LANet-DR GitHub Repo](https://github.com/xia-xx-cv/LANet-DR/)

Additionally, a full dynamic implementation of our adapted HydraLA-Net can be found in this repository.

---

### Datasets
Three datasets are chosen for the project. All datasets contain fundus images and segmentation masks for microaneuryms, hemorrhages, soft exudates, and hard exudates. The IDRiD and DDR datasets contain the masks in the binary form. The TJDR dataset contains the masks as a image with color mappings.
* **IDRiD**: [Indian Diabetic Retinopathy Dataset from Kaggle](https://www.kaggle.com/dataset/saaryapatel98/indian-diabetic-retinopathy-image-dataset)
* **DDR**: [Diabetic Retinopathy Lesion Segmentation and Lesion Detection Dataset from GitHub](https://github.com/nkicsl/DDR-dataset/tree/master)
* **TJDR**: [TJDR: A High-Quality Diabetic Retinopathy Pixel-Level Annotation Dataset from GitHub](https://github.com/NekoPii/TJDR)

Dataset Pixel Distributions by Class
![Distribution](readme_images/pixel_distribution.png)

---

# HydraLA-Net Fundus Image Segmentation Application

This application performs semantic segmentation of diabetic retinopathy lesions in fundus images using **HydraLA-Net**.

## Features

- **Real-time Segmentation**: Upload fundus images and get instant lesion segmentation
- **4 Lesion Types**: Detects Hard Exudates (EX), Hemorrhages (HE), Microaneurysms (MA), and Soft Exudates (SE)
- **CLAHE Preprocessing**: Optional contrast enhancement for better lesion visibility
- **Adjustable Threshold**: Control detection sensitivity

## Setup

**Required:** Place trained model weights at `runs/4B/best_model.pt` before running the app.

## Usage

1. Upload a retinal fundus image (PNG, JPG, or JPEG)
2. Adjust preprocessing and threshold settings in the sidebar
3. Click "Segment Image" to see results

## Model Architecture

HydraLA-Net features:
- **ResNet-50 Backbone** for robust feature extraction
- **Feature Fusion Blocks (FFB)** for multi-scale integration
- **Lesion-Aware Modules (LAM)** for enhanced detection
- **Hydra Segmentation Head** with 4 class-specific branches

## About

This project was developed as part of diabetic retinopathy research. For more details, see the [full repository](https://github.com/jessicayuan1/fundus-image-segmentation).


---

### References
