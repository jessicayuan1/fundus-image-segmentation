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

This project was developed as part of diabetic retinopathy research. For more details, see the [full repository](https://github.com/yourusername/fundus-image-segmentation).
