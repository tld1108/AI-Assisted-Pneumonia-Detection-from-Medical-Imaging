# AI Assisted Pneumonia Detection from Medical Imaging

This is a PyTorch project for detecting and visualizing pneumonia regions in chest X-ray images using a U-Net segmentation model.

---

## Project Overview

This repository contains code for:

- Training a U-Net model (`train.py`)
- Dataset loader (`dataset.py`)
- Evaluating & visualizing results (`evaluate.py`)
- Basic data checking (`check_data.py`)
- Generating approximate masks (`generate_masks.py`)
- Testing dataset loader (`test_dataset.py`)

---

## Model Details

- Model architecture: **U-Net** with ResNet34 encoder  
- Loss: Dice Loss  
- Task: **Segmentation mask prediction** for pneumonia areas  
- Framework: PyTorch + segmentation_models_pytorch  
- Image size: 256×256 pixels :contentReference[oaicite:6]{index=6}

---

## Dataset Requirements

You must download the **Chest X-Ray pneumonia dataset** from Kaggle (or another source) (dataset that i used: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and place images in:
- data/chest_xray/train/NORMAL
- data/chest_xray/train/PNEUMONIA
- data/chest_xray/test/NORMAL
- data/chest_xray/test/PNEUMONIA

Then use `generate_masks.py` to create matching masks and store them in:
- masks/train/NORMAL
- masks/train/PNEUMONIA
  
> Note: These masks are simple threshold masks and not expert annotations.

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## Run Training
```bash
python train.py
```
---

## Evaluation & Visualization
```bash
python evaluate.py
```
---
## NOTES!!

This project uses simple threshold masks, not medical-grade annotations. It is intended for learning and experimentation.


