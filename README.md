# 🫁 Popcorn Lung X-ray Classifier

**A Deep Learning Project for Classifying Popcorn Lung (Bronchiolitis Obliterans) and Related Pulmonary Diseases from Chest X-rays.**

This repository contains a machine learning pipeline built in Python to detect Popcorn Lung and other lung conditions (such as Pneumonia, COPD, and general pulmonary abnormalities) using chest X-ray images and Convolutional Neural Networks (CNNs). The model utilizes transfer learning with pretrained backbones (EfficientNet, ResNet) and can be fine-tuned for real-world medical diagnostic support.

---

## 📌 Features

- 🧠 CNN-based lung disease classification
- 🖼️ Image preprocessing (resizing, denoising, contrast enhancement)
- 🔄 Data augmentation for improved generalization
- 📊 Training/evaluation with accuracy, AUC, F1-score, and confusion matrix
- 🌡️ Grad-CAM for model interpretability
- 📦 Easy-to-use Python project structure
- 🚀 (Optional) REST API using FastAPI for model deployment

---

## 📁 Dataset Sources

This project supports multiple open datasets:

- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXpert Dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [MIMIC-CXR (via PhysioNet)](https://physionet.org/content/mimic-cxr/)
- [VinDr-CXR](https://physionet.org/content/vindr-cxr/1.0.0/)

> **Note:** Some datasets require credentialed access or data usage agreements. Refer to the official source links.

---

## 🧠 Model Architecture

The model is built using:
- Pretrained CNNs: EfficientNetB0, ResNet50 (configurable)
- Loss function: Binary Cross Entropy / Categorical Cross Entropy
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau

---

## 🛠️ Installation

```bash
git clone https://github.com/your-username/popcorn-lung-xray-classifier.git
cd popcorn-lung-xray-classifier
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt


