# Popcorn Lung X-ray Classifier

A deep learning system for detecting popcorn lung (bronchiolitis obliterans) from chest X-ray images using convolutional neural networks (CNNs).

## Project Structure
```bash
popcorn_lung_xray_classifier/
│
├── data/
│ ├── raw/ # Raw X-ray images
│ ├── processed/ # Preprocessed/resized images
│ ├── labels.csv # Labels (image filename, target)
│
├── notebooks/
│ ├── EDA.ipynb # Exploratory data analysis
│ ├── model_training.ipynb # Training and evaluation
│
├── src/
│ ├── init.py
│ ├── data_loader.py # Data loading and preprocessing
│ ├── model.py # CNN model architecture
│ ├── train.py # Training loop
│ ├── evaluate.py # Evaluation metrics
│ ├── utils.py # Helper functions
│
├── outputs/
│ ├── models/ # Saved models
│ ├── logs/ # Training logs
│ ├── results/ # Metrics and evaluation outputs
│
├── app/
│ ├── inference.py # Load model and run prediction on new X-ray
│ ├── web_interface.py # Streamlit web app
│
├── requirements.txt # Python dependencies
├── README.md # Project overview and instructions
└── config.yaml # Configuration file


Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/popcorn_lung_xray_classifier.git
   cd popcorn_lung_xray_classifier
