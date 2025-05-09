# Popcorn Lung X-ray Classifier

This project uses a Convolutional Neural Network (CNN) to detect popcorn lung (bronchiolitis obliterans) from chest X-ray images, targeting risks associated with frequent vaping.

## Project Structure
- `data/`: Raw and processed X-ray images, labels (`labels.csv`)
- `notebooks/`: Exploratory data analysis (`EDA.ipynb`)
- `src/`: Core scripts for data loading, model, training, and evaluation
- `outputs/`: Trained models, logs, and results
- `app/`: Inference script and Streamlit web interface
- `requirements.txt`: Python dependencies
- `config.yaml`: Configuration file
- `README.md`: This file

## Setup
1. Clone the repository: `git clone <repo_url>`
2. Install dependencies: `pip install -r requirements.txt`
3. Place X-ray images in `data/raw/` and update `data/labels.csv` with filenames and labels (0 for healthy, 1 for popcorn lung).

## Usage
- **Exploratory Analysis**: Run `jupyter notebook notebooks/EDA.ipynb` to visualize data.
- **Train Model**: Run `python src/train.py` to train the CNN.
- **Evaluate Model**: Run `python src/evaluate.py` to compute metrics and generate plots.
- **Predict on Single Image**: Run `python app/inference.py --model outputs/models/best_model.h5 --image data/raw/image1.png`
- **Launch Web App**: Run `streamlit run app/web_interface.py` to use the web interface.

## Requirements
- Python 3.9+
- GPU recommended for training
- Ensure X-ray images are in PNG or JPG format

## Notes
- The dataset should include labeled X-ray images. Due to the rarity of popcorn lung, consider synthetic data or transfer learning with pre-trained models (e.g., ResNet50).
- Results are saved in `outputs/results/` (e.g., confusion matrix, AUC).
- The web app allows uploading X-rays for predictions.

## Citation
Inspired by research on bronchiolitis obliterans detection using deep learning ([Nature Communications Medicine](https://www.nature.com/articles/s43856-025-00732-x)) and vaping-related lung risks ([University Hospitals](https://www.uhhospitals.org/blog/articles/2024/04/popcorn-lung-a-dangerous-risk-of-vaping)).