import tensorflow as tf
from src.data_loader import preprocess_image
import argparse

def predict_xray(model_path, image_path):
    """Predict popcorn lung on a single X-ray."""
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Preprocess image
    img = preprocess_image(image_path)

    # Predict
    pred = model.predict(img[np.newaxis, ...])[0][0]
    label = 'Popcorn Lung' if pred > 0.5 else 'Healthy'
    return label, pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict popcorn lung from X-ray.')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--image', required=True, help='Path to X-ray image')
    args = parser.parse_args()

    label, prob = predict_xray(args.model, args.image)
    print(f'Prediction: {label} (Probability: {prob:.2f})')
