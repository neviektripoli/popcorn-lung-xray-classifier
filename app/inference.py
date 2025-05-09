import os
import numpy as np
import tensorflow as tf
from src.utils import preprocess_image

class PopcornLungPredictor:
    def __init__(self, model_path='outputs/models/best_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = 224  # Should match training size
    
    def predict(self, image_path):
        """Predict popcorn lung probability for a given X-ray image"""
        # Preprocess the image
        img = preprocess_image(image_path, self.img_size)
        if img is None:
            return {"error": "Could not load or process the image"}
        
        # Make prediction
        prediction = self.model.predict(img)[0][0]
        probability = float(prediction)
        
        # Return results
        return {
            "probability": probability,
            "diagnosis": "Positive" if probability > 0.5 else "Negative",
            "confidence": probability if probability > 0.5 else 1 - probability
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Popcorn Lung X-ray Classifier')
    parser.add_argument('image_path', type=str, help='Path to the chest X-ray image')
    args = parser.parse_args()
    
    predictor = PopcornLungPredictor()
    result = predictor.predict(args.image_path)
    
    print("\nPrediction Results:")
    print(f"Probability of Popcorn Lung: {result['probability']:.4f}")
    print(f"Diagnosis: {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")
