import os
import tensorflow as tf
from src.data_loader import load_data
from src.utils import load_config
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(config):
    """Evaluate the trained model."""
    # Load data
    _, test_generator = load_data(
        config['data']['raw_dir'],
        config['data']['labels_file'],
        target_size=tuple(config['model']['input_shape'][:2])
    )

    # Load model
    model = tf.keras.models.load_model(os.path.join(config['output']['model_dir'], 'best_model.h5'))

    # Predictions
    y_pred = model.predict(test_generator)
    y_true = test_generator.classes
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred_binary)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (AUC: {auc:.2f})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(config['output']['result_dir'], 'confusion_matrix.png'))
    plt.close()

    # Save metrics
    with open(os.path.join(config['output']['result_dir'], 'metrics.txt'), 'w') as f:
        f.write(f'AUC: {auc:.2f}\n')

if __name__ == '__main__':
    config = load_config()
    os.makedirs(config['output']['result_dir'], exist_ok=True)
    evaluate_model(config)
