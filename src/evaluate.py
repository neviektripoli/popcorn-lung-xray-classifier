import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
import yaml
import tensorflow as tf

class ModelEvaluator:
    def __init__(self, model_path, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = tf.keras.models.load_model(model_path)
        self.data_loader = DataLoader()
        
    def evaluate_model(self):
        """Evaluate the model on test data"""
        # Load test data
        _, X_test, _, y_test = self.data_loader.load_data()
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Classification report
        report = classification_report(y_test, y_pred_classes, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = os.path.join(self.config['paths']['results'], 'classification_report.csv')
        report_df.to_csv(report_path)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        self.plot_confusion_matrix(cm)
        
        # ROC curve
        self.plot_roc_curve(y_test, y_pred)
        
        return report
    
    def plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        save_path = os.path.join(self.config['paths']['results'], 'confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        save_path = os.path.join(self.config['paths']['results'], 'roc_curve.png')
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    evaluator = ModelEvaluator('outputs/models/best_model.h5')
    evaluator.evaluate_model()
