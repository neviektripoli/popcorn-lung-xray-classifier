import os
import cv2
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

class DataLoader:
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.img_size = self.config['data']['img_size']
        self.batch_size = self.config['data']['batch_size']
        
    def load_data(self):
        """Load and preprocess the data"""
        # Read labels
        df = pd.read_csv(self.config['data']['labels_file'])
        
        # Load and preprocess images
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            img_path = os.path.join(self.config['data']['raw_path'], row['filename'])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize and normalize
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img / 255.0
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                
                images.append(img)
                labels.append(row['target'])
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # One-hot encode labels if needed
        if len(np.unique(labels)) > 2:
            labels = to_categorical(labels)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=labels
        )
        
        return X_train, X_test, y_train, y_test
    
    def data_generator(self, X, y, batch_size=32, shuffle=True):
        """Create a data generator for training"""
        num_samples = len(X)
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield X[batch_indices], y[batch_indices]
