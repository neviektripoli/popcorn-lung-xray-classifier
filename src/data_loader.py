import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single X-ray image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

def create_data_generator():
    """Create data generator with augmentations."""
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

def load_data(data_dir, labels_file, target_size=(224, 224)):
    """Load and prepare data for training."""
    labels_df = pd.read_csv(labels_file)
    datagen = create_data_generator()

    train_generator = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=data_dir,
        x_col='filename',
        y_col='target',
        target_size=target_size,
        color_mode='grayscale',
        class_mode='binary',
        batch_size=32,
        subset='training',
        shuffle=True
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=data_dir,
        x_col='filename',
        y_col='target',
        target_size=target_size,
        color_mode='grayscale',
        class_mode='binary',
        batch_size=32,
        subset='validation',
        shuffle=False
    )

    return train_generator, validation_generator
