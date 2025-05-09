import os
import yaml
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from .data_loader import DataLoader
from .model import PopcornLungModel
from .utils import plot_training_history

def train_model():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories if they don't exist
    os.makedirs(config['paths']['model_save'], exist_ok=True)
    os.makedirs(config['paths']['logs'], exist_ok=True)
    os.makedirs(config['paths']['results'], exist_ok=True)
    
    # Initialize data loader and load data
    data_loader = DataLoader()
    X_train, X_test, y_train, y_test = data_loader.load_data()
    
    # Initialize and build model
    model_builder = PopcornLungModel()
    model = model_builder.build_cnn()
    model.summary()
    
    # Prepare callbacks
    callbacks = model_builder.get_callbacks()
    
    # Add TensorBoard callback
    log_dir = os.path.join(
        config['paths']['logs'],
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    callbacks.append(TensorBoard(log_dir=log_dir))
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config['model']['epochs'],
        batch_size=config['data']['batch_size'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history plots
    plot_training_history(history, config['paths']['results'])
    
    return model, history

if __name__ == "__main__":
    train_model()
