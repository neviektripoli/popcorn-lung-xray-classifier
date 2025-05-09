import os
import tensorflow as tf
from src.data_loader import load_data
from src.model import build_cnn
from src.utils import load_config
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import pickle
import datetime

def train_model(config):
    """Train the CNN model."""
    # Load data
    train_generator, validation_generator = load_data(
        config['data']['raw_dir'],
        config['data']['labels_file'],
        target_size=tuple(config['model']['input_shape'][:2])
    )

    # Build model
    model = build_cnn(tuple(config['model']['input_shape']))

    # Callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(config['output']['model_dir'], 'best_model.h5'),
        save_best_only=True,
        monitor='val_loss'
    )
    early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
    log_dir = os.path.join(config['output']['log_dir'], datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=log_dir)

    # Train
    history = model.fit(
        train_generator,
        epochs=config['model']['epochs'],
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, tensorboard]
    )

    # Save history
    with open(os.path.join(config['output']['log_dir'], 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)

    return model

if __name__ == '__main__':
    config = load_config()
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['log_dir'], exist_ok=True)
    train_model(config)
