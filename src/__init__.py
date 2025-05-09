"""
Popcorn Lung X-ray Classifier Package

This package contains all the core functionality for:
- Loading and preprocessing X-ray images
- Building and training CNN models
- Evaluating model performance
- Utility functions for image processing
"""

# Import key classes/functions to make them available at package level
from .data_loader import DataLoader
from .model import PopcornLungModel
from .evaluate import ModelEvaluator
from .utils import (
    plot_training_history,
    preprocess_image
)

# Version of the package
__version__ = "0.1.0"

# Package metadata
__author__ = "Your Name"
__email__ = "your.email@example.com"
__license__ = "MIT"
__description__ = "A CNN-based system for detecting popcorn lung from chest X-rays"

# List of what's available when someone does 'from src import *'
__all__ = [
    'DataLoader',
    'PopcornLungModel',
    'ModelEvaluator',
    'plot_training_history',
    'preprocess_image'
]
