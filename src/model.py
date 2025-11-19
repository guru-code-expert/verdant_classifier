"""CNN model definition."""

import tensorflow as tf
from tensorflow.keras import models, layers
from config.settings import IMAGE_SIZE, CHANNELS, BATCH_SIZE
from src.preprocessing import get_resizing_and_rescaling

def build_cnn_model(num_classes: int, input_shape=None):
    """
    Build a simple but effective CNN for image classification.
    
    Args:
        num_classes: Number of target classes.
        input_shape: Optional custom input shape.
    
    Returns:
        Uncompiled tf.keras.Model
    """
    if input_shape is None:
        input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    
    resize_and_rescale = get_resizing_and_rescaling()
    
    model = models.Sequential([
        resize_and_rescale,
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ], name="VerdantCNN")
    
    model.build(input_shape)
    return model