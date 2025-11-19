"""Preprocessing and augmentation layers."""

import tensorflow as tf
from tensorflow.keras import layers
from config.settings import IMAGE_SIZE

def get_resizing_and_rescaling():
    """Standard resizing + normalization layer."""
    return tf.keras.Sequential([
        layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        layers.Rescaling(1./255),
    ], name="resize_and_rescale")

def get_data_augmentation():
    """Random flip & rotation for robustness."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ], name="data_augmentation")