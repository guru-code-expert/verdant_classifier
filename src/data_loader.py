"""Dataset loading and splitting utilities."""

import tensorflow as tf
from config.settings import *

def load_dataset(data_path: str = None, batch_size: int = BATCH_SIZE, label_mode: str = "int"):
    """
    Load an image dataset from directory using Keras utility.
    
    Args:
        data_path: Path to dataset root. Defaults to config value.
        batch_size: Batch size.
        label_mode: 'int' for sparse labels, 'categorical' for one-hot.
    
    Returns:
        tf.data.Dataset object
    """
    if data_path is None:
        data_path = str(DATA_ROOT / DATASET_NAME)
        
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        label_mode=label_mode
    )
    return dataset

def get_dataset_partitions_tf(ds: tf.data.Dataset,
                              train_split=TRAIN_SPLIT,
                              val_split=VAL_SPLIT,
                              test_split=TEST_SPLIT,
                              shuffle=True,
                              shuffle_size=SHUFFLE_BUFFER):
    """
    Split dataset into train/val/test with optional shuffling.
    
    Returns:
        train_ds, val_ds, test_ds
    """
    assert (train_split + val_split + test_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=SEED)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size + val_size)
    
    return train_ds, val_ds, test_ds

def apply_performance_optimizations(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Cache, shuffle, and prefetch for faster training."""
    return ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)