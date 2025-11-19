"""Central configuration file for the VerdantClassifier project."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"          # <-- Change this to your dataset location
DATASET_NAME = "sample_dataset"                   # Subfolder name inside DATA_ROOT

# Training hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 50

# Dataset split ratios
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# Model
NUM_CLASSES = 3          # Will be overridden automatically at runtime for multi-dataset support

# Random seed for reproducibility
SEED = 42
SHUFFLE_BUFFER = 1000