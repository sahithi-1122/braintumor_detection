# config.py
import os

# Class configuration
CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
NUM_CLASSES = 4

# Model input sizes
MODEL_INPUT_SIZES = {
    "ann":    (64, 64),
    "cnn":    (128, 128),
    "resnet": (224, 224),
}

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 10
RANDOM_SEED = 42

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Directories
DATA_DIR = os.path.join("data")
SAVE_DIR = os.path.join("saved_models")
RESULTS_DIR = os.path.join("results")
REPORTS_DIR = os.path.join("reports")
