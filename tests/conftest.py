"""
Shared pytest fixtures for brain-tumor-detection-classification tests.
"""

import os

import cv2
import numpy as np
import pytest
import tensorflow as tf

from src.models.ann_model import build_ann_model
from src.models.cnn_model import build_cnn_model
from src.models.resnet_model import build_resnet_model

# Model name → (input_height, input_width)
MODEL_INPUT_SIZES = {
    "ann":    (64, 64),
    "cnn":    (128, 128),
    "resnet": (224, 224),
}


@pytest.fixture(scope="session")
def ann_model():
    """Randomly-initialized (untrained) ANN model."""
    return build_ann_model()


@pytest.fixture(scope="session")
def cnn_model():
    """Randomly-initialized (untrained) CNN model."""
    return build_cnn_model()


@pytest.fixture(scope="session")
def resnet_fixtures():
    """Randomly-initialized (untrained) ResNet50 model.  Returns (full_model, base_model)."""
    return build_resnet_model(weights=None)


@pytest.fixture(scope="session")
def resnet_model(resnet_fixtures):
    return resnet_fixtures[0]


@pytest.fixture(scope="session")
def resnet_base(resnet_fixtures):
    return resnet_fixtures[1]


def get_model_and_size(model_name: str, ann_model, cnn_model, resnet_model):
    """Return (model, (H, W, 3)) for the requested architecture."""
    h, w = MODEL_INPUT_SIZES[model_name]
    models = {"ann": ann_model, "cnn": cnn_model, "resnet": resnet_model}
    return models[model_name], (h, w, 3)


def save_random_image(tmp_path, shape=(64, 64)) -> str:
    """
    Save a random uint8 RGB image to a temporary JPEG file.

    Parameters
    ----------
    tmp_path:
        pytest ``tmp_path`` fixture value (pathlib.Path).
    shape:
        (height, width) tuple.

    Returns
    -------
    str
        Absolute path to the saved image file.
    """
    h, w = shape
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    # cv2 writes BGR so convert RGB → BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    path = str(tmp_path / "test_image.jpg")
    cv2.imwrite(path, img_bgr)
    return path
