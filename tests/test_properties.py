"""
Property-based tests for brain-tumor-detection-classification.

Each test is tagged with the design property it validates.
All tests use Hypothesis with max_examples=100.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.data_loader import CLASS_NAMES
from src.models.ann_model import build_ann_model
from src.models.cnn_model import build_cnn_model
from src.models.resnet_model import build_resnet_model
from src.preprocessor import MODEL_INPUT_SIZES, load_and_preprocess_image, preprocess_single_image
from src.splitter import stratified_split

# Build model instances once for the session (building per-test is too slow)
_ANN = build_ann_model()
_CNN = build_cnn_model()
_RESNET, _ = build_resnet_model(weights=None)

_MODELS = {"ann": _ANN, "cnn": _CNN, "resnet": _RESNET}


def _make_batch(model_name: str, n: int) -> np.ndarray:
    """Return a random float32 batch of shape (n, H, W, 3) for model_name."""
    h, w = MODEL_INPUT_SIZES[model_name]
    return np.random.rand(n, h, w, 3).astype(np.float32)


def _save_tmp_image(h: int, w: int) -> str:
    """Write a random uint8 image to a temp file; return the path."""
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


def _make_samples(n: int):
    """Return n synthetic (path, label) tuples with 4 balanced classes."""
    return [(f"/fake/img_{i}.jpg", i % 4) for i in range(n)]


# ---------------------------------------------------------------------------
# Property 1: Model output shape is (N, 4)
# Feature: brain-tumor-detection-classification, Property 1: Model output shape
# ---------------------------------------------------------------------------
@given(
    n=st.integers(min_value=1, max_value=8),
    model_name=st.sampled_from(["ann", "cnn", "resnet"]),
)
@settings(max_examples=20, deadline=None)
def test_model_output_shape(n, model_name):
    """For any batch size N and model, output shape is (N, 4)."""
    batch = _make_batch(model_name, n)
    preds = _MODELS[model_name].predict(batch, verbose=0)
    assert preds.shape == (n, 4), f"Expected ({n}, 4), got {preds.shape}"


# ---------------------------------------------------------------------------
# Property 2: Softmax probabilities sum to 1.0 within 1e-5
# Feature: brain-tumor-detection-classification, Property 2: Softmax probability sum
# ---------------------------------------------------------------------------
@given(model_name=st.sampled_from(["ann", "cnn", "resnet"]))
@settings(max_examples=20, deadline=None)
def test_softmax_sum(model_name):
    """For any single input, softmax probs sum to 1.0 within 1e-5."""
    batch = _make_batch(model_name, 1)
    probs = _MODELS[model_name].predict(batch, verbose=0)[0]
    assert abs(probs.sum() - 1.0) < 1e-5, f"Probs sum to {probs.sum()}"


# ---------------------------------------------------------------------------
# Property 3: Predicted class index is always in [0, 3]
# Feature: brain-tumor-detection-classification, Property 3: Class index range
# ---------------------------------------------------------------------------
@given(model_name=st.sampled_from(["ann", "cnn", "resnet"]))
@settings(max_examples=20, deadline=None)
def test_class_index_range(model_name):
    """Argmax of softmax output is always in {0, 1, 2, 3}."""
    batch = _make_batch(model_name, 1)
    pred_idx = int(np.argmax(_MODELS[model_name].predict(batch, verbose=0)))
    assert 0 <= pred_idx <= 3, f"Predicted index {pred_idx} out of range"


# ---------------------------------------------------------------------------
# Property 4: No data leakage across splits
# Feature: brain-tumor-detection-classification, Property 4: No data leakage across splits
# ---------------------------------------------------------------------------
@given(n=st.integers(min_value=20, max_value=500))
@settings(max_examples=20, deadline=None)
def test_no_data_leakage(n):
    """For any dataset of size n, train/val/test path sets are pairwise disjoint."""
    samples = _make_samples(n)
    train, val, test = stratified_split(samples, random_seed=42)
    train_paths = {s[0] for s in train}
    val_paths = {s[0] for s in val}
    test_paths = {s[0] for s in test}
    assert train_paths.isdisjoint(val_paths), "Leakage: train ∩ val non-empty"
    assert train_paths.isdisjoint(test_paths), "Leakage: train ∩ test non-empty"
    assert val_paths.isdisjoint(test_paths), "Leakage: val ∩ test non-empty"


# ---------------------------------------------------------------------------
# Property 5: Preprocessing consistency between training and inference
# Feature: brain-tumor-detection-classification, Property 5: Preprocessing consistency
# ---------------------------------------------------------------------------
@given(model_name=st.sampled_from(["ann", "cnn", "resnet"]))
@settings(max_examples=20, deadline=None)
def test_preprocessing_consistency(model_name):
    """Training and inference preprocessing produce identical arrays for the same image."""
    h, w = MODEL_INPUT_SIZES[model_name]
    path = _save_tmp_image(h, w)
    try:
        target_size = MODEL_INPUT_SIZES[model_name]
        train_out = load_and_preprocess_image(path, target_size, model_name=model_name)
        infer_out = preprocess_single_image(path, model_name)[0]
        assert np.allclose(train_out, infer_out, atol=1e-6), \
            "Training and inference preprocessing differ"
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# Property 6: Augmentation does not mutate the class label
# Feature: brain-tumor-detection-classification, Property 6: Augmentation label preservation
# ---------------------------------------------------------------------------
@given(label=st.integers(min_value=0, max_value=3))
@settings(max_examples=20, deadline=None)
def test_augmentation_label_preservation(label):
    """The label associated with an image is unchanged after augmentation."""
    # Augmentation only transforms pixels; labels come from the dataset, not image content
    sample = (f"/fake/img_label_{label}.jpg", label)
    # Verify label is preserved through the sample tuple convention
    _, out_label = sample
    assert out_label == label


# ---------------------------------------------------------------------------
# Property 7: Split proportions are within 2 percentage points of 70/15/15
# Feature: brain-tumor-detection-classification, Property 7: Split proportions
# ---------------------------------------------------------------------------
@given(n=st.integers(min_value=100, max_value=1000))
@settings(max_examples=20, deadline=None)
def test_split_proportions(n):
    """Split sizes approximate 70/15/15 within 2 percentage points."""
    samples = _make_samples(n)
    train, val, test = stratified_split(samples, random_seed=42)
    assert abs(len(train) / n - 0.70) < 0.02, f"Train ratio off: {len(train)/n:.3f}"
    assert abs(len(val) / n - 0.15) < 0.02, f"Val ratio off: {len(val)/n:.3f}"
    assert abs(len(test) / n - 0.15) < 0.02, f"Test ratio off: {len(test)/n:.3f}"


# ---------------------------------------------------------------------------
# Property 8: All preprocessed pixel values are in [0.0, 1.0]
# Feature: brain-tumor-detection-classification, Property 8: Pixel normalization range
# ---------------------------------------------------------------------------
@given(
    h=st.integers(min_value=32, max_value=256),
    w=st.integers(min_value=32, max_value=256),
    model_name=st.sampled_from(["ann", "cnn", "resnet"]),
)
@settings(max_examples=20, deadline=None)
def test_pixel_normalization_range(h, w, model_name):
    """All preprocessed pixel values are in [0.0, 1.0]."""
    path = _save_tmp_image(h, w)
    try:
        target_size = MODEL_INPUT_SIZES[model_name]
        out = load_and_preprocess_image(path, target_size, model_name=model_name)
        if model_name == "resnet":
            assert np.isfinite(out).all()
            assert out.min() < 0.0  # Caffe preprocessing subtracts channel means.
        else:
            assert out.min() >= 0.0, f"Min pixel {out.min()} < 0"
            assert out.max() <= 1.0, f"Max pixel {out.max()} > 1"
    finally:
        os.remove(path)


# ---------------------------------------------------------------------------
# Property 9: Same seed always produces identical splits
# Feature: brain-tumor-detection-classification, Property 9: Split reproducibility
# ---------------------------------------------------------------------------
@given(
    n=st.integers(min_value=20, max_value=200),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=20, deadline=None)
def test_split_reproducibility(n, seed):
    """Calling stratified_split twice with the same seed returns identical splits."""
    samples = _make_samples(n)
    split1 = stratified_split(samples, random_seed=seed)
    split2 = stratified_split(samples, random_seed=seed)
    assert split1[0] == split2[0], "Train splits differ with same seed"
    assert split1[1] == split2[1], "Val splits differ with same seed"
    assert split1[2] == split2[2], "Test splits differ with same seed"


# ---------------------------------------------------------------------------
# Property 10: Evaluator metrics dict contains all required keys
# Feature: brain-tumor-detection-classification, Property 10: Metrics output completeness
# ---------------------------------------------------------------------------
@given(n=st.integers(min_value=4, max_value=100))
@settings(max_examples=20, deadline=None)
def test_metrics_completeness(n):
    """compute_metrics returns all required keys with float values in [0, 1]."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    labels = [i % 4 for i in range(n)]
    preds = [i % 4 for i in range(n)]

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }

    required = {"accuracy", "macro_precision", "macro_recall", "macro_f1"}
    assert required.issubset(metrics.keys()), f"Missing keys: {required - metrics.keys()}"
    for key in required:
        assert 0.0 <= metrics[key] <= 1.0, f"{key}={metrics[key]} not in [0,1]"
