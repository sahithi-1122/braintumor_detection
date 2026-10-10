"""
Preprocessor module for Brain Tumor Detection and Classification.

Handles all image loading, resizing, normalization, and augmentation via tf.data.
"""

import os

import cv2
import numpy as np
import tensorflow as tf

from config import MODEL_INPUT_SIZES

# Re-export for convenience so other modules can import from here
__all__ = [
    "MODEL_INPUT_SIZES",
    "load_and_preprocess_image",
    "preprocess_single_image",
    "build_train_dataset",
    "build_val_dataset",
    "build_test_dataset",
]

# ---------------------------------------------------------------------------
# Core image loading / preprocessing
# ---------------------------------------------------------------------------


def load_and_preprocess_image(
    path: str,
    target_size: tuple,
    model_name: str | None = None,
) -> np.ndarray:
    """
    Load an image from *path* using OpenCV, resize, convert BGR→RGB, and
    normalise pixel values to [0.0, 1.0].

    Parameters
    ----------
    path:
        Absolute or relative path to the image file.
    target_size:
        ``(height, width)`` tuple for the desired output spatial dimensions.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(H, W, 3)`` with values in ``[0.0, 1.0]``.

    Raises
    ------
    ValueError
        If the file does not exist or OpenCV cannot decode the image.
    """
    if not os.path.exists(path):
        raise ValueError(
            f"Image not found: {path}"
        )

    img = cv2.imread(path)
    if img is None:
        raise ValueError(
            f"Failed to decode image at {path}: image cannot be read or is corrupted"
        )

    h, w = target_size
    img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32)
    if model_name == "resnet":
        # ResNet50 ImageNet weights use Caffe preprocessing: RGB input in
        # [0, 255], channel order BGR, then ImageNet channel means subtracted.
        return tf.keras.applications.resnet50.preprocess_input(img_float)
    return img_float / 255.0


def preprocess_single_image(
    path: str,
    model_name: str,
) -> np.ndarray:
    """
    Convenience wrapper used by the Predictor CLI.

    Validates *model_name*, calls :func:`load_and_preprocess_image` with the
    correct target size, and returns a batch-ready array.

    Parameters
    ----------
    path:
        Path to the input image file.
    model_name:
        One of ``"ann"``, ``"cnn"``, or ``"resnet"``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(1, H, W, 3)`` ready for ``model.predict()``.

    Raises
    ------
    ValueError
        If *model_name* is not in :data:`MODEL_INPUT_SIZES`.
    ValueError
        If the image cannot be read (propagated from
        :func:`load_and_preprocess_image`).
    """
    if model_name not in MODEL_INPUT_SIZES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {', '.join(sorted(MODEL_INPUT_SIZES.keys()))}"
        )

    target_size = MODEL_INPUT_SIZES[model_name]
    img = load_and_preprocess_image(path, target_size, model_name=model_name)
    return np.expand_dims(img, axis=0)  # shape: (1, H, W, 3)


# ---------------------------------------------------------------------------
# tf.data.Dataset builders
# ---------------------------------------------------------------------------

def _make_load_fn(target_size: tuple, model_name: str):
    """
    Return a Python function that loads and preprocesses one image given its
    file path as a bytes tensor.  Intended for use with ``tf.py_function``.
    """
    def _load(path_tensor, label_tensor):
        # Decode tensors to Python scalars
        path_str = path_tensor.numpy().decode("utf-8")
        label_int = int(label_tensor.numpy())
        # Keep pixels in [0, 1] until augmentation is complete. ResNet's
        # ImageNet transform is applied afterwards, matching inference.
        img = load_and_preprocess_image(path_str, target_size)
        return img, label_int

    return _load


def _build_dataset_from_samples(
    samples: list,
    model_name: str,
    num_classes: int,
) -> tf.data.Dataset:
    """
    Build a base (unbatched, unshuffled) tf.data.Dataset from a list of
    ``(path, label_int)`` tuples.

    Each element of the dataset is ``(image_tensor, one_hot_label)``.
    """
    target_size = MODEL_INPUT_SIZES[model_name]
    h, w = target_size

    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    load_fn = _make_load_fn(target_size, model_name)

    def _load_tf(path_t, label_t):
        img, label = tf.py_function(
            func=load_fn,
            inp=[path_t, label_t],
            Tout=[tf.float32, tf.int32],
        )
        img.set_shape([h, w, 3])
        label.set_shape([])
        one_hot = tf.keras.utils.to_categorical(label, num_classes=num_classes)
        one_hot = tf.cast(one_hot, tf.float32)
        one_hot.set_shape([num_classes])
        return img, one_hot

    ds = ds.map(_load_tf, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def _apply_model_preprocessing(images: tf.Tensor, model_name: str) -> tf.Tensor:
    if model_name == "resnet":
        return tf.keras.applications.resnet50.preprocess_input(images * 255.0)
    return images


def _build_augmentation_layer() -> tf.keras.Sequential:
    """
    Construct and return the augmentation ``tf.keras.Sequential`` pipeline.

    Layers
    ------
    * ``RandomFlip("horizontal")``
    * ``RandomRotation(0.083)``  — ≈ ±15°
    * ``RandomZoom(0.10)``
    * ``RandomBrightness(0.10)`` (falls back to a Lambda layer if unavailable)
    """
    layers = [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.083),
        tf.keras.layers.RandomZoom(0.10),
    ]

    # Brightness delta is expressed in the normalized [0, 1] input scale.
    def _random_brightness(x, training=None):
        if training:
            x = tf.image.random_brightness(x, max_delta=0.10)
            return tf.clip_by_value(x, 0.0, 1.0)
        return x
    layers.append(tf.keras.layers.Lambda(_random_brightness))

    return tf.keras.Sequential(layers)


def build_train_dataset(
    samples: list,
    model_name: str,
    batch_size: int = 32,
    num_classes: int = 4,
) -> tf.data.Dataset:
    """
    Build a shuffled, augmented, batched ``tf.data.Dataset`` for training.

    Processing order
    ----------------
    1. Load and preprocess each image (resize, BGR→RGB, normalize).
    2. One-hot encode labels.
    3. Shuffle with ``buffer_size=len(samples)``.
    4. Batch.
    5. Apply augmentation per batch via the Keras augmentation pipeline.
    6. Prefetch.

    Parameters
    ----------
    samples:
        List of ``(path, class_index)`` tuples.
    model_name:
        Architecture key — determines image resolution.
    batch_size:
        Number of samples per batch (default 32).
    num_classes:
        Number of output classes (default 4).

    Returns
    -------
    tf.data.Dataset
        Yields ``(image_batch, one_hot_label_batch)`` pairs.
    """
    if model_name not in MODEL_INPUT_SIZES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {', '.join(sorted(MODEL_INPUT_SIZES.keys()))}"
        )

    ds = _build_dataset_from_samples(samples, model_name, num_classes)
    ds = ds.shuffle(buffer_size=len(samples), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)

    augmentation = _build_augmentation_layer()

    def _augment(images, labels):
        images = augmentation(images, training=True)
        images = _apply_model_preprocessing(images, model_name)
        return images, labels

    ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_val_dataset(
    samples: list,
    model_name: str,
    batch_size: int = 32,
    num_classes: int = 4,
) -> tf.data.Dataset:
    """
    Build a non-augmented, batched ``tf.data.Dataset`` for validation.

    No shuffle is applied so that validation metrics are deterministic.

    Parameters
    ----------
    samples:
        List of ``(path, class_index)`` tuples.
    model_name:
        Architecture key — determines image resolution.
    batch_size:
        Number of samples per batch (default 32).
    num_classes:
        Number of output classes (default 4).

    Returns
    -------
    tf.data.Dataset
        Yields ``(image_batch, one_hot_label_batch)`` pairs.
    """
    if model_name not in MODEL_INPUT_SIZES:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid options: {', '.join(sorted(MODEL_INPUT_SIZES.keys()))}"
        )

    ds = _build_dataset_from_samples(samples, model_name, num_classes)
    ds = ds.map(
        lambda images, labels: (_apply_model_preprocessing(images, model_name), labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_test_dataset(
    samples: list,
    model_name: str,
    batch_size: int = 32,
    num_classes: int = 4,
) -> tf.data.Dataset:
    """
    Build an ordered, non-augmented, batched ``tf.data.Dataset`` for testing.

    Identical to :func:`build_val_dataset` — ordered (no shuffle),
    no augmentation.

    Parameters
    ----------
    samples:
        List of ``(path, class_index)`` tuples.
    model_name:
        Architecture key — determines image resolution.
    batch_size:
        Number of samples per batch (default 32).
    num_classes:
        Number of output classes (default 4).

    Returns
    -------
    tf.data.Dataset
        Yields ``(image_batch, one_hot_label_batch)`` pairs.
    """
    return build_val_dataset(
        samples,
        model_name=model_name,
        batch_size=batch_size,
        num_classes=num_classes,
    )
