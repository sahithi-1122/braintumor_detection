"""
Custom CNN model for Brain Tumor Detection and Classification.

Three convolutional blocks with BatchNormalization and MaxPooling extract
hierarchical spatial features from 128×128 MRI images.
"""

import tensorflow as tf


def build_cnn_model(
    input_shape: tuple = (128, 128, 3),
    num_classes: int = 4,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    """
    Build and return an uncompiled custom CNN model.

    Architecture
    ------------
    Input(128, 128, 3)
    Block 1: Conv2D(32, 3×3, same) → BatchNorm → ReLU → MaxPooling2D(2×2)  → 64×64
    Block 2: Conv2D(64, 3×3, same) → BatchNorm → ReLU → MaxPooling2D(2×2)  → 32×32
    Block 3: Conv2D(128, 3×3, same) → BatchNorm → ReLU → MaxPooling2D(2×2) → 16×16
    GlobalAveragePooling2D → Dropout(0.4) → Dense(256, relu) → Dense(4, softmax)

    Parameters
    ----------
    input_shape:
        Spatial dimensions of the input image (H, W, C).  Default (128, 128, 3).
    num_classes:
        Number of output classes.  Default 4.
    dropout_rate:
        Dropout probability before the final Dense layer.  Default 0.4.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras model ready for ``model.compile()``.
    """
    inputs = tf.keras.Input(shape=input_shape, name="input")

    # Block 1 — 32 filters, 128×128 → 64×64
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", use_bias=False, name="conv1")(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.Activation("relu", name="relu1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2 — 64 filters, 64×64 → 32×32
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", use_bias=False, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.Activation("relu", name="relu2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3 — 128 filters, 32×32 → 16×16
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", use_bias=False, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.Activation("relu", name="relu3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Classification head
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="dense")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cnn_model")
    return model
