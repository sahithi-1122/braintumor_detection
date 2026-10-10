"""
ANN/MLP baseline model for Brain Tumor Detection and Classification.

Provides a fully-connected feedforward network as a lower-bound baseline.
Images are flattened to a 1-D vector; no spatial structure is exploited.
"""

import tensorflow as tf


def build_ann_model(
    input_shape: tuple = (64, 64, 3),
    num_classes: int = 4,
    dense_units: list = None,
    dropout_rate: float = 0.4,
) -> tf.keras.Model:
    """
    Build and return an uncompiled ANN/MLP model.

    Architecture
    ------------
    Input(64, 64, 3) → Flatten → Dense(512, relu) → Dropout(0.4)
                     → Dense(256, relu) → Dropout(0.4)
                     → Dense(4, softmax)

    Parameters
    ----------
    input_shape:
        Spatial dimensions of the input image (H, W, C).  Default (64, 64, 3).
    num_classes:
        Number of output classes.  Default 4.
    dense_units:
        List of hidden-layer sizes.  Default [512, 256].
    dropout_rate:
        Dropout probability applied after each hidden Dense layer.  Default 0.4.

    Returns
    -------
    tf.keras.Model
        Uncompiled Keras model ready for ``model.compile()``.
    """
    if dense_units is None:
        dense_units = [512, 256]

    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = tf.keras.layers.Flatten(name="flatten")(inputs)

    for i, units in enumerate(dense_units):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i+1}")(x)
        x = tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ann_model")
    return model
