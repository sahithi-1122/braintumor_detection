"""
ResNet50 Transfer Learning model for Brain Tumor Detection and Classification.

Phase 1: Frozen ResNet50 backbone, only the classification head is trained.
Phase 2 (fine-tuning): Upper backbone layers are unfrozen and retrained at a
lower learning rate.  Fine-tuning is orchestrated by src/trainer.py.
"""

import tensorflow as tf


def build_resnet_model(
    input_shape: tuple = (224, 224, 3),
    num_classes: int = 4,
    dense_units: int = 256,
    dropout_rate: float = 0.5,
    weights: str | None = "imagenet",
) -> tuple:
    """
    Build and return an uncompiled ResNet50 transfer-learning model.

    The ResNet50 backbone is loaded with ImageNet weights and all its layers
    are frozen for Phase 1 training.  The caller (trainer) is responsible for
    unfreezing upper layers before Phase 2 fine-tuning.

    Architecture
    ------------
    ResNet50(imagenet, include_top=False) [frozen]
    → GlobalAveragePooling2D
    → Dense(256, relu)
    → Dropout(0.5)
    → Dense(4, softmax)

    Parameters
    ----------
    input_shape:
        Spatial dimensions of the input image (H, W, C).  Default (224, 224, 3).
    num_classes:
        Number of output classes.  Default 4.
    dense_units:
        Number of units in the intermediate Dense layer.  Default 256.
    dropout_rate:
        Dropout probability before the output layer.  Default 0.5.

    Returns
    -------
    tuple[tf.keras.Model, tf.keras.Model]
        ``(full_model, base_model)`` where ``full_model`` is the complete
        trainable model and ``base_model`` is the ResNet50 backbone exposed
        so that ``trainer.fine_tune_resnet()`` can unfreeze its upper layers.
    """
    # Load pretrained backbone — no top classification layers
    base_model = tf.keras.applications.ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape,
    )

    # Freeze all backbone layers for Phase 1
    base_model.trainable = False

    # Build classification head using the Functional API
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = base_model(inputs, training=False)  # training=False keeps BN frozen
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense_head")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_head")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

    full_model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet_model")

    return full_model, base_model
