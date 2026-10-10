"""
Trainer module for Brain Tumor Detection and Classification.

Handles model compilation, callbacks, training loop, fine-tuning, and
model persistence for all three architectures (ANN, CNN, ResNet50).
"""

import os
import random

import numpy as np
import tensorflow as tf


def set_global_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow to support reproducibility.

    Parameters
    ----------
    seed:
        Integer seed value.  Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
) -> None:
    """
    Compile *model* in-place with Adam optimizer and categorical cross-entropy.

    Parameters
    ----------
    model:
        Keras model to compile.
    learning_rate:
        Initial learning rate for the Adam optimizer.  Default 1e-3.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def get_callbacks(
    checkpoint_path: str,
    patience: int = 10,
) -> list:
    """
    Return a list of standard training callbacks.

    Callbacks
    ---------
    * ``EarlyStopping`` — monitors ``val_loss``, restores best weights.
    * ``ModelCheckpoint`` — saves only the best weights to *checkpoint_path*.

    Parameters
    ----------
    checkpoint_path:
        File path where the best model weights will be saved.
    patience:
        Number of epochs with no improvement before early stopping.  Default 10.

    Returns
    -------
    list[tf.keras.callbacks.Callback]
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    return [early_stop, checkpoint]


def train(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_name: str,
    epochs: int = 50,
    patience: int = 10,
    learning_rate: float = 1e-3,
    seed: int = 42,
    save_dir: str = "saved_models",
) -> tf.keras.callbacks.History:
    """
    Full training loop: seed → compile → callbacks → fit → save.

    Parameters
    ----------
    model:
        Uncompiled (or already compiled) Keras model.
    train_ds:
        Training ``tf.data.Dataset`` yielding ``(images, one_hot_labels)``.
    val_ds:
        Validation ``tf.data.Dataset``.
    model_name:
        Architecture identifier (``"ann"``, ``"cnn"``, or ``"resnet"``).
        Used to name the save directory.
    epochs:
        Maximum number of training epochs.  Default 50.
    patience:
        Early-stopping patience.  Default 10.
    learning_rate:
        Initial Adam learning rate.  Default 1e-3.
    seed:
        Random seed for reproducibility.  Default 42.
    save_dir:
        Root directory for saved models.  Default ``"saved_models"``.

    Returns
    -------
    tf.keras.callbacks.History
        Keras history object containing per-epoch metrics.
    """
    set_global_seeds(seed)
    compile_model(model, learning_rate=learning_rate)

    model_save_path = os.path.join(save_dir, model_name)
    os.makedirs(model_save_path, exist_ok=True)
    checkpoint_path = os.path.join(model_save_path, "best_model.keras")

    callbacks = get_callbacks(checkpoint_path=checkpoint_path, patience=patience)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final model (best weights already restored by EarlyStopping)
    final_path = os.path.join(model_save_path, "final_model.keras")
    model.save(final_path)
    print(f"[Trainer] Model saved to: {final_path}")

    return history


def fine_tune_resnet(
    full_model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    unfreeze_from_layer: int = -30,
    fine_tune_lr: float = 1e-5,
    epochs: int = 20,
    patience: int = 5,
    save_dir: str = "saved_models",
) -> tf.keras.callbacks.History:
    """
    Phase 2 fine-tuning for ResNet50: unfreeze upper layers and retrain.

    The upper ``abs(unfreeze_from_layer)`` layers of *base_model* are unfrozen
    and *full_model* is recompiled at *fine_tune_lr* before continuing training.

    Parameters
    ----------
    full_model:
        The complete ResNet50 model (backbone + classification head).
    base_model:
        The ResNet50 backbone returned by ``build_resnet_model()``.
    train_ds:
        Training dataset.
    val_ds:
        Validation dataset.
    unfreeze_from_layer:
        Negative index; layers from this position onward in ``base_model``
        are unfrozen.  Default -30.
    fine_tune_lr:
        Learning rate for fine-tuning.  Must be much lower than Phase 1 lr
        to avoid catastrophic forgetting.  Default 1e-5.
    epochs:
        Maximum additional epochs for fine-tuning.  Default 20.
    patience:
        Early-stopping patience for fine-tuning.  Default 5.
    save_dir:
        Root directory for saved models.  Default ``"saved_models"``.

    Returns
    -------
    tf.keras.callbacks.History
        History object for the fine-tuning phase.
    """
    # Unfreeze the upper portion of the backbone
    base_model.trainable = True
    for layer in base_model.layers[:unfreeze_from_layer]:
        layer.trainable = False

    # Recompile at a lower learning rate
    compile_model(full_model, learning_rate=fine_tune_lr)

    model_save_path = os.path.join(save_dir, "resnet")
    os.makedirs(model_save_path, exist_ok=True)
    checkpoint_path = os.path.join(model_save_path, "best_model_finetuned.keras")

    callbacks = get_callbacks(checkpoint_path=checkpoint_path, patience=patience)

    history = full_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(model_save_path, "final_model.keras")
    full_model.save(final_path)
    print(f"[Trainer] Fine-tuned ResNet50 saved to: {final_path}")

    return history
