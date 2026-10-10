"""
predict.py — Prediction CLI for Brain Tumor Detection and Classification.

Loads a saved model and runs inference on a single MRI image.

Usage
-----
    python scripts/predict.py --image path/to/mri.jpg --model cnn
    python scripts/predict.py --image path/to/mri.jpg --model resnet

Exit codes
----------
0  Successful prediction
1  Invalid model name or saved model not found
2  Image file not found or unreadable

WARNING: This tool is for educational demonstration only.
         It is NOT intended for clinical diagnosis.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

import config
from src.data_loader import CLASS_NAMES
from src.preprocessor import preprocess_single_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run brain tumor classification on a single MRI image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="WARNING: For educational use only. Not for clinical diagnosis.",
    )
    parser.add_argument("--image", required=True, help="Path to the input MRI image (JPEG or PNG)")
    parser.add_argument(
        "--model",
        required=True,
        choices=["ann", "cnn", "resnet"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=config.SAVE_DIR,
        help="Root directory of saved models (default: %(default)s)",
    )
    return parser.parse_args()


def load_model(model_name: str, save_dir: str) -> tf.keras.Model:
    model_path = os.path.join(save_dir, model_name, "final_model.keras")
    if not os.path.exists(model_path):
        print(
            f"ERROR: Saved model not found at {model_path}. "
            f"Run scripts/train_all.py first."
        )
        sys.exit(1)
    return tf.keras.models.load_model(model_path)


def run_prediction(model_name: str, image_path: str, save_dir: str) -> None:
    # Validate image path
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(2)

    # Load model
    model = load_model(model_name, save_dir)

    # Preprocess image
    try:
        img_array = preprocess_single_image(image_path, model_name)
    except ValueError as e:
        print(f"ERROR: Could not decode image at {image_path}. Ensure it is a valid JPEG or PNG.")
        print(f"Detail: {e}")
        sys.exit(2)

    # Predict
    probs = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]

    # Print results
    print("\n" + "=" * 50)
    print(f"  Model         : {model_name.upper()}")
    print(f"  Image         : {image_path}")
    print(f"  Predicted Class: {predicted_class.replace('_', ' ').title()}")
    print("=" * 50)
    print("  Class Probabilities:")
    for i, cls in enumerate(CLASS_NAMES):
        bar = "#" * int(probs[i] * 30)
        print(f"    {cls:<20} {probs[i]:.4f}  {bar}")
    print("=" * 50)
    print("\n  WARNING: This result is for educational demonstration only.")
    print("           Do NOT use for clinical or medical diagnosis.\n")


def main():
    args = parse_args()
    run_prediction(args.model, args.image, args.save_dir)


if __name__ == "__main__":
    main()
