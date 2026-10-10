"""
evaluate_all.py — Evaluation entry point.

Loads each saved model, evaluates on the held-out test set, generates
confusion matrices, and builds the final comparison table.

Usage
-----
    python scripts/evaluate_all.py --data-dir data/Training
    python scripts/evaluate_all.py --data-dir data/Training --seed 42
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

import config
from src.data_loader import discover_dataset
from src.evaluator import build_comparison_table, evaluate_model
from src.preprocessor import build_test_dataset
from src.splitter import stratified_train_val_split


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate all saved brain tumor models.")
    parser.add_argument("--data-dir", type=str, default=os.path.join(config.DATA_DIR, "Testing"),
                        help="Path to the official Kaggle Testing partition")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--save-dir", type=str, default=config.SAVE_DIR)
    parser.add_argument("--results-dir", type=str, default=config.RESULTS_DIR)
    return parser.parse_args()


def load_model(model_name: str, save_dir: str) -> tf.keras.Model:
    """Load the final saved model for *model_name*."""
    model_path = os.path.join(save_dir, model_name, "final_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. "
            f"Run scripts/train_all.py first."
        )
    print(f"[Evaluate] Loading model from: {model_path}")
    return tf.keras.models.load_model(model_path)


def main():
    args = parse_args()

    # Evaluate only on the dataset's official, held-out Testing partition.
    samples = discover_dataset(args.data_dir)
    test_samples = samples

    all_metrics = {}
    model_names = ["ann", "cnn", "resnet"]
    for model_name in model_names:
        model = load_model(model_name, args.save_dir)

        test_ds = build_test_dataset(test_samples, model_name, args.batch_size)
        metrics = evaluate_model(
            model, test_ds, model_name,
            results_dir=args.results_dir,
        )
        all_metrics[model_name] = metrics

    build_comparison_table(all_metrics, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
