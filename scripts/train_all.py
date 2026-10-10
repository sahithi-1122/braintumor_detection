"""
train_all.py — Sequential training entry point.

Trains ANN, CNN, and ResNet50 (Phase 1 + Phase 2 fine-tuning) on the
brain tumor MRI dataset.

Usage
-----
    python scripts/train_all.py --data-dir data/Training
    python scripts/train_all.py --data-dir data/Training --seed 0 --epochs 30
"""

import argparse
import os
import sys

# Ensure the workspace root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.data_loader import discover_dataset, print_dataset_info
from src.evaluator import plot_training_curves
from src.models.ann_model import build_ann_model
from src.models.cnn_model import build_cnn_model
from src.models.resnet_model import build_resnet_model
from src.preprocessor import build_test_dataset, build_train_dataset, build_val_dataset
from src.splitter import assert_no_leakage, stratified_train_val_split
from src.trainer import fine_tune_resnet, set_global_seeds, train


def parse_args():
    parser = argparse.ArgumentParser(description="Train all three brain tumor models.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(config.DATA_DIR, "Training"),
        help="Path to the Kaggle Training partition (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.RANDOM_SEED,
        help="Random seed for reproducibility (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Maximum training epochs per model (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=config.PATIENCE,
        help="Early stopping patience (default: %(default)s)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=config.SAVE_DIR,
        help="Directory to save trained models (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=config.RESULTS_DIR,
        help="Directory to save training curve plots (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-finetune",
        action="store_true",
        help="Skip ResNet50 Phase 2 fine-tuning",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seeds(args.seed)

    # ------------------------------------------------------------------
    # 1. Load and inspect dataset
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 1: Loading dataset")
    print(f"{'='*60}")
    print_dataset_info(args.data_dir)
    samples = discover_dataset(args.data_dir)

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 2: Splitting Training partition (85/15 train/validation)")
    print(f"{'='*60}")
    train_samples, val_samples = stratified_train_val_split(
        samples, val_ratio=0.15, random_seed=args.seed
    )
    assert_no_leakage(train_samples, val_samples, [])
    print("[Train] Training and validation paths are disjoint. The Kaggle Testing partition is reserved for final evaluation.")

    # ------------------------------------------------------------------
    # 3. Train ANN
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 3: Training ANN/MLP baseline")
    print(f"{'='*60}")
    ann_train_ds = build_train_dataset(train_samples, "ann", args.batch_size)
    ann_val_ds = build_val_dataset(val_samples, "ann", args.batch_size)
    ann_model = build_ann_model()
    ann_history = train(
        ann_model, ann_train_ds, ann_val_ds,
        model_name="ann", epochs=args.epochs,
        patience=args.patience, seed=args.seed, save_dir=args.save_dir,
    )
    plot_training_curves(ann_history, "ann", args.results_dir)

    # ------------------------------------------------------------------
    # 4. Train CNN
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 4: Training Custom CNN")
    print(f"{'='*60}")
    cnn_train_ds = build_train_dataset(train_samples, "cnn", args.batch_size)
    cnn_val_ds = build_val_dataset(val_samples, "cnn", args.batch_size)
    cnn_model = build_cnn_model()
    cnn_history = train(
        cnn_model, cnn_train_ds, cnn_val_ds,
        model_name="cnn", epochs=args.epochs,
        patience=args.patience, seed=args.seed, save_dir=args.save_dir,
    )
    plot_training_curves(cnn_history, "cnn", args.results_dir)

    # ------------------------------------------------------------------
    # 5. Train ResNet50 — Phase 1 (frozen backbone)
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("STEP 5: Training ResNet50 — Phase 1 (frozen backbone)")
    print(f"{'='*60}")
    resnet_train_ds = build_train_dataset(train_samples, "resnet", args.batch_size)
    resnet_val_ds = build_val_dataset(val_samples, "resnet", args.batch_size)
    resnet_model, resnet_base = build_resnet_model()
    resnet_history = train(
        resnet_model, resnet_train_ds, resnet_val_ds,
        model_name="resnet", epochs=args.epochs,
        patience=args.patience, seed=args.seed, save_dir=args.save_dir,
    )
    plot_training_curves(resnet_history, "resnet", args.results_dir)

    # ------------------------------------------------------------------
    # 6. Train ResNet50 — Phase 2 (fine-tuning)
    # ------------------------------------------------------------------
    if not args.skip_finetune:
        print(f"\n{'='*60}")
        print("STEP 6: ResNet50 — Phase 2 fine-tuning")
        print(f"{'='*60}")
        ft_history = fine_tune_resnet(
            resnet_model, resnet_base,
            resnet_train_ds, resnet_val_ds,
            save_dir=args.save_dir,
        )
        plot_training_curves(ft_history, "resnet_finetuned", args.results_dir)
    else:
        print("\n[Train] Skipping ResNet50 fine-tuning (--skip-finetune set).")

    print(f"\n{'='*60}")
    print("Training complete. Run scripts/evaluate_all.py to evaluate all models.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
