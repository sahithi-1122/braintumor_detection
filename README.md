# Brain Tumor Detection and Classification Using Deep Learning

A deep learning mini-project that trains and compares three architectures — ANN/MLP, Custom CNN, and ResNet50 Transfer Learning — on the Brain Tumor MRI Dataset to classify MRI scans into four categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**.

> **Educational use only.** This project is not validated for clinical diagnosis.

---

## Project Structure

```
.
├── config.py                   # Global constants (paths, hyperparameters)
├── requirements.txt            # Pinned Python dependencies
├── pytest.ini                  # Test configuration
├── README.md
│
├── src/                        # Reusable source modules
│   ├── data_loader.py          # Dataset discovery and class distribution
│   ├── splitter.py             # Stratified train/validation split helpers
│   ├── preprocessor.py         # Image loading, normalization, augmentation
│   ├── trainer.py              # Training loop, callbacks, fine-tuning
│   ├── evaluator.py            # Metrics, confusion matrix, comparison table
│   └── models/
│       ├── ann_model.py        # Feedforward ANN/MLP
│       ├── cnn_model.py        # Custom CNN
│       └── resnet_model.py     # ResNet50 Transfer Learning
│
├── scripts/
│   ├── train_all.py            # Train all three models sequentially
│   ├── evaluate_all.py         # Evaluate saved models and generate results
│   └── predict.py              # CLI: predict class of a single MRI image
│
├── tests/
│   ├── conftest.py             # Shared fixtures
│   ├── test_properties.py      # 10 Hypothesis property-based tests
│   └── test_splitter.py        # Unit / example tests
│
├── data/                       # Place Kaggle Training/ and Testing/ here
├── saved_models/               # Trained models saved here (ann/, cnn/, resnet/)
├── results/                    # Evaluation artifacts (ann/, cnn/, resnet/)
├── reports/
│   ├── brain_tumor_report.md   # IEEE-format academic report
│   └── presentation_outline.md # 12-slide presentation outline
└── notebooks/                  # Exploratory notebooks
```

---

## Dataset Setup

1. Download and extract the dataset from Kaggle:
   https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

2. Extract and place it so the directory layout matches:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── no_tumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── no_tumor/
    └── pituitary/
```

Each of `Training/` and `Testing/` must contain the four class folders `glioma`, `meningioma`, `pituitary`, and `notumor` (the loader also accepts `no_tumor` or `no-tumor`). The program maps those no-tumor folder spellings to the canonical output label `no_tumor`. Keep the official `Testing/` partition untouched: training creates a stratified 85/15 train/validation split from `Training/`, and final evaluation uses `Testing/`.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ (the dependencies use current TensorFlow/Keras APIs). CPU training is supported but slower. The first default ResNet50 training run downloads ImageNet weights; use an internet-enabled environment or provide a locally cached Keras weight file.

---

## Usage

### 1. Train all models

```bash
python scripts/train_all.py --data-dir data/Training
```

Optional flags:
```bash
python scripts/train_all.py \
    --data-dir data/Training \
    --seed 42 \
    --epochs 50 \
    --batch-size 32 \
    --patience 10 \
    --skip-finetune        # skip ResNet50 Phase 2 fine-tuning
```

Trained models are saved to `saved_models/ann/`, `saved_models/cnn/`, `saved_models/resnet/`.

### 2. Evaluate all models

```bash
python scripts/evaluate_all.py --data-dir data/Testing
```

Generates:
- `results/{model}/confusion_matrix.png`
- `results/{model}/metrics.json`
- `results/{model}/training_curves.png`
- `results/comparison_table.csv`

### 3. Predict on a single MRI image

```bash
python scripts/predict.py --image path/to/mri.jpg --model cnn
python scripts/predict.py --image path/to/mri.jpg --model resnet
python scripts/predict.py --image path/to/mri.jpg --model ann
```

Output example:
```
==================================================
  Model         : CNN
  Image         : path/to/mri.jpg
  Predicted Class: Pituitary
==================================================
  Class Probabilities:
    glioma               0.0211  ######
    meningioma           0.0134  ####
    no_tumor             0.0058  #
    pituitary            0.9597  ############################
==================================================
```

---

## Running Tests

```bash
pytest
```

The test suite includes:
- **10 property-based tests** (Hypothesis, `max_examples=20`) verifying model output shape, softmax sum, class index range, no data leakage, preprocessing consistency, augmentation label preservation, split proportions, pixel normalization, split reproducibility, and metrics completeness.
- **Unit tests** covering architecture properties, error handling, trainer configuration, and CLI exit codes.

---

## Running on Google Colab / Kaggle

For full training (recommended for GPU access):

1. Upload the project files or clone from your repository.
2. Mount the dataset or use the Kaggle API:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
   ```
3. Update `DATA_DIR` in `config.py` to point to the downloaded path.
4. Run `scripts/train_all.py`.

---

## Academic Report

The full IEEE-format academic report is at `reports/brain_tumor_report.md`.
The report's result tables now contain the measured values from `results/comparison_table.csv` and `results/*/metrics.json`. If you rerun training, regenerate evaluation artifacts and update the report to match.

### Current experiment results

The checked-in evaluation artifacts are from the 1,600-image supplied Testing partition:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| ANN/MLP | 59.44% | 54.22% |
| Custom CNN | 70.75% | 68.08% |
| ResNet50 transfer learning | 93.69% | 93.53% |

See `reports/brain_tumor_report.md` and `results/` for per-class metrics, confusion matrices, and training curves. These are dataset-specific educational results, not clinical validation.

### GitHub notes

The `.gitignore` excludes the downloaded dataset and trained model files because they are large and should be obtained separately. The metrics, plots, source, and report remain available to commit. The final ResNet model is over 200 MB, so do not remove this ignore rule for a normal GitHub push; if model weights must be shared, use Git LFS or a release/artifact store and confirm the dataset and weight distribution terms first.

---

## License

This project is for educational purposes only. The dataset is subject to its original Kaggle license.
