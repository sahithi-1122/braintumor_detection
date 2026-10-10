"""Quick smoke test for the three model builders."""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

# ── ANN ──────────────────────────────────────────────────────────────────────
from src.models.ann_model import build_ann_model

ann = build_ann_model()
assert ann.input_shape == (None, 64, 64, 3), f"ANN input shape wrong: {ann.input_shape}"
batch_ann = np.random.rand(4, 64, 64, 3).astype(np.float32)
out_ann = ann.predict(batch_ann, verbose=0)
assert out_ann.shape == (4, 4), f"ANN output shape wrong: {out_ann.shape}"
assert all(abs(row.sum() - 1.0) < 1e-5 for row in out_ann), "ANN softmax sum != 1"
print("ANN    OK  input=(None,64,64,3)  output=(4,4)  softmax_sum~1.0")

# ── CNN ──────────────────────────────────────────────────────────────────────
from src.models.cnn_model import build_cnn_model

cnn = build_cnn_model()
assert cnn.input_shape == (None, 128, 128, 3), f"CNN input shape wrong: {cnn.input_shape}"
batch_cnn = np.random.rand(3, 128, 128, 3).astype(np.float32)
out_cnn = cnn.predict(batch_cnn, verbose=0)
assert out_cnn.shape == (3, 4), f"CNN output shape wrong: {out_cnn.shape}"
assert all(abs(row.sum() - 1.0) < 1e-5 for row in out_cnn), "CNN softmax sum != 1"
print("CNN    OK  input=(None,128,128,3)  output=(3,4)  softmax_sum~1.0")

# ── ResNet ────────────────────────────────────────────────────────────────────
from src.models.resnet_model import build_resnet_model

full_model, base_model = build_resnet_model()
assert full_model.input_shape == (None, 224, 224, 3), f"ResNet input shape wrong: {full_model.input_shape}"
assert not base_model.trainable, "base_model.trainable should be False"
batch_res = np.random.rand(2, 224, 224, 3).astype(np.float32)
out_res = full_model.predict(batch_res, verbose=0)
assert out_res.shape == (2, 4), f"ResNet output shape wrong: {out_res.shape}"
assert all(abs(row.sum() - 1.0) < 1e-5 for row in out_res), "ResNet softmax sum != 1"
print("ResNet OK  input=(None,224,224,3)  output=(2,4)  softmax_sum~1.0  backbone_frozen=True")

print("\nAll three models verified successfully.")
