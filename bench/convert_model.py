# TF.jsグラフモデル (../models) を LiteRT用の .tflite に変換するスクリプト。
# 実行方法は README.md を参照 (TF 2.15 + tfjs-graph-converter の venv が必要)。
import sys

import numpy as np
import tensorflow as tf
import tfjs_graph_converter.api as tfjs_api
from tfjs_graph_converter.common import CompatMode

TFJS_MODEL_DIR = "../models"
SAVED_MODEL_DIR = "/tmp/vr_saved_model"
OUTPUT = "vr_model.tflite"

# 1. TF.jsグラフモデル → SavedModel
#    CompatMode.TFLITE で _FusedConv2D などのTF.js特有の融合オペを展開する
tfjs_api.graph_model_to_saved_model(
    TFJS_MODEL_DIR, SAVED_MODEL_DIR, ["serve"], compat_mode=CompatMode.TFLITE
)
print("SavedModel done")

# 2. SavedModel → TFLite
#    LiteRTのWebGPUデリゲートは静的形状のみ対応のため、バッチ次元を1に固定する
sm = tf.saved_model.load(SAVED_MODEL_DIR)
f = sm.signatures["serving_default"]
input_name = list(f.structured_input_signature[1].keys())[0]


@tf.function(input_signature=[tf.TensorSpec([1, 4, 3072, 32], tf.float32)])
def fixed(x):
    return f(**{input_name: x})


converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [fixed.get_concrete_function()], sm
)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
with open(OUTPUT, "wb") as fp:
    fp.write(tflite_model)
print(f"tflite done: {OUTPUT} ({len(tflite_model) / 1e6:.1f} MB)")

# 3. 出力一致確認 (SavedModel vs TFLite)
np.random.seed(0)
x = np.random.randn(1, 4, 3072, 32).astype(np.float32) * 0.1
ref_out = list(f(**{input_name: tf.constant(x)}).values())[0].numpy()

interp = tf.lite.Interpreter(model_path=OUTPUT)
interp.allocate_tensors()
interp.set_tensor(interp.get_input_details()[0]["index"], x)
interp.invoke()
lite_out = interp.get_tensor(interp.get_output_details()[0]["index"])

diff = np.abs(ref_out - lite_out).max()
print(f"max abs diff vs SavedModel: {diff}")
sys.exit(0 if diff < 1e-3 else 1)
