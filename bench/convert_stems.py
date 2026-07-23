# kuielab_b per-stem MDXモデル (ONNX) を LiteRT用のTFLiteに変換するスクリプト。
#
# 実行環境の準備:
#   uv venv --python 3.12 /tmp/onnx2tf_env
#   uv pip install --python /tmp/onnx2tf_env/bin/python \
#       onnx2tf tensorflow onnx onnxruntime sng4onnx onnx_graphsurgeon psutil "flatbuffers>=23"
#   for s in vocals drums bass other; do
#     curl -sL -o /tmp/stems_onnx/kuielab_b_$s.onnx \
#       "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/kuielab_b_$s.onnx"
#   done
# 実行:
#   cd bench && /tmp/onnx2tf_env/bin/python convert_stems.py
#
# 注意: onnx2tf はレイアウトを NCHW → NHWC に変換するため、
# 生成される tflite の入出力は [1, 2048, 32, 4] (freq, time, ch) になる。
# JS側では [1,4,2048,32] を transpose([0,2,3,1]) してから渡すこと。
import os
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
import tensorflow as tf

STEMS = ["vocals", "drums", "bass", "other"]
DIM_T = 32          # リアルタイム用に時間フレームを256→32に短縮
SRC_DIR = "/tmp/stems_onnx"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stems")

os.makedirs(OUT_DIR, exist_ok=True)
failures = []

for stem in STEMS:
    src = f"{SRC_DIR}/kuielab_b_{stem}.onnx"
    patched = f"{SRC_DIR}/{stem}_t32.onnx"
    tf_dir = f"{SRC_DIR}/{stem}_t32_tf"
    dst = f"{OUT_DIR}/kuielab_b_{stem}.tflite"

    print(f"=== {stem} ===")

    # 1. 時間次元を256→32、バッチを1に固定したONNXを作る
    m = onnx.load(src)
    for t in list(m.graph.input) + list(m.graph.output):
        dims = t.type.tensor_type.shape.dim
        dims[0].ClearField("dim_param")
        dims[0].dim_value = 1
        dims[3].dim_value = DIM_T
    del m.graph.value_info[:]  # 古いshape推論結果との矛盾を防ぐ
    onnx.save(m, patched)

    # 2. onnx2tf で TFLite化 (float32)
    subprocess.run(
        ["onnx2tf", "-i", patched, "-o", tf_dir, "-b", "1", "--non_verbose"],
        check=True,
        env={**os.environ, "PATH": os.path.dirname(sys.executable) + ":" + os.environ["PATH"]},
        capture_output=True,
    )

    # 3. パリティ検証 (ONNX Runtime vs TFLite、NHWC変換を考慮)
    np.random.seed(0)
    x = (np.random.randn(1, 4, 2048, DIM_T) * 0.1).astype(np.float32)
    ref = ort.InferenceSession(patched, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]

    interp = tf.lite.Interpreter(model_path=f"{tf_dir}/{stem}_t32_float32.tflite")
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    nhwc = list(inp["shape"]) == [1, 2048, DIM_T, 4]
    interp.set_tensor(inp["index"], np.transpose(x, (0, 2, 3, 1)) if nhwc else x)
    interp.invoke()
    lite = interp.get_tensor(out["index"])
    if list(out["shape"]) != [1, 4, 2048, DIM_T]:
        lite = np.transpose(lite, (0, 3, 1, 2))
    diff = float(np.abs(ref - lite).max())
    print(f"  layout: {'NHWC' if nhwc else 'NCHW'} / max abs diff: {diff:.2e}")

    if diff < 1e-3:
        os.replace(f"{tf_dir}/{stem}_t32_float32.tflite", dst)
        print(f"  -> {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
    else:
        failures.append(stem)
        print("  -> PARITY FAILED")

print()
if failures:
    print("FAILED:", failures)
    sys.exit(1)
print("all stems converted OK")
