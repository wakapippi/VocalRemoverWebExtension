# バックエンド性能比較: TF.js (WebGPU) vs LiteRT.js

推論バックエンドを [LiteRT.js](https://developers.google.com/edge/litert/web) に置き換えた場合の性能を検証するベンチマーク。

## 結果 (2026-07-20, Apple Silicon Mac / Chrome headless)

同一モデル (UVR-MDX-NET-Inst_HQ_3, dim_t=32)・同一入力で、
「CPU上の入力データ → 推論 → CPUへの出力読み出し」の1往復を計測 (warmup 10回 + 計測30回)。

| バックエンド | batch | median | mean | p90 |
|---|---|---|---|---|
| TF.js WebGPU | 1 | 91.9 ms | 92.1 ms | 94.4 ms |
| TF.js WebGPU | 2 (denoise相当) | 176.1 ms | 186.6 ms | 178.5 ms |
| **LiteRT.js WebGPU** | 1 | **62.4 ms** | **62.7 ms** | 64.6 ms |
| LiteRT.js Wasm (CPU/XNNPACK) | 1 | 2001 ms | 2028 ms | 2116 ms |

- **LiteRT.js WebGPU は TF.js WebGPU の約1.47倍高速** (batch=1: 91.9ms → 62.4ms、約32%削減)
- 出力の数値一致も確認済み (max abs diff ≈ 1.4e-4)
- Wasm(CPU)はリアルタイム処理には非現実的 (約2秒/セグメント)
- 注意: LiteRT用の .tflite はWebGPUデリゲートの制約でバッチ次元を1に固定している。
  denoise (batch=2) を使う場合は2回実行 (約125ms) になるが、それでもTF.jsのbatch=2 (176ms) より速い。
  batch=2固定の .tflite を別途変換する手もある。

## 再現手順

### 1. モデル変換 (TF.jsグラフモデル → .tflite)

TF 2.15 が必要 (Python 3.11)。tensorflowjs / tfjs-graph-converter は古いパッケージのため、
新しいnumpyとの互換パッチが必要:

```bash
uv venv --python 3.11 /tmp/tfconv
uv pip install --python /tmp/tfconv/bin/python \
    "tensorflow==2.15.1" tfjs-graph-converter "protobuf==3.20.3" \
    "tensorflow-hub==0.16.1" "setuptools<81"

# np.object / np.bool 削除への互換パッチ
/tmp/tfconv/bin/python - <<'EOF'
import re, pathlib
for pkg in ['tfjs_graph_converter', 'tensorflowjs']:
    base = pathlib.Path(f'/tmp/tfconv/lib/python3.11/site-packages/{pkg}')
    for f in base.rglob('*.py'):
        s = f.read_text()
        s2 = re.sub(r'np\.object\b(?!_)', 'object', s)
        s2 = re.sub(r'np\.bool\b(?!_)', 'bool', s2)
        if s2 != s: f.write_text(s2)
EOF

cd bench
/tmp/tfconv/bin/python convert_model.py   # vr_model.tflite が生成される
```

### 2. ベンチマーク実行

```bash
cd bench
npm install
node run_bench.mjs        # ヘッドレスChromeで自動実行
```

ブラウザで見たい場合は、リポジトリルートで `python3 -m http.server` などを立てて
`http://localhost:8000/bench/bench.html` を開く (WebGPUが必要)。

## 本実装への組み込みについて

拡張機能本体に組み込む場合の主な作業:
- `@litertjs/core` (+wasmファイル) を拡張機能に同梱し、sandbox.jsで `loadAndCompile` に置き換える
- STFT/ISTFTはTF.jsのまま使い、`@litertjs/tfjs-interop` の `runWithTfjsTensors` でGPU上のまま受け渡す
  (今回のベンチはCPU往復込みなので、interopでGPU直結にすればさらに差が開く可能性がある)
- denoise用に batch=2 固定の .tflite も変換して使い分ける (または2回実行)
- モデル配布サイズ: .tflite 66.8MB (現行のTF.jsシャード合計とほぼ同等)
