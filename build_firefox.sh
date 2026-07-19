#!/bin/bash
# Firefox版のビルド: dist/firefox (展開版) と dist/*-firefox-*.zip を生成する。
# Chrome版と共通のランタイムファイル + Firefox専用のbackground構成 + manifest差し替え。
set -e
cd "$(dirname "$0")"

VERSION=$(python3 -c "import json; print(json.load(open('manifest.firefox.json'))['version'])")
OUT=dist/firefox

rm -rf "$OUT"
mkdir -p "$OUT/models"

cp index.html index.js \
   content_script.js vocal-remover-processor.js \
   sandbox.html sandbox.js \
   tf-csp-shim.js tf.min.js tf-backend-webgpu.js \
   background.html background_firefox.js \
   LICENSE.md \
   "$OUT/"
cp models/vr_model.tflite models/LICENSE.txt "$OUT/models/"
cp -r litert "$OUT/litert"
cp manifest.firefox.json "$OUT/manifest.json"

ZIP="VocalRemoverWebExtension-firefox-$VERSION.zip"
rm -f "dist/$ZIP"
(cd "$OUT" && zip -q -r "../$ZIP" .)

echo "built:"
echo "  $OUT/ (about:debugging から「一時的なアドオンを読み込む」で $OUT/manifest.json を選択)"
echo "  dist/$ZIP"
