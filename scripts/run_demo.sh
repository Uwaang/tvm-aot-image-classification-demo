#!/usr/bin/env bash
set -euo pipefail

rm -rf artifacts/mlf artifacts/tiny_classifier.tar artifacts/tiny_classifier.onnx
mkdir -p artifacts

python scripts/make_tiny_onnx.py
python scripts/build_aot.py
python scripts/inspect_artifact.py

echo
echo "done: ONNX -> TVM AOT + CRT -> generated C/MLF"
