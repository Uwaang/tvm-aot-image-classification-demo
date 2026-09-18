#!/usr/bin/env bash
set -euo pipefail

rm -rf artifacts/mlf artifacts/tiny_classifier.tar artifacts/tiny_classifier.onnx
mkdir -p artifacts

python scripts/make_tiny_onnx.py
python scripts/build_aot.py
python scripts/inspect_artifact.py

# Verify the generated files inside the same container that produced them.
test -f artifacts/tiny_classifier.onnx
test -f artifacts/tiny_classifier.tar
test -f artifacts/mlf/metadata.json
find artifacts/mlf -type f -name '*.c' -print -quit | grep .

echo
echo "done: ONNX -> TVM AOT + CRT -> generated C/MLF"
