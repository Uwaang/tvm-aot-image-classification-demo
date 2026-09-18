# TVM AOT tiny image-classification demo

A small, self-contained example of the path we were discussing:

```
Tiny ONNX CNN -> TVM Relay -> AOT executor + CRT -> Model Library Format (generated C)
```

The model is intentionally tiny: 32x32 RGB input, two small convolutions, global average pooling, and a 10-class output. The weights are deterministic random values, so this repo is for **deployment/AOT structure**, not accuracy benchmarking.

## Run

```bash
docker compose up --build
```

Generated files are written to `./artifacts`.

The demo creates:

- `artifacts/tiny_classifier.onnx`
- `artifacts/tiny_classifier.tar` — TVM Model Library Format
- `artifacts/mlf/` — extracted generated C/metadata

The build uses TVM's **AOT executor** with the **C runtime (CRT)** and embeds model parameters into the generated artifact.

## Why this repo

This is meant to sit just above an `onnx2c`-style deployment in abstraction:

```
direct C -> onnx2c -> TVM AOT + CRT -> tiny runtimes -> ncnn -> ORT minimal -> full ORT
```

TVM does more graph lowering, memory planning, and target-oriented compilation than a simple ONNX-to-C converter, while still allowing a small generated deployment artifact.

## Notes

- Host build environment: Docker / Python 3.11
- TVM: `apache-tvm==0.26.0`
- Target in this demo: portable C
- This does not require TVM to parse ONNX on the final target device; ONNX is consumed at build time.
