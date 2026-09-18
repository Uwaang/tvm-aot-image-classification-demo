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
- TVM: `apache-tvm==0.14.dev264` for the classic Relay + AOT executor + CRT path used here. TVM 0.26 has moved to Relax and does not expose `tvm.relay`.
- Target in this demo: portable C
- This does not require TVM to parse ONNX on the final target device; ONNX is consumed at build time.

CI smoke-test: `.github/workflows/smoke.yml` builds the Docker image and checks that TVM emits C/MLF artifacts.


### TVM wheel note

The classic Relay/AOT demo is pinned to `apache-tvm==0.14.dev264`. PyPI still hosts its CPython 3.11 manylinux wheel, but the current simple index used by pip no longer advertises that old build. The Docker build therefore resolves the exact archived wheel from PyPI's JSON metadata and installs that wheel URL directly.
