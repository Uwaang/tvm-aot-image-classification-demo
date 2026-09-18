#!/usr/bin/env python3
from pathlib import Path
import shutil
import tarfile

import onnx
import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime


ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "tiny_classifier.onnx"
MLF_TAR = ARTIFACTS / "tiny_classifier.tar"
MLF_DIR = ARTIFACTS / "mlf"


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(f"missing model: {MODEL_PATH}")

    print(f"[tvm] version: {tvm.__version__}")
    print(f"[tvm] loading: {MODEL_PATH}")

    onnx_model = onnx.load(MODEL_PATH)
    shape_dict = {"input": (1, 3, 32, 32)}

    # ONNX is consumed on the build machine. The final target receives the
    # AOT-generated artifact rather than an ONNX parser/runtime.
    mod, params = relay.frontend.from_onnx(
        onnx_model,
        shape=shape_dict,
        dtype="float32",
        freeze_params=True,
    )

    target = tvm.target.Target("c -keys=cpu")
    runtime = Runtime("crt", {"system-lib": True})
    executor = Executor(
        "aot",
        {
            "interface-api": "c",
            "unpacked-api": True,
            "workspace-byte-alignment": 8,
            "link-params": True,
        },
    )

    pass_config = {
        "tir.disable_vectorize": True,
        "tir.usmp.enable": True,
        "tir.usmp.algorithm": "hill_climb",
    }

    print("[tvm] compiling with AOT executor + CRT -> C target")
    with tvm.transform.PassContext(
        opt_level=3,
        config=pass_config,
        disabled_pass=["AlterOpLayout"],
    ):
        factory = relay.build(
            mod,
            target=target,
            params=params,
            runtime=runtime,
            executor=executor,
            mod_name="tiny_classifier",
        )

    MLF_TAR.unlink(missing_ok=True)
    if MLF_DIR.exists():
        shutil.rmtree(MLF_DIR)

    tvm.micro.export_model_library_format(factory, str(MLF_TAR))

    MLF_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(MLF_TAR, "r:*") as tf:
        tf.extractall(MLF_DIR)

    print(f"[tvm] wrote: {MLF_TAR}")
    print(f"[tvm] extracted: {MLF_DIR}")


if __name__ == "__main__":
    main()
