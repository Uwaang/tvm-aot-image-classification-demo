#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "tiny_classifier.onnx"


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # Tiny 10-class image classifier:
    # 1x3x32x32 -> Conv(8) -> ReLU -> MaxPool
    # -> Conv(16) -> ReLU -> GlobalAveragePool -> Flatten -> Gemm(10)
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, 32, 32]
    )
    output_info = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, 10]
    )

    w1 = (rng.standard_normal((8, 3, 3, 3)) * 0.05).astype(np.float32)
    b1 = np.zeros((8,), dtype=np.float32)
    w2 = (rng.standard_normal((16, 8, 3, 3)) * 0.05).astype(np.float32)
    b2 = np.zeros((16,), dtype=np.float32)
    w3 = (rng.standard_normal((16, 10)) * 0.05).astype(np.float32)
    b3 = np.zeros((10,), dtype=np.float32)

    initializers = [
        numpy_helper.from_array(w1, "conv1_w"),
        numpy_helper.from_array(b1, "conv1_b"),
        numpy_helper.from_array(w2, "conv2_w"),
        numpy_helper.from_array(b2, "conv2_b"),
        numpy_helper.from_array(w3, "fc_w"),
        numpy_helper.from_array(b3, "fc_b"),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "conv1_w", "conv1_b"],
            ["x1"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["x1"], ["x2"]),
        helper.make_node(
            "MaxPool",
            ["x2"],
            ["x3"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
        helper.make_node(
            "Conv",
            ["x3", "conv2_w", "conv2_b"],
            ["x4"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Relu", ["x4"], ["x5"]),
        helper.make_node("GlobalAveragePool", ["x5"], ["x6"]),
        helper.make_node("Flatten", ["x6"], ["x7"], axis=1),
        helper.make_node(
            "Gemm",
            ["x7", "fc_w", "fc_b"],
            ["logits"],
            alpha=1.0,
            beta=1.0,
            transB=0,
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny_classifier",
        [input_info],
        [output_info],
        initializer=initializers,
    )

    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="tvm-aot-image-classification-demo",
    )
    model.ir_version = min(model.ir_version, 10)
    onnx.checker.check_model(model)
    onnx.save(model, MODEL_PATH)

    param_count = sum(x.size for x in [w1, b1, w2, b2, w3, b3])
    print(f"[onnx] wrote {MODEL_PATH}")
    print(f"[onnx] parameters: {param_count:,} ({param_count * 4 / 1024:.1f} KiB fp32)")


if __name__ == "__main__":
    main()
