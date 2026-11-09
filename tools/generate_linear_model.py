#!/usr/bin/env python3
"""Generates `linear_model.onnx`: y = x * 2 + 1 (float32, input shape [N, 1]).

Dependency-free: it hand-encodes the ONNX protobuf so no `onnx` pip package is needed.
Usage: python3 tools/generate_linear_model.py <output.onnx> [<output2.onnx> ...]
"""
import struct
import sys


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def key(field: int, wire: int) -> bytes:
    return varint((field << 3) | wire)


def f_varint(field: int, value: int) -> bytes:
    return key(field, 0) + varint(value)


def f_bytes(field: int, value: bytes) -> bytes:
    return key(field, 2) + varint(len(value)) + value


def f_str(field: int, value: str) -> bytes:
    return f_bytes(field, value.encode())


FLOAT = 1


def scalar_initializer(name: str, value: float) -> bytes:
    # TensorProto: data_type=2, name=8, raw_data=9 (scalar => no dims)
    return f_varint(2, FLOAT) + f_str(8, name) + f_bytes(9, struct.pack("<f", value))


def value_info(name: str) -> bytes:
    dim_n = f_str(2, "N")  # Dimension.dim_param
    dim_1 = f_varint(1, 1)  # Dimension.dim_value
    shape = f_bytes(1, dim_n) + f_bytes(1, dim_1)  # TensorShapeProto.dim
    tensor_type = f_varint(1, FLOAT) + f_bytes(2, shape)  # TypeProto.Tensor
    type_proto = f_bytes(1, tensor_type)  # TypeProto.tensor_type
    return f_str(1, name) + f_bytes(2, type_proto)


def node(op: str, inputs, outputs, name: str) -> bytes:
    data = b"".join(f_str(1, i) for i in inputs) + b"".join(f_str(2, o) for o in outputs)
    return data + f_str(3, name) + f_str(4, op)


graph = (
    f_bytes(1, node("Mul", ["x", "w"], ["xw"], "mul"))
    + f_bytes(1, node("Add", ["xw", "b"], ["y"], "add"))
    + f_str(2, "linear")
    + f_bytes(5, scalar_initializer("w", 2.0))
    + f_bytes(5, scalar_initializer("b", 1.0))
    + f_bytes(11, value_info("x"))
    + f_bytes(12, value_info("y"))
)
opset = f_str(1, "") + f_varint(2, 13)
model = f_varint(1, 8) + f_str(2, "onnxrt-kt") + f_bytes(7, graph) + f_bytes(8, opset)

for path in sys.argv[1:] or ["linear_model.onnx"]:
    with open(path, "wb") as fh:
        fh.write(model)
    print(f"wrote {path} ({len(model)} bytes)")
