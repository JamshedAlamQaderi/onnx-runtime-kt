package com.jamshedalamqaderi.kmp.onnxrtkt

/** Element types supported by [Tensor], on every platform. */
public enum class TensorElementType(public val byteSize: Int) {
    FLOAT(4),
    DOUBLE(8),
    INT8(1),
    UINT8(1),
    INT32(4),
    INT64(8),
    BOOL(1),

    /** Variable-size UTF-8 strings. [byteSize] is meaningless for this type. */
    STRING(-1),
}
