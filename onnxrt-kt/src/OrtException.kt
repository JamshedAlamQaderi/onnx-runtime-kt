package com.jamshedalamqaderi.kmp.onnxrtkt

/** Thrown when ONNX Runtime reports an error or an operation is not supported on the current platform. */
public class OrtException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)
