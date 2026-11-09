package com.jamshedalamqaderi.kmp.onnxrtkt

/** Entry point of the library. */
public object OnnxRuntime {
    /** Version of the native ONNX Runtime in use, e.g. `1.31.1`. */
    public val version: String get() = platformVersion()

    /** Creates a session from an in-memory `.onnx` / `.ort` model. */
    public fun createSession(modelBytes: ByteArray, options: SessionOptions = SessionOptions()): InferenceSession {
        if (modelBytes.isEmpty()) throw OrtException("Model bytes are empty")
        return platformCreateSession(ModelSource.Bytes(modelBytes), options)
    }

    /** Creates a session from a model file on the local file system. */
    public fun createSession(modelPath: String, options: SessionOptions = SessionOptions()): InferenceSession =
        platformCreateSession(ModelSource.File(modelPath), options)
}

internal sealed interface ModelSource {
    class Bytes(val bytes: ByteArray) : ModelSource
    class File(val path: String) : ModelSource
}

internal expect fun platformVersion(): String

internal expect fun platformCreateSession(model: ModelSource, options: SessionOptions): InferenceSession
