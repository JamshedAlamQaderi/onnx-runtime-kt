package com.jamshedalamqaderi.kmp.onnxrtkt

/**
 * Entry point of the library.
 *
 * [loadSession] works on every platform, including the web, where ONNX Runtime is asynchronous.
 * On all other platforms you can also create sessions synchronously with `OnnxRuntime.createSession`.
 */
public object OnnxRuntime {
    /** Version of the native ONNX Runtime in use, e.g. `1.31.1`. */
    public val version: String get() = platformVersion()

    /** Loads a session from an in-memory `.onnx` / `.ort` model. */
    public suspend fun loadSession(modelBytes: ByteArray, options: SessionOptions = SessionOptions()): InferenceSession {
        if (modelBytes.isEmpty()) throw OrtException("Model bytes are empty")
        return platformLoadSession(ModelSource.Bytes(modelBytes), options)
    }

    /**
     * Loads a session from a model file. On the web, [modelPath] is a URL that is fetched;
     * elsewhere it is a path on the local file system.
     */
    public suspend fun loadSession(modelPath: String, options: SessionOptions = SessionOptions()): InferenceSession =
        platformLoadSession(ModelSource.File(modelPath), options)
}

internal sealed interface ModelSource {
    class Bytes(val bytes: ByteArray) : ModelSource
    class File(val path: String) : ModelSource
}

internal expect fun platformVersion(): String

internal expect suspend fun platformLoadSession(model: ModelSource, options: SessionOptions): InferenceSession
