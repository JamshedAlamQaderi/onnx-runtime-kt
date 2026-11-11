package com.jamshedalamqaderi.kmp.onnxrtkt

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Creates a session from an in-memory `.onnx` / `.ort` model, on the calling thread. Not available on the web. */
public fun OnnxRuntime.createSession(modelBytes: ByteArray, options: SessionOptions = SessionOptions()): InferenceSession {
    if (modelBytes.isEmpty()) throw OrtException("Model bytes are empty")
    return platformCreateSession(ModelSource.Bytes(modelBytes), options)
}

/** Creates a session from a model file on the local file system, on the calling thread. Not available on the web. */
public fun OnnxRuntime.createSession(modelPath: String, options: SessionOptions = SessionOptions()): InferenceSession =
    platformCreateSession(ModelSource.File(modelPath), options)

/**
 * Runs the model on the calling thread. Not available on the web; use [InferenceSession.infer] there.
 *
 * @see InferenceSession.infer
 */
public fun InferenceSession.run(
    inputs: Map<String, Tensor>,
    outputNames: Collection<String> = this.outputNames,
): Map<String, Tensor> = (this as BlockingInferenceSession).runSync(inputs, outputNames)

/** Runs a model with a single input and returns its first output, on the calling thread. */
public fun InferenceSession.runSingle(input: Tensor): Tensor {
    val (inputName, outputName) = singleInputAndFirstOutput()
    return run(mapOf(inputName to input), listOf(outputName)).getValue(outputName)
}

/** Base class of the sessions on platforms where ONNX Runtime is synchronous (everything but the web). */
internal abstract class BlockingInferenceSession : InferenceSession {
    abstract fun runSync(inputs: Map<String, Tensor>, outputNames: Collection<String>): Map<String, Tensor>

    final override suspend fun infer(inputs: Map<String, Tensor>, outputNames: Collection<String>): Map<String, Tensor> =
        withContext(Dispatchers.Default) { runSync(inputs, outputNames) }
}

internal expect fun platformCreateSession(model: ModelSource, options: SessionOptions): InferenceSession

internal actual suspend fun platformLoadSession(model: ModelSource, options: SessionOptions): InferenceSession =
    withContext(Dispatchers.Default) { platformCreateSession(model, options) }
