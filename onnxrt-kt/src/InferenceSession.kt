package com.jamshedalamqaderi.kmp.onnxrtkt

/**
 * A loaded ONNX model ready to run inference. Create it with [OnnxRuntime.loadSession]
 * and [close] it when done to release native memory.
 */
public interface InferenceSession : AutoCloseable {
    /** Names of the model inputs, in model order. */
    public val inputNames: List<String>

    /** Names of the model outputs, in model order. */
    public val outputNames: List<String>

    /**
     * Runs the model without blocking the calling thread. Works on every platform, including the web.
     * On other platforms, the synchronous `run` extension does the same on the calling thread.
     *
     * @param inputs a tensor for every required model input, keyed by input name.
     * @param outputNames which outputs to compute; defaults to all of them.
     * @return the requested outputs, keyed by output name.
     * @throws OrtException if ONNX Runtime fails or an output is not a tensor.
     */
    public suspend fun infer(inputs: Map<String, Tensor>, outputNames: Collection<String> = this.outputNames): Map<String, Tensor>
}

/** Runs a model with a single input and returns its first output. */
public suspend fun InferenceSession.inferSingle(input: Tensor): Tensor {
    val (inputName, outputName) = singleInputAndFirstOutput()
    return infer(mapOf(inputName to input), listOf(outputName)).getValue(outputName)
}

internal fun InferenceSession.singleInputAndFirstOutput(): Pair<String, String> {
    val inputName = inputNames.singleOrNull()
        ?: throw OrtException("A single-input call requires a model with exactly one input, found $inputNames")
    val outputName = outputNames.firstOrNull() ?: throw OrtException("The model has no outputs")
    return inputName to outputName
}
