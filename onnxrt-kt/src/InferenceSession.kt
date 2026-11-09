package com.jamshedalamqaderi.kmp.onnxrtkt

/**
 * A loaded ONNX model ready to run inference. Create it with [OnnxRuntime.createSession]
 * and [close] it when done to release native memory.
 */
public interface InferenceSession : AutoCloseable {
    /** Names of the model inputs, in model order. */
    public val inputNames: List<String>

    /** Names of the model outputs, in model order. */
    public val outputNames: List<String>

    /**
     * Runs the model.
     *
     * @param inputs a tensor for every required model input, keyed by input name.
     * @param outputNames which outputs to compute; defaults to all of them.
     * @return the requested outputs, keyed by output name.
     * @throws OrtException if ONNX Runtime fails or an output is not a tensor.
     */
    public fun run(inputs: Map<String, Tensor>, outputNames: Collection<String> = this.outputNames): Map<String, Tensor>
}

/** Runs a model with a single input and returns its first output. */
public fun InferenceSession.runSingle(input: Tensor): Tensor {
    val inputName = inputNames.singleOrNull()
        ?: throw OrtException("runSingle() requires a model with exactly one input, found $inputNames")
    val outputName = outputNames.first()
    return run(mapOf(inputName to input), listOf(outputName)).getValue(outputName)
}
