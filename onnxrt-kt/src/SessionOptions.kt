package com.jamshedalamqaderi.kmp.onnxrtkt

public enum class GraphOptimizationLevel { NONE, BASIC, EXTENDED, ALL }

public enum class LoggingLevel { VERBOSE, INFO, WARNING, ERROR, FATAL }

/** Hardware accelerators that can be requested for a session. CPU is always available as a fallback. */
public enum class ExecutionProvider {
    /** Android Neural Networks API. Android only. */
    NNAPI,

    /** XNNPACK accelerated CPU kernels (Android, iOS). */
    XNNPACK,

    /** Apple Core ML (iOS, iPadOS, macOS). */
    CORE_ML,

    /** WebGPU. Web only (onnxruntime-web). */
    WEBGPU,
}

/**
 * Options used when creating an [InferenceSession].
 *
 * @property intraOpNumThreads threads used to parallelize a single operator; `0` lets ONNX Runtime decide.
 * @property interOpNumThreads threads used to run independent operators in parallel; `0` lets ONNX Runtime decide.
 *   Ignored on iOS, where the Objective-C API does not expose this setting.
 * @property executionProviders accelerators to try, in priority order. Requesting one that is unavailable
 *   on the current platform makes session creation fail with [OrtException].
 */
public data class SessionOptions(
    val intraOpNumThreads: Int = 0,
    val interOpNumThreads: Int = 0,
    val optimizationLevel: GraphOptimizationLevel = GraphOptimizationLevel.ALL,
    val logSeverityLevel: LoggingLevel = LoggingLevel.WARNING,
    val executionProviders: List<ExecutionProvider> = emptyList(),
) {
    init {
        require(intraOpNumThreads >= 0) { "intraOpNumThreads must be >= 0" }
        require(interOpNumThreads >= 0) { "interOpNumThreads must be >= 0" }
    }
}
