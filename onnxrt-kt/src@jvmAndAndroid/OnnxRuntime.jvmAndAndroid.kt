package com.jamshedalamqaderi.kmp.onnxrtkt

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo.OnnxTensorType
import java.nio.ByteBuffer
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import ai.onnxruntime.OrtException as JavaOrtException

private val environment: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

internal actual fun platformVersion(): String = environment.version

internal actual fun platformCreateSession(model: ModelSource, options: SessionOptions): InferenceSession = ortCall {
    val ortOptions = options.toOrt()
    try {
        val session = when (model) {
            is ModelSource.Bytes -> environment.createSession(model.bytes, ortOptions)
            is ModelSource.File -> environment.createSession(model.path, ortOptions)
        }
        JvmInferenceSession(session, ortOptions)
    } catch (t: Throwable) {
        ortOptions.close()
        throw t
    }
}

private class JvmInferenceSession(
    private val session: OrtSession,
    private val options: OrtSession.SessionOptions,
) : InferenceSession {
    override val inputNames: List<String> = session.inputNames.toList()
    override val outputNames: List<String> = session.outputNames.toList()

    override fun run(inputs: Map<String, Tensor>, outputNames: Collection<String>): Map<String, Tensor> = ortCall {
        val created = ArrayList<OnnxTensor>(inputs.size)
        try {
            val ortInputs = inputs.mapValues { (_, tensor) -> tensor.toOnnxTensor().also(created::add) }
            session.run(ortInputs, outputNames.toSet()).use { result ->
                buildMap {
                    for ((name, value) in result) {
                        val tensor = value as? OnnxTensor
                            ?: throw OrtException("Output '$name' is not a tensor (${value.type}); only tensor outputs are supported")
                        put(name, tensor.toTensor())
                    }
                }
            }
        } finally {
            created.forEach { it.close() }
        }
    }

    override fun close() {
        session.close()
        options.close()
    }
}

private inline fun <T> ortCall(block: () -> T): T = try {
    block()
} catch (e: JavaOrtException) {
    throw OrtException(e.message ?: "ONNX Runtime error", e)
}

private fun SessionOptions.toOrt(): OrtSession.SessionOptions {
    val o = OrtSession.SessionOptions()
    try {
        if (intraOpNumThreads > 0) o.setIntraOpNumThreads(intraOpNumThreads)
        if (interOpNumThreads > 0) o.setInterOpNumThreads(interOpNumThreads)
        o.setOptimizationLevel(
            when (optimizationLevel) {
                GraphOptimizationLevel.NONE -> OrtSession.SessionOptions.OptLevel.NO_OPT
                GraphOptimizationLevel.BASIC -> OrtSession.SessionOptions.OptLevel.BASIC_OPT
                GraphOptimizationLevel.EXTENDED -> OrtSession.SessionOptions.OptLevel.EXTENDED_OPT
                GraphOptimizationLevel.ALL -> OrtSession.SessionOptions.OptLevel.ALL_OPT
            }
        )
        o.setSessionLogLevel(
            when (logSeverityLevel) {
                LoggingLevel.VERBOSE -> OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE
                LoggingLevel.INFO -> OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO
                LoggingLevel.WARNING -> OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING
                LoggingLevel.ERROR -> OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
                LoggingLevel.FATAL -> OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL
            }
        )
        for (provider in executionProviders) {
            when (provider) {
                ExecutionProvider.NNAPI -> o.addNnapi()
                ExecutionProvider.XNNPACK -> o.addXnnpack(emptyMap())
                ExecutionProvider.CORE_ML -> o.addCoreML()
            }
        }
        return o
    } catch (t: Throwable) {
        o.close()
        throw t
    }
}

private fun Tensor.toOnnxTensor(): OnnxTensor {
    val shape = shape
    return when (type) {
        TensorElementType.FLOAT -> OnnxTensor.createTensor(environment, FloatBuffer.wrap(data as FloatArray), shape)
        TensorElementType.DOUBLE -> OnnxTensor.createTensor(environment, DoubleBuffer.wrap(data as DoubleArray), shape)
        TensorElementType.INT32 -> OnnxTensor.createTensor(environment, IntBuffer.wrap(data as IntArray), shape)
        TensorElementType.INT64 -> OnnxTensor.createTensor(environment, LongBuffer.wrap(data as LongArray), shape)
        TensorElementType.INT8 ->
            OnnxTensor.createTensor(environment, ByteBuffer.wrap(data as ByteArray), shape, OnnxJavaType.INT8)
        TensorElementType.UINT8 ->
            OnnxTensor.createTensor(environment, ByteBuffer.wrap(data as ByteArray), shape, OnnxJavaType.UINT8)
        TensorElementType.BOOL -> {
            val bools = data as BooleanArray
            val bytes = ByteArray(bools.size) { if (bools[it]) 1 else 0 }
            OnnxTensor.createTensor(environment, ByteBuffer.wrap(bytes), shape, OnnxJavaType.BOOL)
        }
        TensorElementType.STRING -> OnnxTensor.createTensor(environment, stringData(), shape)
    }
}

private fun OnnxTensor.toTensor(): Tensor {
    val shape = info.shape
    return when (val t = info.onnxType) {
        OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ->
            Tensor.wrap(TensorElementType.FLOAT, shape, floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } })
        OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE ->
            Tensor.wrap(TensorElementType.DOUBLE, shape, doubleBuffer.let { b -> DoubleArray(b.remaining()).also { b.get(it) } })
        OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ->
            Tensor.wrap(TensorElementType.INT32, shape, intBuffer.let { b -> IntArray(b.remaining()).also { b.get(it) } })
        OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 ->
            Tensor.wrap(TensorElementType.INT64, shape, longBuffer.let { b -> LongArray(b.remaining()).also { b.get(it) } })
        OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ->
            Tensor.wrap(TensorElementType.INT8, shape, byteBuffer.readBytes())
        OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 ->
            Tensor.wrap(TensorElementType.UINT8, shape, byteBuffer.readBytes())
        OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL -> {
            val bytes = byteBuffer.readBytes()
            Tensor.wrap(TensorElementType.BOOL, shape, BooleanArray(bytes.size) { bytes[it] != 0.toByte() })
        }
        OnnxTensorType.ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING -> {
            val strings = ArrayList<String>()
            flattenStrings(value, strings)
            Tensor.wrap(TensorElementType.STRING, shape, strings.toTypedArray())
        }
        else -> throw OrtException("Unsupported output tensor element type: $t")
    }
}

@Suppress("UNCHECKED_CAST")
private fun Tensor.stringData(): Array<String> = data as Array<String>

private fun ByteBuffer.readBytes(): ByteArray = ByteArray(remaining()).also { get(it) }

private fun flattenStrings(value: Any?, out: MutableList<String>) {
    when (value) {
        is String -> out += value
        is Array<*> -> value.forEach { flattenStrings(it, out) }
        else -> throw OrtException("Unexpected string tensor content: $value")
    }
}
