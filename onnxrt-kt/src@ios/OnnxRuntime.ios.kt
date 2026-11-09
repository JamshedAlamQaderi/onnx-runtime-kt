package com.jamshedalamqaderi.kmp.onnxrtkt

import kotlinx.cinterop.CPointer
import kotlinx.cinterop.ObjCObjectVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import platform.Foundation.NSData
import platform.Foundation.NSError
import platform.Foundation.NSFileManager
import platform.Foundation.NSMutableData
import platform.Foundation.NSNumber
import platform.Foundation.NSTemporaryDirectory
import platform.Foundation.NSUUID
import platform.Foundation.create
import platform.Foundation.writeToFile
import platform.posix.memcpy
// Objective-C API of ONNX Runtime, imported from the `onnxruntime` Swift package (see module.yaml).
import swiftPMImport.onnxrt.kt.ORTCoreMLExecutionProviderOptions
import swiftPMImport.onnxrt.kt.ORTEnv
import swiftPMImport.onnxrt.kt.ORTGraphOptimizationLevel
import swiftPMImport.onnxrt.kt.ORTLoggingLevel
import swiftPMImport.onnxrt.kt.ORTSession
import swiftPMImport.onnxrt.kt.ORTSessionOptions
import swiftPMImport.onnxrt.kt.ORTTensorElementDataType
import swiftPMImport.onnxrt.kt.ORTValue
import swiftPMImport.onnxrt.kt.ORTVersion
import swiftPMImport.onnxrt.kt.appendCoreMLExecutionProviderWithOptions

private typealias ErrorPtr = CPointer<ObjCObjectVar<NSError?>>

/** Calls an Objective-C API that reports failures through an `NSError**` out-parameter. */
private inline fun <T : Any> ortCall(what: String, block: (ErrorPtr) -> T?): T = memScoped {
    val error = alloc<ObjCObjectVar<NSError?>>()
    val result = try {
        block(error.ptr)
    } catch (t: Throwable) {
        throw OrtException("$what failed: ${error.value?.localizedDescription ?: t.message}", t)
    }
    result ?: throw OrtException("$what failed: ${error.value?.localizedDescription ?: "unknown error"}")
}

private inline fun ortCheck(what: String, block: (ErrorPtr) -> Boolean) {
    ortCall(what) { err -> if (block(err)) Unit else null }
}

private val environment: ORTEnv by lazy {
    ortCall("Creating ORTEnv") { err -> ORTEnv(loggingLevel = ORTLoggingLevel.ORTLoggingLevelWarning, error = err) }
}

internal actual fun platformVersion(): String = ORTVersion() ?: "unknown"

internal actual fun platformCreateSession(model: ModelSource, options: SessionOptions): InferenceSession {
    val ortOptions = options.toOrt()
    return when (model) {
        is ModelSource.File -> IosInferenceSession(openSession(model.path, ortOptions))
        // The Objective-C API only loads models from files, so spill the bytes to a temporary file.
        is ModelSource.Bytes -> {
            val path = NSTemporaryDirectory() + "onnxrtkt-" + NSUUID().UUIDString + ".onnx"
            val data = model.bytes.usePinned { NSData.create(bytes = it.addressOf(0), length = model.bytes.size.toULong()) }
            if (!data.writeToFile(path, atomically = true)) throw OrtException("Could not write model to $path")
            try {
                IosInferenceSession(openSession(path, ortOptions))
            } finally {
                NSFileManager.defaultManager.removeItemAtPath(path, error = null)
            }
        }
    }
}

private fun openSession(path: String, options: ORTSessionOptions): ORTSession =
    ortCall("Creating ORTSession for $path") { err ->
        ORTSession(env = environment, modelPath = path, sessionOptions = options, error = err)
    }

private class IosInferenceSession(session: ORTSession) : InferenceSession {
    private var session: ORTSession? = session

    override val inputNames: List<String> =
        ortCall("Reading input names") { err -> session.inputNamesWithError(err) }.map { it as String }
    override val outputNames: List<String> =
        ortCall("Reading output names") { err -> session.outputNamesWithError(err) }.map { it as String }

    override fun run(inputs: Map<String, Tensor>, outputNames: Collection<String>): Map<String, Tensor> {
        val active = session ?: throw OrtException("Session is closed")
        val ortInputs: Map<Any?, Any?> = inputs.mapValues { (_, tensor) -> tensor.toOrtValue() }
        val outputs = ortCall("Running session") { err ->
            active.runWithInputs(inputs = ortInputs, outputNames = outputNames.toSet(), runOptions = null, error = err)
        }
        return outputs.entries.associate { (name, value) -> (name as String) to (value as ORTValue).toTensor() }
    }

    override fun close() {
        // ORTSession is reference counted; dropping our reference releases the native session.
        session = null
    }
}

private fun SessionOptions.toOrt(): ORTSessionOptions {
    val o = ortCall("Creating ORTSessionOptions") { err -> ORTSessionOptions(error = err) }
    if (intraOpNumThreads > 0) {
        ortCheck("Setting intra-op threads") { err -> o.setIntraOpNumThreads(intraOpNumThreads, error = err) }
    }
    ortCheck("Setting optimization level") { err ->
        o.setGraphOptimizationLevel(
            when (optimizationLevel) {
                GraphOptimizationLevel.NONE -> ORTGraphOptimizationLevel.ORTGraphOptimizationLevelNone
                GraphOptimizationLevel.BASIC -> ORTGraphOptimizationLevel.ORTGraphOptimizationLevelBasic
                GraphOptimizationLevel.EXTENDED -> ORTGraphOptimizationLevel.ORTGraphOptimizationLevelExtended
                GraphOptimizationLevel.ALL -> ORTGraphOptimizationLevel.ORTGraphOptimizationLevelAll
            },
            error = err,
        )
    }
    ortCheck("Setting log level") { err ->
        o.setLogSeverityLevel(
            when (logSeverityLevel) {
                LoggingLevel.VERBOSE -> ORTLoggingLevel.ORTLoggingLevelVerbose
                LoggingLevel.INFO -> ORTLoggingLevel.ORTLoggingLevelInfo
                LoggingLevel.WARNING -> ORTLoggingLevel.ORTLoggingLevelWarning
                LoggingLevel.ERROR -> ORTLoggingLevel.ORTLoggingLevelError
                LoggingLevel.FATAL -> ORTLoggingLevel.ORTLoggingLevelFatal
            },
            error = err,
        )
    }
    for (provider in executionProviders) {
        when (provider) {
            ExecutionProvider.CORE_ML -> ortCheck("Adding Core ML") { err ->
                o.appendCoreMLExecutionProviderWithOptions(ORTCoreMLExecutionProviderOptions(), error = err)
            }
            ExecutionProvider.XNNPACK -> ortCheck("Adding XNNPACK") { err ->
                o.appendExecutionProvider("XNNPACK", providerOptions = emptyMap<Any?, Any?>(), error = err)
            }
            ExecutionProvider.NNAPI -> throw OrtException("NNAPI is only available on Android")
        }
    }
    return o
}

private fun Tensor.toOrtValue(): ORTValue {
    val shape = shape.map { NSNumber(longLong = it) }
    if (type == TensorElementType.STRING) {
        @Suppress("UNCHECKED_CAST")
        val strings = (data as Array<String>).toList()
        return ortCall("Creating string tensor") { err -> ORTValue(tensorStringData = strings, shape = shape, error = err) }
    }
    val (elementType, bytes) = when (type) {
        TensorElementType.FLOAT -> ORTTensorElementDataType.ORTTensorElementDataTypeFloat to
            (data as FloatArray).let { a -> mutableData(a.size * 4) { d -> a.usePinned { d.appendBytes(it.addressOf(0), (a.size * 4).toULong()) } } }
        TensorElementType.INT32 -> ORTTensorElementDataType.ORTTensorElementDataTypeInt32 to
            (data as IntArray).let { a -> mutableData(a.size * 4) { d -> a.usePinned { d.appendBytes(it.addressOf(0), (a.size * 4).toULong()) } } }
        TensorElementType.INT64 -> ORTTensorElementDataType.ORTTensorElementDataTypeInt64 to
            (data as LongArray).let { a -> mutableData(a.size * 8) { d -> a.usePinned { d.appendBytes(it.addressOf(0), (a.size * 8).toULong()) } } }
        TensorElementType.INT8 -> ORTTensorElementDataType.ORTTensorElementDataTypeInt8 to
            (data as ByteArray).let { a -> mutableData(a.size) { d -> a.usePinned { d.appendBytes(it.addressOf(0), a.size.toULong()) } } }
        TensorElementType.UINT8 -> ORTTensorElementDataType.ORTTensorElementDataTypeUInt8 to
            (data as ByteArray).let { a -> mutableData(a.size) { d -> a.usePinned { d.appendBytes(it.addressOf(0), a.size.toULong()) } } }
        TensorElementType.DOUBLE, TensorElementType.BOOL, TensorElementType.STRING ->
            throw OrtException("$type tensors are not supported by the ONNX Runtime Objective-C API on iOS")
    }
    return ortCall("Creating $type tensor") { err ->
        ORTValue(tensorData = bytes, elementType = elementType, shape = shape, error = err)
    }
}

private inline fun mutableData(byteCount: Int, fill: (NSMutableData) -> Unit): NSMutableData {
    val data = NSMutableData()
    if (byteCount > 0) fill(data)
    return data
}

private fun ORTValue.toTensor(): Tensor {
    val info = ortCall("Reading tensor info") { err -> tensorTypeAndShapeInfoWithError(err) }
    val shape = info.shape.map { (it as NSNumber).longLongValue }.toLongArray()
    if (info.elementType == ORTTensorElementDataType.ORTTensorElementDataTypeString) {
        val strings = ortCall("Reading string tensor") { err -> tensorStringDataWithError(err) }.map { it as String }
        return Tensor.wrap(TensorElementType.STRING, shape, strings.toTypedArray())
    }
    val bytes = ortCall("Reading tensor data") { err -> tensorDataWithError(err) }
    val length = bytes.length.toInt()
    return when (val t = info.elementType) {
        ORTTensorElementDataType.ORTTensorElementDataTypeFloat ->
            Tensor.wrap(TensorElementType.FLOAT, shape, FloatArray(length / 4).also { a -> a.usePinned { copyOut(bytes, it.addressOf(0), length) } })
        ORTTensorElementDataType.ORTTensorElementDataTypeInt32 ->
            Tensor.wrap(TensorElementType.INT32, shape, IntArray(length / 4).also { a -> a.usePinned { copyOut(bytes, it.addressOf(0), length) } })
        ORTTensorElementDataType.ORTTensorElementDataTypeInt64 ->
            Tensor.wrap(TensorElementType.INT64, shape, LongArray(length / 8).also { a -> a.usePinned { copyOut(bytes, it.addressOf(0), length) } })
        ORTTensorElementDataType.ORTTensorElementDataTypeInt8 ->
            Tensor.wrap(TensorElementType.INT8, shape, ByteArray(length).also { a -> a.usePinned { copyOut(bytes, it.addressOf(0), length) } })
        ORTTensorElementDataType.ORTTensorElementDataTypeUInt8 ->
            Tensor.wrap(TensorElementType.UINT8, shape, ByteArray(length).also { a -> a.usePinned { copyOut(bytes, it.addressOf(0), length) } })
        else -> throw OrtException("Unsupported output tensor element type on iOS: $t")
    }
}

private fun copyOut(src: NSData, dst: CPointer<*>, length: Int) {
    if (length > 0) memcpy(dst, src.bytes, length.toULong())
}
