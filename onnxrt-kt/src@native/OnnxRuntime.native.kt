package com.jamshedalamqaderi.kmp.onnxrtkt

import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ORTKT_BOOL
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ORTKT_DOUBLE
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ORTKT_FLOAT
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ORTKT_INT32
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ORTKT_INT64
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ORTKT_INT8
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ORTKT_PROVIDER_COREML
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ORTKT_PROVIDER_XNNPACK
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ORTKT_STRING
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ORTKT_UINT8
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_create_session
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_create_string_tensor
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_create_tensor
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_free_string
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_input_count
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_input_name
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_is_loaded
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_is_static
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_load_library
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_options
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_output_count
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_output_name
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_release_session
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_release_value
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_run
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_session
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_string_tensor_content
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_string_tensor_length
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_tensor_data
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_tensor_info
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_tensor_shape
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_value
import com.jamshedalamqaderi.kmp.onnxrtkt.cinterop.ortkt_version
import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.COpaquePointer
import kotlinx.cinterop.COpaquePointerVar
import kotlinx.cinterop.CPointer
import kotlinx.cinterop.CPointerVar
import kotlinx.cinterop.IntVar
import kotlinx.cinterop.LongVar
import kotlinx.cinterop.ULongVar
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.allocArrayOf
import kotlinx.cinterop.cstr
import kotlinx.cinterop.get
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.readBytes
import kotlinx.cinterop.set
import kotlinx.cinterop.toKString
import kotlinx.cinterop.usePinned
import kotlinx.cinterop.value
import platform.posix.memcpy
import kotlin.concurrent.AtomicReference

/**
 * Loads ONNX Runtime from [path] instead of the library embedded in this build.
 * Only meaningful on Linux and Windows, and only before ONNX Runtime is first used;
 * on Apple platforms ONNX Runtime is linked statically and this does nothing.
 *
 * Without this call, the library is taken from the `ONNXRUNTIME_LIBRARY_PATH` environment
 * variable or, by default, unpacked from this build into a per-user cache directory.
 *
 * @throws OrtException if the library cannot be loaded or ONNX Runtime is already loaded.
 */
public fun OnnxRuntime.loadLibrary(path: String) {
    if (ortkt_is_loaded() != 0) {
        if (ortkt_is_static() != 0) return
        throw OrtException("ONNX Runtime is already loaded; call loadLibrary() before using OnnxRuntime")
    }
    checkError("Loading ONNX Runtime", ortkt_load_library(path))
}

/** Loads the default library (environment variable or embedded copy) on first use. Thread-safe in C. */
private fun ensureLoaded() {
    if (ortkt_is_loaded() == 0) checkError("Loading ONNX Runtime", ortkt_load_library(null))
}

internal actual fun platformVersion(): String {
    ensureLoaded()
    return ortkt_version()?.toKString() ?: "unknown"
}

internal actual fun platformCreateSession(model: ModelSource, options: SessionOptions): InferenceSession {
    ensureLoaded()
    val bytes = when (model) {
        is ModelSource.Bytes -> model.bytes
        is ModelSource.File -> readFile(model.path)
    }
    if (bytes.isEmpty()) throw OrtException("Model is empty")
    return memScoped {
        val providers = options.executionProviders.map {
            when (it) {
                ExecutionProvider.CORE_ML -> ORTKT_PROVIDER_COREML
                ExecutionProvider.XNNPACK -> ORTKT_PROVIDER_XNNPACK
                ExecutionProvider.NNAPI -> throw OrtException("NNAPI is only available on Android")
                ExecutionProvider.WEBGPU -> throw OrtException("WebGPU is only available on the web")
            }
        }
        val o = alloc<ortkt_options>()
        o.intra_op_threads = options.intraOpNumThreads
        o.inter_op_threads = options.interOpNumThreads
        o.optimization_level = when (options.optimizationLevel) {
            GraphOptimizationLevel.NONE -> 0
            GraphOptimizationLevel.BASIC -> 1
            GraphOptimizationLevel.EXTENDED -> 2
            GraphOptimizationLevel.ALL -> 99
        }
        o.log_level = options.logSeverityLevel.ordinal
        o.provider_count = providers.size
        o.providers = if (providers.isEmpty()) null else allocArray<IntVar>(providers.size).also { array ->
            providers.forEachIndexed { i, p -> array[i] = p }
        }
        val out = alloc<CPointerVar<ortkt_session>>()
        bytes.usePinned { pinned ->
            checkError("Creating session", ortkt_create_session(pinned.addressOf(0), bytes.size.toULong(), o.ptr, out.ptr))
        }
        NativeInferenceSession(checkNotNull(out.value))
    }
}

private class NativeInferenceSession(handle: CPointer<ortkt_session>) : BlockingInferenceSession() {
    private val handle = AtomicReference<CPointer<ortkt_session>?>(handle)

    override val inputNames: List<String> =
        List(ortkt_input_count(handle).toInt()) { ortkt_input_name(handle, it.toULong())!!.toKString() }
    override val outputNames: List<String> =
        List(ortkt_output_count(handle).toInt()) { ortkt_output_name(handle, it.toULong())!!.toKString() }

    override fun runSync(inputs: Map<String, Tensor>, outputNames: Collection<String>): Map<String, Tensor> {
        val session = handle.value ?: throw OrtException("Session is closed")
        val names = outputNames.toList()
        val entries = inputs.entries.toList()
        val created = ArrayList<CPointer<ortkt_value>>(entries.size)
        return memScoped {
            try {
                entries.forEach { (name, tensor) -> created += tensor.toOrtValue("input '$name'") }
                val inputNames = allocArrayOf(entries.map { it.key.cstr.ptr })
                val inputValues = allocArrayOf(created)
                val outputNamesC = allocArrayOf(names.map { it.cstr.ptr })
                val outputs = allocArray<CPointerVar<ortkt_value>>(names.size.coerceAtLeast(1))
                checkError(
                    "Running session",
                    ortkt_run(
                        session,
                        inputNames, inputValues, entries.size.toULong(),
                        outputNamesC, names.size.toULong(), outputs,
                    ),
                )
                val values = List(names.size) { checkNotNull(outputs[it]) }
                try {
                    names.indices.associate { names[it] to values[it].toTensor("output '${names[it]}'") }
                } finally {
                    values.forEach { ortkt_release_value(it) }
                }
            } finally {
                created.forEach { ortkt_release_value(it) }
            }
        }
    }

    override fun close() {
        handle.getAndSet(null)?.let { ortkt_release_session(it) }
    }
}

// ------------------------------------------------------------------------------- tensors

private fun Tensor.toOrtValue(what: String): CPointer<ortkt_value> = memScoped {
    val dims = shape
    val shapePtr = allocArray<LongVar>(dims.size.coerceAtLeast(1)).also { a ->
        dims.forEachIndexed { i, d -> a[i] = d }
    }
    val rank = dims.size.toULong()
    val out = alloc<CPointerVar<ortkt_value>>()
    if (type == TensorElementType.STRING) {
        @Suppress("UNCHECKED_CAST")
        val strings = data as Array<String>
        val array = allocArray<CPointerVar<ByteVar>>(strings.size.coerceAtLeast(1))
        strings.forEachIndexed { i, s -> array[i] = s.cstr.ptr }
        checkError("Creating $what", ortkt_create_string_tensor(array, strings.size.toULong(), shapePtr, rank, out.ptr))
    } else {
        val (elementType, raw) = when (type) {
            TensorElementType.FLOAT -> ORTKT_FLOAT to data
            TensorElementType.DOUBLE -> ORTKT_DOUBLE to data
            TensorElementType.INT32 -> ORTKT_INT32 to data
            TensorElementType.INT64 -> ORTKT_INT64 to data
            TensorElementType.INT8 -> ORTKT_INT8 to data
            TensorElementType.UINT8 -> ORTKT_UINT8 to data
            TensorElementType.BOOL -> ORTKT_BOOL to (data as BooleanArray).let { b -> ByteArray(b.size) { if (b[it]) 1 else 0 } }
            TensorElementType.STRING -> error("unreachable")
        }
        withPinnedData(raw) { ptr, byteSize ->
            checkError("Creating $what", ortkt_create_tensor(elementType, ptr, byteSize.toULong(), shapePtr, rank, out.ptr))
        }
    }
    checkNotNull(out.value)
}

/** Calls [block] with a pointer to the array's elements (null when empty) and its size in bytes. */
private inline fun <R> withPinnedData(array: Any, block: (COpaquePointer?, Long) -> R): R = when (array) {
    is FloatArray -> if (array.isEmpty()) block(null, 0) else array.usePinned { block(it.addressOf(0), array.size * 4L) }
    is DoubleArray -> if (array.isEmpty()) block(null, 0) else array.usePinned { block(it.addressOf(0), array.size * 8L) }
    is IntArray -> if (array.isEmpty()) block(null, 0) else array.usePinned { block(it.addressOf(0), array.size * 4L) }
    is LongArray -> if (array.isEmpty()) block(null, 0) else array.usePinned { block(it.addressOf(0), array.size * 8L) }
    is ByteArray -> if (array.isEmpty()) block(null, 0) else array.usePinned { block(it.addressOf(0), array.size.toLong()) }
    else -> error("Unsupported tensor storage: ${array::class}")
}

private fun CPointer<ortkt_value>.toTensor(what: String): Tensor = memScoped {
    val typeVar = alloc<IntVar>()
    val rankVar = alloc<ULongVar>()
    val countVar = alloc<ULongVar>()
    checkError("Reading $what", ortkt_tensor_info(this@toTensor, typeVar.ptr, rankVar.ptr, countVar.ptr))
    val rank = rankVar.value.toInt()
    val count = countVar.value.toInt()
    val shapePtr = allocArray<LongVar>(rank.coerceAtLeast(1))
    checkError("Reading $what", ortkt_tensor_shape(this@toTensor, shapePtr, rank.toULong()))
    val shape = LongArray(rank) { shapePtr[it] }

    val elementType = typeVar.value
    if (elementType == ORTKT_STRING) {
        return@memScoped Tensor.wrap(TensorElementType.STRING, shape, readStrings(this@toTensor, count, what))
    }
    val type = when (elementType) {
        ORTKT_FLOAT -> TensorElementType.FLOAT
        ORTKT_DOUBLE -> TensorElementType.DOUBLE
        ORTKT_INT32 -> TensorElementType.INT32
        ORTKT_INT64 -> TensorElementType.INT64
        ORTKT_INT8 -> TensorElementType.INT8
        ORTKT_UINT8 -> TensorElementType.UINT8
        ORTKT_BOOL -> TensorElementType.BOOL
        else -> throw OrtException("Unsupported element type $elementType in $what")
    }
    val dataVar = alloc<COpaquePointerVar>()
    if (count > 0) checkError("Reading $what", ortkt_tensor_data(this@toTensor, dataVar.ptr))
    val src = dataVar.value
    val data: Any = when (type) {
        TensorElementType.FLOAT -> FloatArray(count).also { copyOut(src, it, count * 4L) }
        TensorElementType.DOUBLE -> DoubleArray(count).also { copyOut(src, it, count * 8L) }
        TensorElementType.INT32 -> IntArray(count).also { copyOut(src, it, count * 4L) }
        TensorElementType.INT64 -> LongArray(count).also { copyOut(src, it, count * 8L) }
        TensorElementType.INT8, TensorElementType.UINT8 -> ByteArray(count).also { copyOut(src, it, count.toLong()) }
        TensorElementType.BOOL -> ByteArray(count).also { copyOut(src, it, count.toLong()) }
            .let { b -> BooleanArray(count) { b[it] != 0.toByte() } }
        TensorElementType.STRING -> error("unreachable")
    }
    Tensor.wrap(type, shape, data)
}

private fun copyOut(src: COpaquePointer?, dst: Any, byteSize: Long) {
    if (byteSize == 0L || src == null) return
    withPinnedData(dst) { ptr, _ -> memcpy(ptr, src, byteSize.toULong()) }
}

private fun readStrings(value: CPointer<ortkt_value>, count: Int, what: String): Array<String> = memScoped {
    if (count == 0) return@memScoped emptyArray()
    val totalVar = alloc<ULongVar>()
    checkError("Reading $what", ortkt_string_tensor_length(value, totalVar.ptr))
    val total = totalVar.value.toInt()
    val buffer = allocArray<ByteVar>(total.coerceAtLeast(1))
    val offsets = allocArray<ULongVar>(count)
    checkError("Reading $what", ortkt_string_tensor_content(value, buffer, total.toULong(), offsets, count.toULong()))
    val bytes = if (total > 0) buffer.readBytes(total) else ByteArray(0)
    Array(count) { i ->
        val start = offsets[i].toInt()
        val end = if (i + 1 < count) offsets[i + 1].toInt() else total
        bytes.decodeToString(start, end)
    }
}

// ------------------------------------------------------------------------------- helpers

/** Throws [OrtException] for a non-null error string returned by the C layer (and frees it). */
private fun checkError(what: String, error: CPointer<ByteVar>?) {
    if (error == null) return
    val message = error.toKString()
    ortkt_free_string(error)
    throw OrtException("$what failed: $message")
}

private fun readFile(path: String): ByteArray {
    val file = platform.posix.fopen(path, "rb") ?: throw OrtException("Cannot open model file $path")
    try {
        val chunks = ArrayList<ByteArray>()
        var total = 0
        val buffer = ByteArray(1 shl 16)
        while (true) {
            val n = buffer.usePinned { platform.posix.fread(it.addressOf(0), 1u, buffer.size.toULong(), file).toInt() }
            if (n <= 0) break
            chunks += buffer.copyOf(n)
            total += n
        }
        if (platform.posix.ferror(file) != 0) throw OrtException("Error reading model file $path")
        val result = ByteArray(total)
        var offset = 0
        for (c in chunks) {
            c.copyInto(result, offset)
            offset += c.size
        }
        return result
    } finally {
        platform.posix.fclose(file)
    }
}
