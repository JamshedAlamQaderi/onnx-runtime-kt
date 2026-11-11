package com.jamshedalamqaderi.kmp.onnxrtkt

import kotlinx.coroutines.await
import kotlin.js.JsAny
import kotlin.js.Promise
import kotlin.js.toJsString

/*
 * Web implementation on onnxruntime-web (https://onnxruntime.ai/docs/get-started/with-javascript/web.html).
 *
 * It uses the global `ort` object, which the page provides by loading onnxruntime-web, e.g.
 *   <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.min.js"></script>
 * or, with a bundler, `import * as ort from "onnxruntime-web"; globalThis.ort = ort;`.
 *
 * All JavaScript access goes through the small `js(...)` helpers at the bottom of this file, so the
 * same code compiles for Kotlin/JS and Kotlin/Wasm.
 */

internal actual fun platformVersion(): String {
    requireOrt()
    return ortVersion() ?: "unknown"
}

internal actual suspend fun platformLoadSession(model: ModelSource, options: SessionOptions): InferenceSession {
    requireOrt()
    val jsOptions = options.toJs()
    val promise = when (model) {
        is ModelSource.Bytes -> ortCreateSession(model.bytes.toUint8Array(), jsOptions)
        is ModelSource.File -> ortCreateSession(model.path.toJsString(), jsOptions)
    }
    val session = ortAwait("Creating session") { promise.await<JsAny>() }
    return WebInferenceSession(session)
}

private class WebInferenceSession(private var session: JsAny?) : InferenceSession {
    override val inputNames: List<String> = session!!.let { s -> stringList(sessionInputNames(s)) }
    override val outputNames: List<String> = session!!.let { s -> stringList(sessionOutputNames(s)) }

    override suspend fun infer(inputs: Map<String, Tensor>, outputNames: Collection<String>): Map<String, Tensor> {
        val active = session ?: throw OrtException("Session is closed")
        val feeds = newObject()
        for ((name, tensor) in inputs) setProperty(feeds, name, tensor.toOrtTensor())
        val fetches = newArray()
        for (name in outputNames) pushString(fetches, name)
        val results = ortAwait("Running session") { sessionRun(active, feeds, fetches).await<JsAny>() }
        return outputNames.associateWith { name ->
            val value = getProperty(results, name) ?: throw OrtException("Output '$name' missing from results")
            value.toTensor(name)
        }
    }

    override fun close() {
        session?.let { sessionRelease(it) }
        session = null
    }
}

// ------------------------------------------------------------------------------- options

private fun SessionOptions.toJs(): JsAny {
    val o = newObject()
    if (intraOpNumThreads > 0) setInt(o, "intraOpNumThreads", intraOpNumThreads)
    if (interOpNumThreads > 0) setInt(o, "interOpNumThreads", interOpNumThreads)
    setString(
        o, "graphOptimizationLevel",
        when (optimizationLevel) {
            GraphOptimizationLevel.NONE -> "disabled"
            GraphOptimizationLevel.BASIC -> "basic"
            GraphOptimizationLevel.EXTENDED -> "extended"
            GraphOptimizationLevel.ALL -> "all"
        },
    )
    setInt(o, "logSeverityLevel", logSeverityLevel.ordinal)
    if (executionProviders.isNotEmpty()) {
        val providers = newArray()
        for (p in executionProviders) {
            when (p) {
                ExecutionProvider.WEBGPU -> pushString(providers, "webgpu")
                else -> throw OrtException("$p is not available on the web; use WEBGPU or the default (WebAssembly)")
            }
        }
        pushString(providers, "wasm") // CPU fallback, as on the other platforms
        setProperty(o, "executionProviders", providers)
    }
    return o
}

// ------------------------------------------------------------------------------- tensors

private fun Tensor.toOrtTensor(): JsAny {
    val dims = newArray()
    for (d in shape) pushNumber(dims, d.toDouble())
    val n = elementCount.toInt()
    val (typeName, array) = when (type) {
        TensorElementType.FLOAT -> "float32" to newTypedArray("Float32Array", n).also { a ->
            val v = data as FloatArray
            for (i in 0 until n) setNumber(a, i, v[i].toDouble())
        }
        TensorElementType.DOUBLE -> "float64" to newTypedArray("Float64Array", n).also { a ->
            val v = data as DoubleArray
            for (i in 0 until n) setNumber(a, i, v[i])
        }
        TensorElementType.INT32 -> "int32" to newTypedArray("Int32Array", n).also { a ->
            val v = data as IntArray
            for (i in 0 until n) setNumber(a, i, v[i].toDouble())
        }
        TensorElementType.INT64 -> "int64" to newTypedArray("BigInt64Array", n).also { a ->
            val v = data as LongArray
            for (i in 0 until n) setInt64(a, i, (v[i] ushr 32).toInt(), v[i].toInt())
        }
        TensorElementType.INT8 -> "int8" to newTypedArray("Int8Array", n).also { a ->
            val v = data as ByteArray
            for (i in 0 until n) setNumber(a, i, v[i].toDouble())
        }
        TensorElementType.UINT8 -> "uint8" to (data as ByteArray).toUint8Array()
        TensorElementType.BOOL -> "bool" to newTypedArray("Uint8Array", n).also { a ->
            val v = data as BooleanArray
            for (i in 0 until n) setNumber(a, i, if (v[i]) 1.0 else 0.0)
        }
        TensorElementType.STRING -> "string" to newArray().also { a ->
            @Suppress("UNCHECKED_CAST")
            for (s in data as Array<String>) pushString(a, s)
        }
    }
    return newOrtTensor(typeName, array, dims)
}

private fun JsAny.toTensor(name: String): Tensor {
    val typeName = tensorType(this)
    val dimsJs = tensorDims(this)
    val shape = LongArray(arrayLength(dimsJs)) { getNumber(dimsJs, it).toLong() }
    val a = tensorData(this)
    val n = arrayLength(a)
    return when (typeName) {
        "float32" -> Tensor.wrap(TensorElementType.FLOAT, shape, FloatArray(n) { getNumber(a, it).toFloat() })
        "float64" -> Tensor.wrap(TensorElementType.DOUBLE, shape, DoubleArray(n) { getNumber(a, it) })
        "int32" -> Tensor.wrap(TensorElementType.INT32, shape, IntArray(n) { getNumber(a, it).toInt() })
        "int64" -> Tensor.wrap(TensorElementType.INT64, shape, LongArray(n) {
            (getInt64High(a, it).toLong() shl 32) or (getInt64Low(a, it).toLong() and 0xFFFFFFFFL)
        })
        "int8" -> Tensor.wrap(TensorElementType.INT8, shape, ByteArray(n) { getNumber(a, it).toInt().toByte() })
        "uint8" -> Tensor.wrap(TensorElementType.UINT8, shape, ByteArray(n) { getNumber(a, it).toInt().toByte() })
        "bool" -> Tensor.wrap(TensorElementType.BOOL, shape, BooleanArray(n) { getNumber(a, it) != 0.0 })
        "string" -> Tensor.wrap(TensorElementType.STRING, shape, Array(n) { getString(a, it) })
        else -> throw OrtException("Unsupported element type '$typeName' in output '$name'")
    }
}

private fun stringList(array: JsAny): List<String> = List(arrayLength(array)) { getString(array, it) }

// ------------------------------------------------------------------------------- helpers

/** Copies the bytes into a new Uint8Array (zero-copy view on Kotlin/JS). */
internal expect fun ByteArray.toUint8Array(): JsAny

private fun requireOrt() {
    if (!ortAvailable()) {
        throw OrtException(
            "onnxruntime-web is not loaded: the global `ort` object is missing. Load it before using OnnxRuntime, " +
                "e.g. <script src=\"https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js\"></script>",
        )
    }
}

private suspend fun <T> ortAwait(what: String, block: suspend () -> T): T = try {
    block()
} catch (e: OrtException) {
    throw e
} catch (t: Throwable) {
    throw OrtException("$what failed: ${t.message ?: t.toString()}", t)
}

// ----------------------------------------------- JavaScript access (valid on both JS and Wasm)

private fun ortAvailable(): Boolean = js("typeof globalThis.ort !== 'undefined' && globalThis.ort.InferenceSession !== undefined")
private fun ortVersion(): String? = js("(globalThis.ort.env && globalThis.ort.env.versions && (globalThis.ort.env.versions.web || globalThis.ort.env.versions.common)) || null")
private fun ortCreateSession(model: JsAny, options: JsAny): Promise<JsAny> = js("globalThis.ort.InferenceSession.create(model, options)")
private fun sessionInputNames(session: JsAny): JsAny = js("session.inputNames")
private fun sessionOutputNames(session: JsAny): JsAny = js("session.outputNames")
private fun sessionRun(session: JsAny, feeds: JsAny, fetches: JsAny): Promise<JsAny> = js("session.run(feeds, fetches)")
private fun sessionRelease(session: JsAny): Unit = js("{ session.release().catch(function () {}); }")
private fun newOrtTensor(type: String, data: JsAny, dims: JsAny): JsAny = js("new globalThis.ort.Tensor(type, data, dims)")
private fun tensorType(tensor: JsAny): String = js("tensor.type")
private fun tensorDims(tensor: JsAny): JsAny = js("tensor.dims")
private fun tensorData(tensor: JsAny): JsAny = js("tensor.data")

private fun newObject(): JsAny = js("({})")
private fun newArray(): JsAny = js("[]")
private fun newTypedArray(kind: String, size: Int): JsAny = js("new globalThis[kind](size)")
private fun setProperty(o: JsAny, key: String, value: JsAny): Unit = js("{ o[key] = value; }")
private fun setString(o: JsAny, key: String, value: String): Unit = js("{ o[key] = value; }")
private fun setInt(o: JsAny, key: String, value: Int): Unit = js("{ o[key] = value; }")
private fun getProperty(o: JsAny, key: String): JsAny? = js("o[key]")
private fun pushString(a: JsAny, value: String): Unit = js("{ a.push(value); }")
private fun pushNumber(a: JsAny, value: Double): Unit = js("{ a.push(value); }")
private fun arrayLength(a: JsAny): Int = js("a.length")
private fun setNumber(a: JsAny, i: Int, value: Double): Unit = js("{ a[i] = value; }")
private fun getNumber(a: JsAny, i: Int): Double = js("Number(a[i])")
private fun getString(a: JsAny, i: Int): String = js("String(a[i])")
private fun setInt64(a: JsAny, i: Int, high: Int, low: Int): Unit = js("{ a[i] = (BigInt(high) << BigInt(32)) | BigInt(low >>> 0); }")
private fun getInt64High(a: JsAny, i: Int): Int = js("Number(BigInt.asIntN(32, a[i] >> BigInt(32)))")
private fun getInt64Low(a: JsAny, i: Int): Int = js("Number(BigInt.asIntN(32, a[i]))")
