package com.jamshedalamqaderi.kmp.onnxrtkt

import kotlinx.coroutines.await
import kotlinx.coroutines.test.runTest
import kotlin.js.JsAny
import kotlin.js.Promise
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * Web tests. `./kotlin test -p wasmJs` runs them in a browser without onnxruntime-web, so they check the
 * error reporting. `tools/wasm-node-tests.mjs` runs them in Node with onnxruntime-web (it sets
 * `ORT_WEB_MODULE`), which runs real inference.
 */
class WebRuntimeTest {
    @Test
    fun `missing onnxruntime-web raises a helpful OrtException`() = runTest {
        if (ortWebModule() != null) return@runTest
        val e = assertFailsWith<OrtException> { OnnxRuntime.loadSession(TestModels.linear) }
        assertTrue("onnxruntime-web" in e.message.orEmpty(), e.message)
    }

    @Test
    @OptIn(ExperimentalUnsignedTypes::class)
    fun `runs models with onnxruntime-web`() = runTest {
        val module = ortWebModule() ?: return@runTest
        installOrt(importModule("file://$module").await())
        assertTrue(OnnxRuntime.version.first().isDigit(), OnnxRuntime.version)

        OnnxRuntime.loadSession(TestModels.linear).use { session ->
            assertEquals(listOf("x"), session.inputNames)
            val y = session.inferSingle(Tensor.ofFloats(floatArrayOf(1f, 2f, 3f), longArrayOf(3, 1)))
            assertContentEquals(longArrayOf(3, 1), y.shape)
            assertContentEquals(floatArrayOf(3f, 5f, 7f), y.toFloatArray())
        }

        OnnxRuntime.loadSession(TestModels.identityTypes).use { session ->
            val inputs = mapOf(
                "in_f32" to Tensor.ofFloats(floatArrayOf(1.5f, -2f)),
                "in_f64" to Tensor.ofDoubles(doubleArrayOf(1e-300, -2.25)),
                "in_i8" to Tensor.ofBytes(byteArrayOf(-128, 127)),
                "in_u8" to Tensor.ofUBytes(ubyteArrayOf(0u, 255u)),
                "in_i32" to Tensor.ofInts(intArrayOf(Int.MIN_VALUE, Int.MAX_VALUE)),
                "in_i64" to Tensor.ofLongs(longArrayOf(Long.MIN_VALUE, -1, 1L shl 40, Long.MAX_VALUE)),
                "in_b" to Tensor.ofBooleans(booleanArrayOf(true, false)),
                "in_s" to Tensor.ofStrings(arrayOf("hello", "ünïcødé ✓")),
            )
            val out = session.infer(inputs)
            assertContentEquals(floatArrayOf(1.5f, -2f), out.getValue("out_f32").toFloatArray())
            assertContentEquals(doubleArrayOf(1e-300, -2.25), out.getValue("out_f64").toDoubleArray())
            assertContentEquals(byteArrayOf(-128, 127), out.getValue("out_i8").toByteArray())
            assertContentEquals(ubyteArrayOf(0u, 255u), out.getValue("out_u8").toUByteArray())
            assertContentEquals(intArrayOf(Int.MIN_VALUE, Int.MAX_VALUE), out.getValue("out_i32").toIntArray())
            assertContentEquals(longArrayOf(Long.MIN_VALUE, -1, 1L shl 40, Long.MAX_VALUE), out.getValue("out_i64").toLongArray())
            assertContentEquals(booleanArrayOf(true, false), out.getValue("out_b").toBooleanArray())
            assertContentEquals(arrayOf("hello", "ünïcødé ✓"), out.getValue("out_s").toStringArray())
        }
    }
}

private fun ortWebModule(): String? =
    js("(typeof process !== 'undefined' && process.env && process.env.ORT_WEB_MODULE) || null")

private fun importModule(url: String): Promise<JsAny> = js("import(url)")

private fun installOrt(module: JsAny): Unit =
    js("{ globalThis.ort = (module.default && module.default.InferenceSession) ? module.default : module; }")
