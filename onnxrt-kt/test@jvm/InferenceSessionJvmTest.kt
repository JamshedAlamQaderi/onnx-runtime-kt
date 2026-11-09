package com.jamshedalamqaderi.kmp.onnxrtkt

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * `linear_model.onnx` (testResources) is a hand-built ONNX graph computing `y = x * 2 + 1`:
 * input `x` float32 [N, 1], output `y` float32 [N, 1], opset 13.
 */
class InferenceSessionJvmTest {
    private val model: ByteArray =
        checkNotNull(javaClass.classLoader.getResourceAsStream("linear_model.onnx")) { "test model missing" }
            .use { it.readBytes() }

    @Test
    fun `reports runtime version`() {
        assertTrue(OnnxRuntime.version.isNotBlank())
    }

    @Test
    fun `exposes input and output names`() {
        OnnxRuntime.createSession(model).use { session ->
            assertEquals(listOf("x"), session.inputNames)
            assertEquals(listOf("y"), session.outputNames)
        }
    }

    @Test
    fun `runs linear model`() {
        OnnxRuntime.createSession(model, SessionOptions(intraOpNumThreads = 1)).use { session ->
            val x = Tensor.ofFloats(floatArrayOf(1f, 2f, 3f), longArrayOf(3, 1))
            val y = session.run(mapOf("x" to x)).getValue("y")
            assertContentEquals(longArrayOf(3, 1), y.shape)
            assertContentEquals(floatArrayOf(3f, 5f, 7f), y.toFloatArray())
        }
    }

    @Test
    fun `runSingle uses the only input and first output`() {
        OnnxRuntime.createSession(model).use { session ->
            val y = session.runSingle(Tensor.ofFloats(floatArrayOf(10f), longArrayOf(1, 1)))
            assertContentEquals(floatArrayOf(21f), y.toFloatArray())
        }
    }

    @Test
    fun `invalid model raises OrtException`() {
        assertFailsWith<OrtException> { OnnxRuntime.createSession(byteArrayOf(1, 2, 3)) }
    }

    @Test
    fun `wrong input type raises OrtException`() {
        OnnxRuntime.createSession(model).use { session ->
            assertFailsWith<OrtException> {
                session.run(mapOf("x" to Tensor.ofInts(intArrayOf(1), longArrayOf(1, 1))))
            }
        }
    }
}
