package com.jamshedalamqaderi.kmp.onnxrtkt

import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** Real inference on the JVM and every native target (Apple, Linux, Windows). */
class InferenceSessionTest {
    @Test
    fun `reports runtime version`() {
        assertTrue(OnnxRuntime.version.first().isDigit(), "version: ${OnnxRuntime.version}")
    }

    @Test
    fun `exposes input and output names`() {
        OnnxRuntime.createSession(TestModels.linear).use { session ->
            assertEquals(listOf("x"), session.inputNames)
            assertEquals(listOf("y"), session.outputNames)
        }
    }

    @Test
    fun `runs linear model`() {
        OnnxRuntime.createSession(TestModels.linear, SessionOptions(intraOpNumThreads = 1)).use { session ->
            val x = Tensor.ofFloats(floatArrayOf(1f, 2f, 3f), longArrayOf(3, 1))
            val y = session.run(mapOf("x" to x)).getValue("y")
            assertContentEquals(longArrayOf(3, 1), y.shape)
            assertContentEquals(floatArrayOf(3f, 5f, 7f), y.toFloatArray())
        }
    }

    @Test
    fun `runSingle uses the only input and first output`() {
        OnnxRuntime.createSession(TestModels.linear).use { session ->
            val y = session.runSingle(Tensor.ofFloats(floatArrayOf(10f), longArrayOf(1, 1)))
            assertContentEquals(floatArrayOf(21f), y.toFloatArray())
        }
    }

    @Test
    fun `suspend API loads and infers`() = runTest {
        OnnxRuntime.loadSession(TestModels.linear).use { session ->
            val y = session.inferSingle(Tensor.ofFloats(floatArrayOf(0f, -1f), longArrayOf(2, 1)))
            assertContentEquals(floatArrayOf(1f, -1f), y.toFloatArray())
        }
    }

    @Test
    @OptIn(ExperimentalUnsignedTypes::class)
    fun `every element type round trips`() {
        OnnxRuntime.createSession(TestModels.identityTypes).use { session ->
            val inputs = mapOf(
                "in_f32" to Tensor.ofFloats(floatArrayOf(1.5f, -2f, Float.MAX_VALUE)),
                "in_f64" to Tensor.ofDoubles(doubleArrayOf(1e-300, -2.25)),
                "in_i8" to Tensor.ofBytes(byteArrayOf(-128, 0, 127)),
                "in_u8" to Tensor.ofUBytes(ubyteArrayOf(0u, 200u, 255u)),
                "in_i32" to Tensor.ofInts(intArrayOf(Int.MIN_VALUE, 0, Int.MAX_VALUE)),
                "in_i64" to Tensor.ofLongs(longArrayOf(Long.MIN_VALUE, -1, 1L shl 40, Long.MAX_VALUE)),
                "in_b" to Tensor.ofBooleans(booleanArrayOf(true, false, true)),
                "in_s" to Tensor.ofStrings(arrayOf("hello", "", "ünïcødé ✓")),
            )
            val out = session.run(inputs)
            assertContentEquals(floatArrayOf(1.5f, -2f, Float.MAX_VALUE), out.getValue("out_f32").toFloatArray())
            assertContentEquals(doubleArrayOf(1e-300, -2.25), out.getValue("out_f64").toDoubleArray())
            assertContentEquals(byteArrayOf(-128, 0, 127), out.getValue("out_i8").toByteArray())
            assertContentEquals(ubyteArrayOf(0u, 200u, 255u), out.getValue("out_u8").toUByteArray())
            assertContentEquals(intArrayOf(Int.MIN_VALUE, 0, Int.MAX_VALUE), out.getValue("out_i32").toIntArray())
            assertContentEquals(
                longArrayOf(Long.MIN_VALUE, -1, 1L shl 40, Long.MAX_VALUE),
                out.getValue("out_i64").toLongArray(),
            )
            assertContentEquals(booleanArrayOf(true, false, true), out.getValue("out_b").toBooleanArray())
            assertContentEquals(arrayOf("hello", "", "ünïcødé ✓"), out.getValue("out_s").toStringArray())
        }
    }

    @Test
    fun `empty tensors round trip`() {
        OnnxRuntime.createSession(TestModels.linear).use { session ->
            val y = session.runSingle(Tensor.ofFloats(floatArrayOf(), longArrayOf(0, 1)))
            assertContentEquals(longArrayOf(0, 1), y.shape)
            assertEquals(0, y.toFloatArray().size)
        }
    }

    @Test
    fun `invalid model raises OrtException`() {
        assertFailsWith<OrtException> { OnnxRuntime.createSession(byteArrayOf(1, 2, 3)) }
    }

    @Test
    fun `missing model file raises OrtException`() {
        assertFailsWith<OrtException> { OnnxRuntime.createSession("/definitely/not/here.onnx") }
    }

    @Test
    fun `wrong input type raises OrtException`() {
        OnnxRuntime.createSession(TestModels.linear).use { session ->
            assertFailsWith<OrtException> {
                session.run(mapOf("x" to Tensor.ofInts(intArrayOf(1), longArrayOf(1, 1))))
            }
        }
    }

    @Test
    fun `closed session raises OrtException`() {
        val session = OnnxRuntime.createSession(TestModels.linear)
        session.close()
        session.close() // idempotent
        assertFailsWith<OrtException> {
            session.run(mapOf("x" to Tensor.ofFloats(floatArrayOf(1f), longArrayOf(1, 1))))
        }
    }

    @Test
    fun `WebGPU is rejected outside the web`() {
        assertFailsWith<OrtException> {
            OnnxRuntime.createSession(TestModels.linear, SessionOptions(executionProviders = listOf(ExecutionProvider.WEBGPU)))
        }
    }
}
