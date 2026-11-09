package com.jamshedalamqaderi.kmp.onnxrtkt.sample

import androidx.compose.ui.geometry.Offset
import kotlinx.coroutines.runBlocking
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals

class DigitClassifierJvmTest {
    private val classifier = runBlocking { DigitClassifier.load() }

    @AfterTest
    fun tearDown() = classifier.close()

    /** Reference input/output from the ONNX Model Zoo (mnist-12 test_data_set_0). */
    @Test
    fun `matches the model zoo reference output`() {
        val bytes = checkNotNull(javaClass.classLoader.getResourceAsStream("mnist_test_input_0.bin")).use { it.readBytes() }
        val input = FloatArray(784).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
        val expected = floatArrayOf(
            -37.045937f, -5.5006814f, 24.384777f, 62.737465f, 35.51966f,
            33.86698f, -29.088867f, 38.012005f, -45.07267f, -71.59397f,
        )
        val prediction = classifier.classify(input)
        expected.forEachIndexed { i, e -> assertEquals(e, prediction.logits[i], 1e-3f, "logit $i") }
        assertEquals(3, prediction.digit)
    }

    @Test
    fun `recognizes a drawn one`() {
        val one = listOf(listOf(Offset(150f, 40f), Offset(150f, 260f)))
        assertEquals(1, classify(one))
    }

    @Test
    fun `recognizes a drawn zero`() {
        val zero = listOf((0..48).map { i ->
            val a = 2 * PI * i / 48
            Offset((150 + 70 * cos(a)).toFloat(), (150 + 110 * sin(a)).toFloat())
        })
        assertEquals(0, classify(zero))
    }

    @Test
    fun `recognizes a drawn seven`() {
        val seven = listOf(listOf(Offset(60f, 40f), Offset(240f, 40f), Offset(120f, 280f)))
        assertEquals(7, classify(seven))
    }

    private fun classify(strokes: List<List<Offset>>): Int =
        classifier.classify(DigitRasterizer.rasterize(strokes)!!).digit
}
