package com.jamshedalamqaderi.kmp.onnxrtkt.sample

import androidx.compose.ui.geometry.Offset
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class DigitRasterizerTest {
    @Test
    fun `empty drawing produces no image`() {
        assertNull(DigitRasterizer.rasterize(emptyList()))
    }

    @Test
    fun `image is 28x28 with values in 0 to 1`() {
        val pixels = DigitRasterizer.rasterize(listOf(listOf(Offset(10f, 10f), Offset(200f, 300f))))!!
        assertEquals(28 * 28, pixels.size)
        assertTrue(pixels.all { it in 0f..1f })
        assertTrue(pixels.any { it > 0.9f })
    }

    @Test
    fun `softmax sums to one`() {
        val p = softmax(floatArrayOf(1f, 2f, 3f, -50f))
        assertEquals(1f, p.sum(), 1e-5f)
    }
}
