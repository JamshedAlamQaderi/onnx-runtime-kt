package com.jamshedalamqaderi.kmp.onnxrtkt

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class TensorTest {
    @Test
    fun `default shape is one-dimensional`() {
        val t = Tensor.ofFloats(floatArrayOf(1f, 2f, 3f))
        assertContentEquals(longArrayOf(3), t.shape)
        assertEquals(3, t.elementCount)
        assertEquals(TensorElementType.FLOAT, t.type)
    }

    @Test
    fun `shape must match element count`() {
        assertFailsWith<IllegalArgumentException> { Tensor.ofInts(intArrayOf(1, 2, 3), longArrayOf(2, 2)) }
    }

    @Test
    fun `scalar has empty shape`() {
        val t = Tensor.ofLongs(longArrayOf(42), longArrayOf())
        assertEquals(1, t.elementCount)
        assertContentEquals(longArrayOf(42), t.toLongArray())
    }

    @Test
    fun `reading with the wrong type fails`() {
        val t = Tensor.ofFloats(floatArrayOf(1f))
        assertFailsWith<OrtException> { t.toIntArray() }
    }

    @Test
    fun `tensor is immutable`() {
        val source = floatArrayOf(1f, 2f)
        val t = Tensor.ofFloats(source)
        source[0] = 99f
        t.toFloatArray()[1] = 99f
        assertContentEquals(floatArrayOf(1f, 2f), t.toFloatArray())
    }

    @Test
    @OptIn(ExperimentalUnsignedTypes::class)
    fun `uint8 round trips`() {
        val t = Tensor.ofUBytes(ubyteArrayOf(0u, 128u, 255u))
        assertContentEquals(ubyteArrayOf(0u, 128u, 255u), t.toUByteArray())
    }
}
