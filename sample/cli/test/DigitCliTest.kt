package com.jamshedalamqaderi.kmp.onnxrtkt.sample.cli

import kotlin.test.Test
import kotlin.test.assertEquals

class DigitCliTest {
    /** Reference input/output from the ONNX Model Zoo (mnist-12 test_data_set_0). */
    @Test
    fun `matches the model zoo reference output`() {
        val logits = classify(readFile(defaultModelPath()), floatsFromLittleEndian(readFile(defaultInputPath())))
        val expected = floatArrayOf(
            -37.045937f, -5.5006814f, 24.384777f, 62.737465f, 35.51966f,
            33.86698f, -29.088867f, 38.012005f, -45.07267f, -71.59397f,
        )
        expected.forEachIndexed { i, e -> assertEquals(e, logits[i], 1e-3f, "logit $i") }
        assertEquals(3, logits.indices.maxBy { logits[it] })
    }

    @Test
    fun `runs end to end`() {
        assertEquals(0, runDigitCli(emptyArray()))
    }
}
