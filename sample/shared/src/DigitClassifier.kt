package com.jamshedalamqaderi.kmp.onnxrtkt.sample

import com.jamshedalamqaderi.kmp.onnxrtkt.InferenceSession
import com.jamshedalamqaderi.kmp.onnxrtkt.OnnxRuntime
import com.jamshedalamqaderi.kmp.onnxrtkt.SessionOptions
import com.jamshedalamqaderi.kmp.onnxrtkt.Tensor
import com.jamshedalamqaderi.kmp.onnxrtkt.createSession
import com.jamshedalamqaderi.kmp.onnxrtkt.runSingle
import com.jamshedalamqaderi.kmp.onnxrtkt.sample.resources.Res
import kotlin.math.exp

/**
 * Handwritten digit recognizer backed by the MNIST model from the ONNX Model Zoo
 * (https://github.com/onnx/models/tree/main/validated/vision/classification/mnist, MIT license).
 *
 * Model contract:
 *  - input  `Input3`:           float32 [1, 1, 28, 28], white digit on black background, values in [0, 1]
 *  - output `Plus214_Output_0`: float32 [1, 10], one logit per digit (apply softmax for probabilities)
 */
class DigitClassifier(modelBytes: ByteArray) : AutoCloseable {
    private val session: InferenceSession = OnnxRuntime.createSession(modelBytes, SessionOptions(intraOpNumThreads = 1))

    fun classify(pixels: FloatArray): Prediction {
        require(pixels.size == IMAGE_SIZE * IMAGE_SIZE) { "Expected ${IMAGE_SIZE * IMAGE_SIZE} pixels, got ${pixels.size}" }
        val input = Tensor.ofFloats(pixels, shape = longArrayOf(1, 1, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
        val logits = session.runSingle(input).toFloatArray()
        return Prediction(logits)
    }

    override fun close() = session.close()

    companion object {
        const val IMAGE_SIZE = 28
        const val MODEL_PATH = "files/mnist-12.onnx"

        /**
         * Reads the model when it is not available as a Compose resource. The native macOS app sets this:
         * the Kotlin Toolchain does not package Compose resources next to `macos/app` executables.
         */
        var fallbackModelLoader: (suspend () -> ByteArray)? = null

        /** Loads the model bundled as a Compose Multiplatform resource (or through [fallbackModelLoader]). */
        suspend fun load(): DigitClassifier {
            val bytes = try {
                Res.readBytes(MODEL_PATH)
            } catch (e: Exception) {
                fallbackModelLoader?.invoke() ?: throw e
            }
            return DigitClassifier(bytes)
        }
    }
}

class Prediction(val logits: FloatArray) {
    val probabilities: FloatArray = softmax(logits)
    val digit: Int = probabilities.indices.maxBy { probabilities[it] }
    val confidence: Float get() = probabilities[digit]
}

internal fun softmax(values: FloatArray): FloatArray {
    val max = values.max()
    val exps = FloatArray(values.size) { exp(values[it] - max) }
    val sum = exps.sum()
    return FloatArray(exps.size) { exps[it] / sum }
}
