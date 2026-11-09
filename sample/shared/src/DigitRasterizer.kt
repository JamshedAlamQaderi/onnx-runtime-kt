package com.jamshedalamqaderi.kmp.onnxrtkt.sample

import androidx.compose.ui.geometry.Offset
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Turns freehand strokes into the 28×28 grayscale image MNIST models expect, mimicking how the
 * MNIST dataset was prepared: the digit is scaled to fit a 20×20 box (keeping its aspect ratio),
 * anti-aliased, and then shifted so its center of mass sits in the middle of the 28×28 image.
 */
object DigitRasterizer {
    private const val SIZE = DigitClassifier.IMAGE_SIZE
    private const val DIGIT_BOX = 20f
    private const val SUPERSAMPLE = 4
    private const val BRUSH_RADIUS = 1.3f // in 28×28 pixels

    /** Returns [SIZE]×[SIZE] row-major pixels in [0, 1], or `null` when nothing has been drawn. */
    fun rasterize(strokes: List<List<Offset>>): FloatArray? {
        val points = strokes.flatten()
        if (points.isEmpty()) return null

        val minX = points.minOf { it.x }
        val maxX = points.maxOf { it.x }
        val minY = points.minOf { it.y }
        val maxY = points.maxOf { it.y }
        val extent = max(max(maxX - minX, maxY - minY), 1f)
        val scale = (DIGIT_BOX - 2 * BRUSH_RADIUS) / extent * SUPERSAMPLE
        val centerX = (minX + maxX) / 2
        val centerY = (minY + maxY) / 2
        val hiRes = SIZE * SUPERSAMPLE
        val half = hiRes / 2f

        fun map(p: Offset) = Offset((p.x - centerX) * scale + half, (p.y - centerY) * scale + half)

        val canvas = FloatArray(hiRes * hiRes)
        val radius = BRUSH_RADIUS * SUPERSAMPLE
        for (stroke in strokes) {
            val mapped = stroke.map(::map)
            if (mapped.size == 1) stamp(canvas, hiRes, mapped[0], radius)
            for (i in 1 until mapped.size) {
                val a = mapped[i - 1]
                val b = mapped[i]
                val steps = max(1, ceil(hypot(b.x - a.x, b.y - a.y)).toInt())
                for (s in 0..steps) {
                    val t = s.toFloat() / steps
                    stamp(canvas, hiRes, Offset(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t), radius)
                }
            }
        }

        // Downsample (box filter) to 28×28 for anti-aliasing.
        val image = FloatArray(SIZE * SIZE)
        for (y in 0 until SIZE) for (x in 0 until SIZE) {
            var sum = 0f
            for (dy in 0 until SUPERSAMPLE) for (dx in 0 until SUPERSAMPLE) {
                sum += canvas[(y * SUPERSAMPLE + dy) * hiRes + x * SUPERSAMPLE + dx]
            }
            image[y * SIZE + x] = sum / (SUPERSAMPLE * SUPERSAMPLE)
        }
        return centerOfMass(image)
    }

    private fun stamp(canvas: FloatArray, size: Int, c: Offset, r: Float) {
        val x0 = max(0, floor(c.x - r).toInt())
        val x1 = minOf(size - 1, ceil(c.x + r).toInt())
        val y0 = max(0, floor(c.y - r).toInt())
        val y1 = minOf(size - 1, ceil(c.y + r).toInt())
        for (y in y0..y1) for (x in x0..x1) {
            if (hypot(x + 0.5f - c.x, y + 0.5f - c.y) <= r) canvas[y * size + x] = 1f
        }
    }

    private fun centerOfMass(image: FloatArray): FloatArray {
        var total = 0f
        var sx = 0f
        var sy = 0f
        for (y in 0 until SIZE) for (x in 0 until SIZE) {
            val v = image[y * SIZE + x]
            total += v
            sx += v * x
            sy += v * y
        }
        if (total == 0f) return image
        val shiftX = ((SIZE - 1) / 2f - sx / total).roundToInt()
        val shiftY = ((SIZE - 1) / 2f - sy / total).roundToInt()
        val shifted = FloatArray(SIZE * SIZE)
        for (y in 0 until SIZE) for (x in 0 until SIZE) {
            val sxp = x - shiftX
            val syp = y - shiftY
            if (sxp in 0 until SIZE && syp in 0 until SIZE) shifted[y * SIZE + x] = image[syp * SIZE + sxp]
        }
        return shifted
    }
}
