package com.jamshedalamqaderi.kmp.onnxrtkt.sample.cli

import com.jamshedalamqaderi.kmp.onnxrtkt.OnnxRuntime
import com.jamshedalamqaderi.kmp.onnxrtkt.OrtException
import com.jamshedalamqaderi.kmp.onnxrtkt.SessionOptions
import com.jamshedalamqaderi.kmp.onnxrtkt.Tensor
import com.jamshedalamqaderi.kmp.onnxrtkt.createSession
import com.jamshedalamqaderi.kmp.onnxrtkt.loadLibrary
import com.jamshedalamqaderi.kmp.onnxrtkt.runSingle
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.posix.fclose
import platform.posix.fopen
import platform.posix.fread
import kotlin.math.exp
import kotlin.math.roundToInt

private const val IMAGE_SIZE = 28
private const val MODEL_FILE = "mnist-12.onnx"
private const val INPUT_FILE = "mnist_test_input_0.bin"

/** Where the sample's files are found when no path is given: next to the binary's working directory, or in the repo. */
private val MODEL_CANDIDATES = listOf(MODEL_FILE, "sample/shared/composeResources/files/$MODEL_FILE", "../shared/composeResources/files/$MODEL_FILE")
private val INPUT_CANDIDATES = listOf(INPUT_FILE, "sample/shared/testResources/$INPUT_FILE", "../shared/testResources/$INPUT_FILE")

private const val USAGE = """Usage: onnxrt-digits [--model <mnist-12.onnx>] [--input <28x28 float32 .bin>] [--library <onnxruntime .so/.dll>]

Classifies a handwritten digit with the MNIST model from the ONNX Model Zoo.
  --model    the model (default: $MODEL_FILE in the current directory, or the sample's copy in this repo)
  --input    784 little-endian float32 pixels in [0, 1], row-major (default: the model zoo's test input, a 3)
  --library  use this ONNX Runtime library instead of the one embedded in the binary"""

/** Entry point of the CLI sample; returns the process exit code. */
fun runDigitCli(args: Array<String>): Int {
    val options = try {
        parseArgs(args)
    } catch (e: IllegalArgumentException) {
        println(e.message)
        println()
        println(USAGE)
        return 2
    }
    if (options.containsKey("help")) {
        println(USAGE)
        return 0
    }
    return try {
        options["library"]?.let { OnnxRuntime.loadLibrary(it) }
        val modelPath = options["model"] ?: findFile(MODEL_CANDIDATES, "model")
        val inputPath = options["input"] ?: findFile(INPUT_CANDIDATES, "input")
        val pixels = floatsFromLittleEndian(readFile(inputPath))
        require(pixels.size == IMAGE_SIZE * IMAGE_SIZE) { "$inputPath has ${pixels.size} floats, expected ${IMAGE_SIZE * IMAGE_SIZE}" }

        println("ONNX Runtime ${OnnxRuntime.version}")
        println("Model: $modelPath")
        println("Input: $inputPath")
        println()
        println(asciiArt(pixels))

        val probabilities = softmax(classify(readFile(modelPath), pixels))
        val best = probabilities.indices.maxBy { probabilities[it] }
        for (digit in probabilities.indices) {
            val p = probabilities[digit]
            val bar = "█".repeat((p * 40).roundToInt())
            val marker = if (digit == best) "  ◀" else ""
            println("  $digit  ${formatPercent(p)}  $bar$marker")
        }
        println()
        println("Prediction: $best (${formatPercent(probabilities[best]).trim()})")
        0
    } catch (e: OrtException) {
        println("ONNX Runtime error: ${e.message}")
        1
    } catch (e: IllegalArgumentException) {
        println("Error: ${e.message}")
        1
    }
}

/** Runs the MNIST model on 28×28 pixels and returns the 10 logits. */
fun classify(model: ByteArray, pixels: FloatArray): FloatArray =
    OnnxRuntime.createSession(model, SessionOptions(intraOpNumThreads = 1)).use { session ->
        val input = Tensor.ofFloats(pixels, longArrayOf(1, 1, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong()))
        session.runSingle(input).toFloatArray()
    }

fun softmax(values: FloatArray): FloatArray {
    val max = values.max()
    val exps = FloatArray(values.size) { exp(values[it] - max) }
    val sum = exps.sum()
    return FloatArray(exps.size) { exps[it] / sum }
}

/** Finds the first existing file among [candidates]. */
fun findFile(candidates: List<String>, what: String): String =
    candidates.firstOrNull { fileExists(it) }
        ?: throw IllegalArgumentException("No $what file found (looked for ${candidates.joinToString()}); pass --$what <path>")

internal fun defaultModelPath(): String = findFile(MODEL_CANDIDATES, "model")
internal fun defaultInputPath(): String = findFile(INPUT_CANDIDATES, "input")

private fun parseArgs(args: Array<String>): Map<String, String> {
    val result = mutableMapOf<String, String>()
    var i = 0
    while (i < args.size) {
        when (val arg = args[i]) {
            "-h", "--help" -> result["help"] = ""
            "--model", "--input", "--library" -> {
                require(i + 1 < args.size) { "$arg needs a value" }
                result[arg.removePrefix("--")] = args[++i]
            }
            else -> throw IllegalArgumentException("Unknown argument: $arg")
        }
        i++
    }
    return result
}

private fun asciiArt(pixels: FloatArray): String {
    val shades = " .:-=+*#%@"
    return (0 until IMAGE_SIZE).joinToString("\n") { y ->
        "  " + (0 until IMAGE_SIZE).joinToString("") { x ->
            val v = pixels[y * IMAGE_SIZE + x].coerceIn(0f, 1f)
            shades[(v * (shades.length - 1)).roundToInt()].toString().repeat(2)
        }
    }
}

private fun formatPercent(p: Float): String {
    val tenths = (p * 1000).roundToInt()
    return "${tenths / 10}.${tenths % 10}%".padStart(6)
}

internal fun floatsFromLittleEndian(bytes: ByteArray): FloatArray = FloatArray(bytes.size / 4) { i ->
    val o = i * 4
    Float.fromBits(
        (bytes[o].toInt() and 0xFF) or
            ((bytes[o + 1].toInt() and 0xFF) shl 8) or
            ((bytes[o + 2].toInt() and 0xFF) shl 16) or
            ((bytes[o + 3].toInt() and 0xFF) shl 24),
    )
}

private fun fileExists(path: String): Boolean {
    val file = fopen(path, "rb") ?: return false
    fclose(file)
    return true
}

internal fun readFile(path: String): ByteArray {
    val file = fopen(path, "rb") ?: throw IllegalArgumentException("Cannot open $path")
    try {
        var result = ByteArray(0)
        val buffer = ByteArray(1 shl 16)
        while (true) {
            val n = buffer.usePinned { fread(it.addressOf(0), 1u, buffer.size.toULong(), file).toInt() }
            if (n <= 0) break
            result += buffer.copyOf(n)
        }
        return result
    } finally {
        fclose(file)
    }
}
