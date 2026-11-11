package com.jamshedalamqaderi.kmp.onnxrtkt.sample

import com.jamshedalamqaderi.kmp.onnxrtkt.sample.cli.runDigitCli
import kotlin.system.exitProcess

/** Native command-line sample; the official ONNX Runtime library is embedded in the binary. */
fun main(args: Array<String>) {
    exitProcess(runDigitCli(args))
}
