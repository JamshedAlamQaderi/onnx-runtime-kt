package com.jamshedalamqaderi.kmp.onnxrtkt.sample

import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.usePinned
import platform.AppKit.NSApp
import platform.AppKit.NSApplication
import platform.AppKit.NSApplicationActivationPolicy
import platform.Foundation.NSBundle
import platform.Foundation.NSData
import platform.Foundation.NSFileManager
import platform.Foundation.dataWithContentsOfFile
import platform.posix.memcpy

/** Native macOS app (Kotlin/Native + Compose Multiplatform); ONNX Runtime is linked statically. */
fun main() {
    DigitClassifier.fallbackModelLoader = ::readModelFromDisk
    NSApplication.sharedApplication()
    NSApp?.setActivationPolicy(NSApplicationActivationPolicy.NSApplicationActivationPolicyRegular)
    Window(title = "onnxrt-kt digits", size = DpSize(480.dp, 820.dp)) {
        App()
    }
    NSApp?.activateIgnoringOtherApps(true)
    NSApp?.run()
}

private const val MODEL_FILE = "mnist-12.onnx"
private const val REPO_MODEL_PATH = "sample/shared/composeResources/files/$MODEL_FILE"

/**
 * The Kotlin Toolchain builds `macos/app` as a bare executable without Compose resources, so read the
 * model from next to the executable, or from this repository (found by walking up from the executable).
 */
@OptIn(ExperimentalForeignApi::class)
private suspend fun readModelFromDisk(): ByteArray {
    val fm = NSFileManager.defaultManager
    val executableDir = NSBundle.mainBundle.executablePath?.substringBeforeLast('/') ?: fm.currentDirectoryPath
    val candidates = buildList {
        add("$executableDir/$MODEL_FILE")
        var dir = executableDir
        while (dir.isNotEmpty() && dir != "/") {
            add("$dir/$REPO_MODEL_PATH")
            dir = dir.substringBeforeLast('/', "")
        }
        add("${fm.currentDirectoryPath}/$REPO_MODEL_PATH")
    }
    val path = candidates.firstOrNull { fm.fileExistsAtPath(it) }
        ?: error("$MODEL_FILE not found; put it next to the executable (looked in ${candidates.first()})")
    val data = NSData.dataWithContentsOfFile(path) ?: error("Cannot read $path")
    return ByteArray(data.length.toInt()).also { bytes ->
        if (bytes.isNotEmpty()) bytes.usePinned { memcpy(it.addressOf(0), data.bytes, data.length) }
    }
}
