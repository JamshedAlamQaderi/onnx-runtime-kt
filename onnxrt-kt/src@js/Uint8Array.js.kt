package com.jamshedalamqaderi.kmp.onnxrtkt

import kotlin.js.JsAny

// On Kotlin/JS a ByteArray is an Int8Array, so this is a zero-copy view of the same memory.
internal actual fun ByteArray.toUint8Array(): JsAny = int8ToUint8(this)

private fun int8ToUint8(bytes: ByteArray): JsAny = js("new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.length)")
