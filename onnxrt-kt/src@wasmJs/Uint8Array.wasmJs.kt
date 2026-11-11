package com.jamshedalamqaderi.kmp.onnxrtkt

import kotlin.js.JsAny

internal actual fun ByteArray.toUint8Array(): JsAny {
    val array = newUint8Array(size)
    for (i in indices) setByte(array, i, this[i].toInt() and 0xFF)
    return array
}

private fun newUint8Array(size: Int): JsAny = js("new Uint8Array(size)")
private fun setByte(array: JsAny, index: Int, value: Int): Unit = js("{ array[index] = value; }")
