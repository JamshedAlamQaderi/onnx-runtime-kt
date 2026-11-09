package com.jamshedalamqaderi.kmp.onnxrtkt.sample

interface Platform {
    val name: String
}

expect fun getPlatform(): Platform
