package com.jamshedalamqaderi.kmp.onnxrtkt.sample

class JVMPlatform : Platform {
    override val name: String = "Java ${System.getProperty("java.version")}"
}

actual fun getPlatform(): Platform = JVMPlatform()
