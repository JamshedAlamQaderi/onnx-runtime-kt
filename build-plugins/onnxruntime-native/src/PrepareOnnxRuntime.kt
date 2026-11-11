package com.jamshedalamqaderi.kmp.onnxrtkt.build

import org.jetbrains.amper.plugins.Input
import org.jetbrains.amper.plugins.Output
import org.jetbrains.amper.plugins.TaskAction
import java.io.InputStream
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.nio.file.StandardCopyOption
import java.security.MessageDigest
import java.util.UUID
import java.util.zip.GZIPInputStream
import java.util.zip.ZipInputStream
import kotlin.io.path.Path
import kotlin.io.path.createDirectories
import kotlin.io.path.deleteIfExists
import kotlin.io.path.deleteRecursively
import kotlin.io.path.exists
import kotlin.io.path.inputStream
import kotlin.io.path.moveTo
import kotlin.io.path.name
import kotlin.io.path.readText
import kotlin.io.path.writeBytes
import kotlin.io.path.writeText

/**
 * Prepares ONNX Runtime for one Kotlin/Native target and writes the cinterop `.def` file for it.
 *
 * - Apple targets: the static `onnxruntime.xcframework` from Microsoft's CocoaPods archive.
 *   The static library is packed into the cinterop klib, so consumers link it automatically.
 * - Linux and Windows: the official shared library from the GitHub release, embedded as bytes
 *   (C23 `#embed`) and loaded at runtime by `ortkt.c`.
 *
 * Every archive is checked against its pinned SHA-256 before use.
 */
@TaskAction
@OptIn(kotlin.io.path.ExperimentalPathApi::class)
fun prepareOnnxRuntime(
    target: String,
    @Input shimDir: Path,
    @Output defFile: Path,
) {
    val spec = TARGETS[target] ?: error("Unsupported target '$target'; expected one of ${TARGETS.keys}")
    val outDir = defFile.parent
    if (outDir.exists()) outDir.deleteRecursively()
    val includeDir = outDir.resolve("include").createDirectories()
    val libDir = outDir.resolve("lib").createDirectories()

    val archive = download(spec.archive)
    val headers = mutableMapOf<String, ByteArray>()
    var library: ByteArray? = null
    readArchive(archive, spec.archive.isZip) { path, bytes ->
        when {
            path.startsWith(spec.headersPrefix) && path.endsWith(".h") ->
                headers[path.substringAfterLast('/')] = bytes()
            path == spec.libraryPath -> library = bytes()
        }
    }
    check(headers.containsKey("onnxruntime_c_api.h")) { "onnxruntime_c_api.h not found in ${spec.archive.url}" }
    val libraryBytes = checkNotNull(library) { "${spec.libraryPath} not found in ${spec.archive.url}" }

    headers.forEach { (name, bytes) -> includeDir.resolve(name).writeBytes(bytes) }
    shimDir.resolve("ortkt.h").also { it.toFile().copyTo(includeDir.resolve(it.name).toFile(), overwrite = true) }

    val shimSource = shimDir.resolve("ortkt.c").readText()
    val def = StringBuilder()
    def.appendLine("package = com.jamshedalamqaderi.kmp.onnxrtkt.cinterop")
    def.appendLine("headers = ortkt.h")
    def.appendLine("headerFilter = ortkt.h")

    when (spec.linkage) {
        Linkage.STATIC -> {
            val thin = thinMachO(libraryBytes, CPU_TYPE_ARM64)
            libDir.resolve("libonnxruntime.a").writeBytes(thin)
            def.appendLine("compilerOpts = -I$includeDir -DORTKT_STATIC")
            def.appendLine("staticLibraries = libonnxruntime.a")
            def.appendLine("libraryPaths = $libDir")
            def.appendLine("linkerOpts = ${spec.linkerOpts}")
            def.appendLine("---")
            def.append(shimSource)
        }
        Linkage.EMBEDDED -> {
            val blob = libDir.resolve(spec.libraryPath.substringAfterLast('/'))
            blob.writeBytes(libraryBytes)
            def.appendLine("compilerOpts = -I$includeDir -std=gnu2x")
            if (spec.linkerOpts.isNotEmpty()) def.appendLine("linkerOpts = ${spec.linkerOpts}")
            def.appendLine("---")
            def.append(shimSource)
            def.appendLine()
            def.appendLine("const unsigned char ortkt_library_bytes[] = {")
            def.appendLine("#embed \"$blob\"")
            def.appendLine("};")
            def.appendLine("const size_t ortkt_library_size = sizeof(ortkt_library_bytes);")
        }
    }
    defFile.writeText(def.toString())
    println("ONNX Runtime ${spec.archive.version} for $target prepared in $outDir")
}

// ------------------------------------------------------------------------------------ targets

private enum class Linkage { STATIC, EMBEDDED }

private class Archive(val version: String, val url: String, val sha256: String) {
    val isZip get() = url.endsWith(".zip")
    val fileName get() = url.substringAfterLast('/')
}

private class TargetSpec(
    val archive: Archive,
    val headersPrefix: String,
    val libraryPath: String,
    val linkage: Linkage,
    val linkerOpts: String,
)

/** Microsoft's newest CocoaPods archive; it lags behind the GitHub releases. */
private const val APPLE_VERSION = "1.30.0"
private const val DESKTOP_VERSION = "1.31.1"

private val APPLE_ARCHIVE = Archive(
    APPLE_VERSION,
    "https://download.onnxruntime.ai/pod-archive-onnxruntime-c-$APPLE_VERSION.zip",
    "e6f1670c14406fd9f082bb400ab197a9b0a9646058ca6366e440642e2b54a2ea",
)

private fun githubRelease(asset: String, sha256: String) = Archive(
    DESKTOP_VERSION,
    "https://github.com/microsoft/onnxruntime/releases/download/v$DESKTOP_VERSION/onnxruntime-$asset-$DESKTOP_VERSION." +
        if (asset.startsWith("win")) "zip" else "tgz",
    sha256,
)

private const val XCFRAMEWORK = "onnxruntime.xcframework"
private const val MACOS_LINKER_OPTS = "-framework Foundation -framework CoreML -framework Network -lc++"
private const val IOS_LINKER_OPTS = "-framework Foundation -framework CoreML -framework Network -framework UIKit -lc++"

private val TARGETS: Map<String, TargetSpec> = mapOf(
    "iosArm64" to TargetSpec(
        APPLE_ARCHIVE, "Headers/", "$XCFRAMEWORK/ios-arm64/onnxruntime.framework/onnxruntime",
        Linkage.STATIC, IOS_LINKER_OPTS,
    ),
    "iosSimulatorArm64" to TargetSpec(
        APPLE_ARCHIVE, "Headers/", "$XCFRAMEWORK/ios-arm64_x86_64-simulator/onnxruntime.framework/onnxruntime",
        Linkage.STATIC, IOS_LINKER_OPTS,
    ),
    "macosArm64" to TargetSpec(
        APPLE_ARCHIVE, "Headers/", "$XCFRAMEWORK/macos-arm64_x86_64/onnxruntime.framework/Versions/A/onnxruntime",
        Linkage.STATIC, MACOS_LINKER_OPTS,
    ),
    "linuxX64" to TargetSpec(
        githubRelease("linux-x64", "c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9"),
        "onnxruntime-linux-x64-$DESKTOP_VERSION/include/",
        "onnxruntime-linux-x64-$DESKTOP_VERSION/lib/libonnxruntime.so.$DESKTOP_VERSION",
        Linkage.EMBEDDED, "-ldl",
    ),
    "linuxArm64" to TargetSpec(
        githubRelease("linux-aarch64", "5317505e2da99be65f8e2775ba3c2afd6d2961fd78aba1bd8f56159f48820c78"),
        "onnxruntime-linux-aarch64-$DESKTOP_VERSION/include/",
        "onnxruntime-linux-aarch64-$DESKTOP_VERSION/lib/libonnxruntime.so.$DESKTOP_VERSION",
        Linkage.EMBEDDED, "-ldl",
    ),
    "mingwX64" to TargetSpec(
        githubRelease("win-x64", "5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c"),
        "onnxruntime-win-x64-$DESKTOP_VERSION/include/",
        "onnxruntime-win-x64-$DESKTOP_VERSION/lib/onnxruntime.dll",
        Linkage.EMBEDDED, "",
    ),
)

// ------------------------------------------------------------------------------- downloading

/**
 * Downloads [archive] into a per-user cache shared by all targets and checks its SHA-256.
 * Parallel tasks in this build synchronize on an interned (JVM-wide) key; parallel builds use a file lock.
 */
private fun download(archive: Archive): Path = synchronized("onnxrt-kt-download:${archive.url}".intern()) {
    downloadLocked(archive)
}

private fun downloadLocked(archive: Archive): Path {
    val cacheDir = Path(System.getProperty("user.home"), ".cache", "onnxrt-kt", "downloads").createDirectories()
    val file = cacheDir.resolve(archive.fileName)
    FileChannel.open(cacheDir.resolve("${archive.fileName}.lock"), StandardOpenOption.CREATE, StandardOpenOption.WRITE).use { channel ->
        channel.lock().use {
            if (file.exists() && sha256(file) == archive.sha256) return file

            println("Downloading ${archive.url}")
            val tmp = cacheDir.resolve("${archive.fileName}.${UUID.randomUUID()}.part")
            try {
                val client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL).build()
                val request = HttpRequest.newBuilder(URI(archive.url)).GET().build()
                val response = client.send(request, HttpResponse.BodyHandlers.ofFile(tmp))
                check(response.statusCode() == 200) { "Downloading ${archive.url} failed with HTTP ${response.statusCode()}" }
                val actual = sha256(tmp)
                check(actual == archive.sha256) { "Checksum mismatch for ${archive.url}: expected ${archive.sha256}, got $actual" }
                tmp.moveTo(file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
            } finally {
                tmp.deleteIfExists()
            }
            return file
        }
    }
}

private fun sha256(file: Path): String {
    val digest = MessageDigest.getInstance("SHA-256")
    file.inputStream().use { input ->
        val buffer = ByteArray(1 shl 16)
        while (true) {
            val n = input.read(buffer)
            if (n < 0) break
            digest.update(buffer, 0, n)
        }
    }
    return digest.digest().joinToString("") { "%02x".format(it) }
}

// -------------------------------------------------------------------------- archive reading

/** Calls [onEntry] for each regular file; `bytes()` reads the entry's content. */
private fun readArchive(file: Path, isZip: Boolean, onEntry: (path: String, bytes: () -> ByteArray) -> Unit) {
    if (isZip) {
        ZipInputStream(file.inputStream().buffered()).use { zip ->
            while (true) {
                val entry = zip.nextEntry ?: break
                if (!entry.isDirectory) onEntry(entry.name) { zip.readBytes() }
            }
        }
    } else {
        GZIPInputStream(file.inputStream().buffered(), 1 shl 16).use { readTar(it, onEntry) }
    }
}

/** Minimal tar reader: ustar names with prefix, GNU long names and pax `path` records. */
private fun readTar(input: InputStream, onEntry: (path: String, bytes: () -> ByteArray) -> Unit) {
    val header = ByteArray(512)
    var longName: String? = null
    while (true) {
        if (input.readNBytes(header, 0, 512) < 512 || header.all { it == 0.toByte() }) return
        val size = String(header, 124, 12, Charsets.US_ASCII).trim(' ', '\u0000').ifEmpty { "0" }.toLong(8)
        val type = header[156].toInt().toChar()
        val name = longName ?: run {
            val n = cString(header, 0, 100)
            val prefix = if (String(header, 257, 5, Charsets.US_ASCII) == "ustar") cString(header, 345, 155) else ""
            if (prefix.isEmpty()) n else "$prefix/$n"
        }
        longName = null
        val padded = (size + 511) / 512 * 512
        var consumed = false
        val readContent = {
            consumed = true
            input.readNBytes(size.toInt()).also { input.skipNBytes(padded - size) }
        }
        when (type) {
            'L' -> longName = String(readContent(), Charsets.UTF_8).trimEnd('\u0000')
            'x' -> longName = String(readContent(), Charsets.UTF_8).lineSequence()
                .map { it.substringAfter(' ') }
                .firstOrNull { it.startsWith("path=") }?.removePrefix("path=")
            '0', '\u0000' -> onEntry(name.removePrefix("./"), readContent)
        }
        if (!consumed) input.skipNBytes(padded)
    }
}

private fun cString(bytes: ByteArray, offset: Int, length: Int): String {
    var end = offset
    while (end < offset + length && bytes[end] != 0.toByte()) end++
    return String(bytes, offset, end - offset, Charsets.UTF_8)
}

// ------------------------------------------------------------------------------ Mach-O thinning

private const val FAT_MAGIC = 0xCAFEBABE.toInt()
private const val CPU_TYPE_ARM64 = 0x0100000C

/** Extracts one architecture from a universal (fat) Mach-O file; returns thin files unchanged. */
private fun thinMachO(bytes: ByteArray, cpuType: Int): ByteArray {
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN)
    if (buf.getInt(0) != FAT_MAGIC) return bytes
    val count = buf.getInt(4)
    for (i in 0 until count) {
        val base = 8 + i * 20
        if (buf.getInt(base) == cpuType) {
            val offset = buf.getInt(base + 8)
            val size = buf.getInt(base + 12)
            return bytes.copyOfRange(offset, offset + size)
        }
    }
    error("Architecture 0x${cpuType.toString(16)} not found in universal binary")
}
