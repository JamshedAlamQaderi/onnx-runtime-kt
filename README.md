# onnxrt-kt

Kotlin Multiplatform bindings for [ONNX Runtime](https://onnxruntime.ai). You load and run ONNX models from common
Kotlin code, and each platform uses Microsoft's official runtime underneath.

| Platform                                        | Backend                                                     | Version | Setup for apps                        |
|-------------------------------------------------|-------------------------------------------------------------|---------|---------------------------------------|
| JVM (desktop)                                   | `com.microsoft.onnxruntime:onnxruntime` (Java API)          | 1.31.1  | none                                  |
| Android                                         | `com.microsoft.onnxruntime:onnxruntime-android` (Java API)  | 1.31.1  | none                                  |
| iOS / iPadOS (`iosArm64`, `iosSimulatorArm64`)  | C API, static library from Microsoft's CocoaPods archive    | 1.30.0¹ | none (linked into the app)            |
| macOS (`macosArm64`)                            | C API, static library from Microsoft's CocoaPods archive    | 1.30.0¹ | none (linked into the app)            |
| Linux (`linuxX64`, `linuxArm64`)                | C API, official shared library from the GitHub release      | 1.31.1  | none (embedded, unpacked on first use) |
| Windows (`mingwX64`)                            | C API, official shared library from the GitHub release      | 1.31.1  | Visual C++ Redistributable²           |
| Web (`js`, `wasmJs`)                            | [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web) | your choice | load onnxruntime-web on the page  |

¹ Microsoft publishes static Apple builds through CocoaPods only, and that channel lags the GitHub releases.
² `onnxruntime.dll` needs `MSVCP140.dll`, `VCRUNTIME140.dll` and `VCRUNTIME140_1.dll`. Most Windows machines have
them; otherwise install the [Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)
or ship those DLLs next to your `.exe`.

The project is built with the [Kotlin Toolchain](https://kotlin-toolchain.org/latest/) (Kotlin 2.4.20,
Compose Multiplatform 1.12.1). The `./kotlin` wrapper script downloads the pinned toolchain on first use.

## Usage

The suspend API works on every platform, including the web:

```kotlin
import com.jamshedalamqaderi.kmp.onnxrtkt.*

val session = OnnxRuntime.loadSession(modelBytes)               // or loadSession("/path/to/model.onnx"), a URL on the web
session.use {
    val input = Tensor.ofFloats(pixels, shape = longArrayOf(1, 1, 28, 28))
    val outputs = it.infer(mapOf(it.inputNames.first() to input))  // Map<String, Tensor>
    val logits = outputs.getValue(it.outputNames.first()).toFloatArray()
}
```

Everywhere except the web, you can also call ONNX Runtime synchronously on the current thread:

```kotlin
val session = OnnxRuntime.createSession(modelBytes)
val logits = session.runSingle(input).toFloatArray()   // or session.run(mapOf(...))
```

- `Tensor` is an immutable tensor that does not depend on any platform. It supports `FLOAT`, `DOUBLE`, `INT8`,
  `UINT8`, `INT32`, `INT64`, `BOOL` and `STRING` on every platform.
- `SessionOptions` sets thread counts, the graph optimization level, the log level and execution providers
  (`NNAPI` on Android, `XNNPACK`, `CORE_ML` on Apple platforms, `WEBGPU` on the web).
- Errors are reported as `OrtException` on every platform.
- `infer` runs on `Dispatchers.Default` on non-web platforms, so it never blocks the caller.

### Linux and Windows

The official ONNX Runtime library is embedded in the `onnxrt-kt` klib. On first use it is unpacked once into a
per-user cache (`$XDG_CACHE_HOME/onnxrt-kt` or `~/.cache/onnxrt-kt` on Linux, `%LOCALAPPDATA%\onnxrt-kt` on Windows)
and loaded from there. Your app grows by the library's size (about 29 MB on Linux x64, 17 MB on Windows x64).

To use another build of ONNX Runtime (a GPU or reduced build, or a system copy), point to it before first use:

```kotlin
OnnxRuntime.loadLibrary("/opt/onnxruntime/lib/libonnxruntime.so.1")   // native only
```

or set the `ONNXRUNTIME_LIBRARY_PATH` environment variable. The official Linux library needs glibc 2.28 or newer.

### Web

The toolchain cannot declare npm dependencies, so the page provides onnxruntime-web as the global `ort` object:

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.min.js"></script>
```

With a bundler, `import * as ort from "onnxruntime-web"; globalThis.ort = ort;` does the same. Without it,
`loadSession` throws an `OrtException` that says so. `loadSession(path)` fetches the model from that URL.

## How the native targets are built

`build-plugins/onnxruntime-native` is a Kotlin Toolchain plugin that, for each native target:

1. downloads Microsoft's official archive once into `~/.cache/onnxrt-kt/downloads` and checks its pinned SHA-256:
   the CocoaPods archive for Apple targets, the GitHub release for Linux and Windows;
2. extracts the headers and the library (the arm64 slice of the static xcframework for Apple targets);
3. writes the cinterop `.def` for the target. Apple targets pack `libonnxruntime.a` into the klib
   (`staticLibraries`); Linux and Windows embed the shared library's bytes with C23 `#embed`.

Kotlin binds to `onnxrt-kt/native-shim/ortkt.h`, a small C layer over the ONNX Runtime C API (`ortkt.c`). Every
native target therefore gets the same bindings, and one Kotlin implementation (`src@native`) serves them all. To
update ONNX Runtime, change the versions and checksums at the top of the plugin's `PrepareOnnxRuntime.kt`.

## Project layout

```
onnxrt-kt/            the library (kmp/lib)
  src/                common API: OnnxRuntime, InferenceSession, Tensor, SessionOptions
  src@nonWeb/         synchronous API (createSession, run, runSingle) for every platform but the web
  src@jvmAndAndroid/  implementation on the ai.onnxruntime Java API (shared by JVM and Android)
  src@native/         implementation on the C API for Apple, Linux and Windows
  src@web/            implementation on onnxruntime-web, shared by Kotlin/JS and Kotlin/Wasm
  native-shim/        the C layer between Kotlin/Native and the ONNX Runtime C API
build-plugins/
  onnxruntime-native/ downloads ONNX Runtime and generates the cinterop definitions
sample/               "draw a digit" apps built with the MNIST model from the ONNX Model Zoo
  shared/             Compose UI, DigitClassifier (inference) and DigitRasterizer (preprocessing)
  androidApp/  desktopApp/ (JVM)  iosApp/  macosApp/ (native Compose)
  cli/                command-line classifier shared by the native Linux and Windows apps
  linuxApp/  windowsApp/
tools/                wasm-node-tests.mjs: runs the web tests in Node with onnxruntime-web
```

## Sample: handwritten digit recognition

The sample uses [`mnist-12.onnx`](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)
(26 KB, MIT license). When you finish drawing a stroke, it:

1. rasterizes your strokes into a 28×28 grayscale image, scaled and centered the way the MNIST dataset was
   (`DigitRasterizer`)
2. feeds it to the model as a `[1, 1, 28, 28]` float tensor (`DigitClassifier`)
3. applies softmax to the 10 logits and shows the predicted digit, the probability of each digit, and the exact
   tensor the model received

```bash
./kotlin run -m androidApp   # Android device or emulator
./kotlin run -m desktopApp   # desktop (JVM)
./kotlin run -m iosApp       # iOS / iPadOS simulator (macOS with Xcode only)
./kotlin run -m macosApp     # native macOS app (Compose Multiplatform, ONNX Runtime linked statically)
```

The native macOS app is a bare executable: the Kotlin Toolchain does not package Compose resources for
`macos/app`, so it reads `mnist-12.onnx` from next to the executable, or from this repository when run with
`./kotlin run`.

Compose Multiplatform has no native Linux or Windows target, so those samples are command-line apps. They
classify the model zoo's test digit (or any 28×28 float32 `.bin` passed with `--input`), print it as ASCII art,
and show each digit's probability:

```bash
./kotlin run -m linuxApp          # on Linux (x64 or arm64)
./kotlin run -m windowsApp        # on Windows
./kotlin run -m linuxApp -- --help
```

## Build and test

```bash
./kotlin build
./kotlin test -m onnxrt-kt -p jvm -p macosArm64 -p iosSimulatorArm64   # real inference, every element type
./kotlin test -m shared -p jvm                                          # sample tests (model zoo reference output)
./kotlin test -m cli -p macosArm64                                      # CLI sample (same reference output, native)
```

Where each platform's inference tests run:

| Platform | How |
|---|---|
| JVM, macOS, iOS simulator | `./kotlin test` on a Mac |
| Linux | `./kotlin test -p linuxX64` (or `linuxArm64`) on a Linux machine |
| Windows | `./kotlin test -p mingwX64` on Windows |
| Web (`wasmJs`) | `tools/wasm-node-tests.mjs` (see the file for the commands); `./kotlin test -p wasmJs` checks only the error path |
| Android | the local test runner can't load the Android library; inference tests need a device or emulator |
| Web (`js`) | compiled only; the toolchain does not run Kotlin/JS tests |

Coordinates: `com.jamshedalamqaderi.kmp:onnxrt-kt:0.1.0-SNAPSHOT`.
