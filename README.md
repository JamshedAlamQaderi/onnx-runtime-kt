# onnxrt-kt

Kotlin Multiplatform bindings for [ONNX Runtime](https://onnxruntime.ai). You load and run ONNX models from common
Kotlin code, and each platform uses Microsoft's official runtime underneath.

| Platform        | Backend                                                        | Version |
|-----------------|----------------------------------------------------------------|---------|
| JVM (desktop)   | `com.microsoft.onnxruntime:onnxruntime` (Java API)             | 1.31.1  |
| Android         | `com.microsoft.onnxruntime:onnxruntime-android` (Java API)     | 1.31.1  |
| iOS (arm64/sim) | `onnxruntime-swift-package-manager` (Objective-C API, SwiftPM) | 1.24.2¹ |

¹ Microsoft's newest Swift package release lags behind the Maven artifacts.

The project is built with the [Kotlin Toolchain](https://kotlin-toolchain.org/latest/) (Kotlin 2.4.20,
Compose Multiplatform 1.12.1). The `./kotlin` wrapper script downloads the pinned toolchain on first use.

## Usage

```kotlin
import com.jamshedalamqaderi.kmp.onnxrtkt.*

val session = OnnxRuntime.createSession(modelBytes)            // or createSession("/path/to/model.onnx")
session.use {
    val input = Tensor.ofFloats(pixels, shape = longArrayOf(1, 1, 28, 28))
    val outputs = it.run(mapOf(it.inputNames.first() to input))  // Map<String, Tensor>
    val logits = outputs.getValue(it.outputNames.first()).toFloatArray()
}
```

- `Tensor` is an immutable tensor that does not depend on any platform. It supports `FLOAT`, `DOUBLE`, `INT8`,
  `UINT8`, `INT32`, `INT64`, `BOOL` and `STRING`. iOS does not support `DOUBLE` or `BOOL`, because the ONNX Runtime
  Objective-C API does not expose them.
- `SessionOptions` sets thread counts, the graph optimization level, the log level and execution providers
  (`NNAPI`, `XNNPACK`, `CORE_ML`).
- Errors are reported as `OrtException` on every platform.
- `session.runSingle(tensor)` is a shortcut for models with exactly one input.

## Project layout

```
onnxrt-kt/            the library (kmp/lib)
  src/                common API: OnnxRuntime, InferenceSession, Tensor, SessionOptions
  src@jvmAndAndroid/  implementation on the ai.onnxruntime Java API (shared by JVM and Android)
  src@ios/            implementation on the ONNX Runtime Objective-C API, imported via SwiftPM
sample/               "draw a digit" app built with the MNIST model from the ONNX Model Zoo
  shared/             Compose UI, DigitClassifier (inference) and DigitRasterizer (preprocessing)
  androidApp/  desktopApp/  iosApp/
tools/                generate_linear_model.py (builds the tiny test model in onnxrt-kt/testResources)
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
./kotlin run -m iosApp       # iOS simulator (macOS with Xcode only)
```

## Build and test

```bash
./kotlin build
./kotlin test -m onnxrt-kt --platform=jvm   # library tests, including real inference
./kotlin test -m shared  --platform=jvm   # sample tests (model zoo reference output, drawn digits)
```

To publish the library to your local Maven repository, run `./kotlin publish mavenLocal -m onnxrt-kt`.
Coordinates: `com.jamshedalamqaderi.kmp:onnxrt-kt:0.1.0-SNAPSHOT`.
