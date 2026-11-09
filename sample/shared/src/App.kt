package com.jamshedalamqaderi.kmp.onnxrtkt.sample

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeContentPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.jamshedalamqaderi.kmp.onnxrtkt.OnnxRuntime
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.jetbrains.compose.reload.DevelopmentEntryPoint
import kotlin.math.roundToInt

@Composable
@Preview
@DevelopmentEntryPoint
fun App() {
    MaterialTheme {
        Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            DigitRecognizerScreen()
        }
    }
}

private sealed interface ModelState {
    data object Loading : ModelState
    class Ready(val classifier: DigitClassifier) : ModelState
    class Failed(val error: Throwable) : ModelState
}

@Composable
fun DigitRecognizerScreen() {
    var model by remember { mutableStateOf<ModelState>(ModelState.Loading) }
    val strokes = remember { mutableStateListOf<List<Offset>>() }
    val currentStroke = remember { mutableStateListOf<Offset>() }
    var modelInput by remember { mutableStateOf<FloatArray?>(null) }
    var prediction by remember { mutableStateOf<Prediction?>(null) }
    val scope = rememberCoroutineScope()

    // 1. Load the ONNX model once, off the main thread.
    LaunchedEffect(Unit) {
        model = try {
            ModelState.Ready(withContext(Dispatchers.Default) { DigitClassifier.load() })
        } catch (t: Throwable) {
            ModelState.Failed(t)
        }
    }
    DisposableEffect(Unit) { onDispose { (model as? ModelState.Ready)?.classifier?.close() } }

    // 2. Run inference whenever a stroke is finished.
    fun classify() {
        val classifier = (model as? ModelState.Ready)?.classifier ?: return
        val snapshot = strokes.toList()
        scope.launch {
            val (pixels, result) = withContext(Dispatchers.Default) {
                val pixels = DigitRasterizer.rasterize(snapshot)
                pixels to pixels?.let(classifier::classify)
            }
            modelInput = pixels
            prediction = result
        }
    }

    Column(
        modifier = Modifier
            .safeContentPadding()
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Handwritten digit recognizer", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
        Text(
            "MNIST · ONNX Runtime ${remember { runCatching { OnnxRuntime.version }.getOrDefault("?") }} · ${getPlatform().name}",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        when (val m = model) {
            ModelState.Loading -> CircularProgressIndicator()
            is ModelState.Failed -> Text("Could not load model: ${m.error.message}", color = MaterialTheme.colorScheme.error)
            is ModelState.Ready -> Text("Draw a digit (0–9) in the box", style = MaterialTheme.typography.bodyMedium)
        }

        DrawingPad(
            strokes = strokes,
            currentStroke = currentStroke,
            onStrokeFinished = {
                if (currentStroke.isNotEmpty()) strokes += currentStroke.toList()
                currentStroke.clear()
                classify()
            },
            modifier = Modifier.widthIn(max = 360.dp).fillMaxWidth(),
        )

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = {
                strokes.clear(); currentStroke.clear(); prediction = null; modelInput = null
            }) { Text("Clear") }
            Button(
                onClick = { if (strokes.isNotEmpty()) { strokes.removeAt(strokes.lastIndex); classify() } },
                enabled = strokes.isNotEmpty(),
            ) { Text("Undo") }
        }

        prediction?.let { p ->
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("${p.digit}", fontSize = 64.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary)
                    Text("${(p.confidence * 100).roundToInt()}% confident", style = MaterialTheme.typography.bodySmall)
                }
                modelInput?.let { ModelInputPreview(it) }
            }
            ProbabilityBars(p, Modifier.widthIn(max = 360.dp).fillMaxWidth())
        }
    }
}

@Composable
private fun DrawingPad(
    strokes: List<List<Offset>>,
    currentStroke: MutableList<Offset>,
    onStrokeFinished: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier
            .aspectRatio(1f)
            .clip(RoundedCornerShape(16.dp))
            .background(Color.Black)
            .border(2.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(16.dp))
            .pointerInput(Unit) {
                detectDragGestures(
                    onDragStart = { currentStroke.clear(); currentStroke += it },
                    onDragEnd = { onStrokeFinished() },
                    onDragCancel = { onStrokeFinished() },
                ) { change, _ -> currentStroke += change.position }
            }
            .pointerInput(Unit) {
                detectTapGestures { currentStroke.clear(); currentStroke += it; onStrokeFinished() }
            },
    ) {
        Canvas(Modifier.fillMaxSize()) {
            val width = size.minDimension / 14f
            for (stroke in strokes + listOf(currentStroke.toList())) {
                if (stroke.isEmpty()) continue
                if (stroke.size == 1) {
                    drawCircle(Color.White, radius = width / 2, center = stroke[0])
                    continue
                }
                val path = Path().apply {
                    moveTo(stroke[0].x, stroke[0].y)
                    for (i in 1 until stroke.size) lineTo(stroke[i].x, stroke[i].y)
                }
                drawPath(path, Color.White, style = Stroke(width = width, cap = StrokeCap.Round, join = StrokeJoin.Round))
            }
        }
    }
}

/** Shows the exact 28×28 tensor fed to the model. */
@Composable
private fun ModelInputPreview(pixels: FloatArray) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Canvas(Modifier.size(84.dp).border(1.dp, MaterialTheme.colorScheme.outline)) {
            val n = DigitClassifier.IMAGE_SIZE
            val cell = size.width / n
            drawRect(Color.Black)
            for (y in 0 until n) for (x in 0 until n) {
                val v = pixels[y * n + x]
                if (v > 0f) drawRect(Color(v, v, v), topLeft = Offset(x * cell, y * cell), size = Size(cell, cell))
            }
        }
        Text("model input", style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun ProbabilityBars(prediction: Prediction, modifier: Modifier = Modifier) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(4.dp)) {
        prediction.probabilities.forEachIndexed { digit, p ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("$digit", Modifier.widthIn(min = 20.dp), style = MaterialTheme.typography.labelLarge)
                Box(
                    Modifier
                        .weight(1f)
                        .height(14.dp)
                        .clip(RoundedCornerShape(7.dp))
                        .background(MaterialTheme.colorScheme.surfaceVariant),
                ) {
                    Box(
                        Modifier
                            .fillMaxHeight()
                            .fillMaxWidth(p.coerceIn(0f, 1f))
                            .background(
                                if (digit == prediction.digit) MaterialTheme.colorScheme.primary
                                else MaterialTheme.colorScheme.secondary.copy(alpha = 0.5f),
                            ),
                    )
                }
                Spacer(Modifier.size(8.dp))
                Text("${(p * 100).roundToInt()}%", Modifier.widthIn(min = 40.dp), style = MaterialTheme.typography.labelSmall)
            }
        }
    }
}
