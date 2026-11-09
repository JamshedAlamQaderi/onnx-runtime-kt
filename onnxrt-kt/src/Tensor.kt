package com.jamshedalamqaderi.kmp.onnxrtkt

/**
 * An immutable, platform-independent n-dimensional tensor.
 *
 * Data is stored in row-major order. Use the `of*` factories to create input tensors
 * and the `to*Array` accessors to read results returned by [InferenceSession.run].
 *
 * ```kotlin
 * val input = Tensor.ofFloats(floatArrayOf(1f, 2f, 3f), shape = longArrayOf(3, 1))
 * ```
 */
public class Tensor private constructor(
    public val type: TensorElementType,
    shape: LongArray,
    internal val data: Any,
) {
    private val dims: LongArray = shape.copyOf()

    /** Dimensions of this tensor. An empty array means a scalar. Returns a defensive copy. */
    public val shape: LongArray get() = dims.copyOf()

    /** Number of elements (the product of all dimensions). */
    public val elementCount: Long = dims.fold(1L) { acc, d -> acc * d }

    init {
        require(dims.all { it >= 0 }) { "Tensor dimensions must be non-negative: ${dims.contentToString()}" }
        val size = dataSize(data)
        require(size.toLong() == elementCount) {
            "Shape ${dims.contentToString()} needs $elementCount elements, but $size were provided"
        }
    }

    public fun toFloatArray(): FloatArray = (expect(TensorElementType.FLOAT) as FloatArray).copyOf()
    public fun toDoubleArray(): DoubleArray = (expect(TensorElementType.DOUBLE) as DoubleArray).copyOf()
    public fun toIntArray(): IntArray = (expect(TensorElementType.INT32) as IntArray).copyOf()
    public fun toLongArray(): LongArray = (expect(TensorElementType.INT64) as LongArray).copyOf()
    public fun toBooleanArray(): BooleanArray = (expect(TensorElementType.BOOL) as BooleanArray).copyOf()

    @Suppress("UNCHECKED_CAST")
    public fun toStringArray(): Array<String> = (expect(TensorElementType.STRING) as Array<String>).copyOf()

    /** Raw bytes of an [TensorElementType.INT8] tensor. */
    public fun toByteArray(): ByteArray = (expect(TensorElementType.INT8) as ByteArray).copyOf()

    /** Values of a [TensorElementType.UINT8] tensor. */
    @OptIn(ExperimentalUnsignedTypes::class)
    public fun toUByteArray(): UByteArray = (expect(TensorElementType.UINT8) as ByteArray).copyOf().asUByteArray()

    private fun expect(expected: TensorElementType): Any {
        if (type != expected) throw OrtException("Tensor has element type $type, not $expected")
        return data
    }

    override fun toString(): String = "Tensor(type=$type, shape=${dims.contentToString()})"

    public companion object {
        public fun ofFloats(data: FloatArray, shape: LongArray = longArrayOf(data.size.toLong())): Tensor =
            Tensor(TensorElementType.FLOAT, shape, data.copyOf())

        public fun ofDoubles(data: DoubleArray, shape: LongArray = longArrayOf(data.size.toLong())): Tensor =
            Tensor(TensorElementType.DOUBLE, shape, data.copyOf())

        public fun ofInts(data: IntArray, shape: LongArray = longArrayOf(data.size.toLong())): Tensor =
            Tensor(TensorElementType.INT32, shape, data.copyOf())

        public fun ofLongs(data: LongArray, shape: LongArray = longArrayOf(data.size.toLong())): Tensor =
            Tensor(TensorElementType.INT64, shape, data.copyOf())

        public fun ofBytes(data: ByteArray, shape: LongArray = longArrayOf(data.size.toLong())): Tensor =
            Tensor(TensorElementType.INT8, shape, data.copyOf())

        @OptIn(ExperimentalUnsignedTypes::class)
        public fun ofUBytes(data: UByteArray, shape: LongArray = longArrayOf(data.size.toLong())): Tensor =
            Tensor(TensorElementType.UINT8, shape, data.asByteArray().copyOf())

        public fun ofBooleans(data: BooleanArray, shape: LongArray = longArrayOf(data.size.toLong())): Tensor =
            Tensor(TensorElementType.BOOL, shape, data.copyOf())

        public fun ofStrings(data: Array<String>, shape: LongArray = longArrayOf(data.size.toLong())): Tensor =
            Tensor(TensorElementType.STRING, shape, data.copyOf())

        /**
         * Platform implementations use this to wrap output data without an extra copy.
         * For [TensorElementType.UINT8], [data] must be a [ByteArray].
         */
        internal fun wrap(type: TensorElementType, shape: LongArray, data: Any): Tensor = Tensor(type, shape, data)

        private fun dataSize(data: Any): Int = when (data) {
            is FloatArray -> data.size
            is DoubleArray -> data.size
            is IntArray -> data.size
            is LongArray -> data.size
            is ByteArray -> data.size
            is BooleanArray -> data.size
            is Array<*> -> data.size
            else -> error("Unsupported tensor storage: ${data::class}")
        }
    }
}
