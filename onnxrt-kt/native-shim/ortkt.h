/*
 * ortkt: a small C layer between Kotlin/Native and the ONNX Runtime C API.
 *
 * Kotlin binds to this header only (see build-plugins/onnxruntime-native), so every native
 * target gets identical bindings that the Kotlin/Native commonizer can share in `src@native`.
 * The ONNX Runtime headers are used only inside ortkt.c.
 *
 * Error convention: functions returning `char*` return NULL on success, or an error message
 * that the caller must release with ortkt_free_string().
 */
#ifndef ORTKT_H
#define ORTKT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ortkt_session ortkt_session;
typedef struct ortkt_value ortkt_value;

/* ONNX tensor element types (same values as ONNXTensorElementDataType). */
#define ORTKT_FLOAT 1
#define ORTKT_UINT8 2
#define ORTKT_INT8 3
#define ORTKT_INT32 6
#define ORTKT_INT64 7
#define ORTKT_STRING 8
#define ORTKT_BOOL 9
#define ORTKT_DOUBLE 11

#define ORTKT_PROVIDER_COREML 1
#define ORTKT_PROVIDER_XNNPACK 2

typedef struct ortkt_options {
    int32_t intra_op_threads;   /* 0 = ONNX Runtime default */
    int32_t inter_op_threads;   /* 0 = ONNX Runtime default */
    int32_t optimization_level; /* 0 none, 1 basic, 2 extended, 99 all */
    int32_t log_level;          /* 0 verbose .. 4 fatal */
    int32_t provider_count;
    const int32_t* providers;   /* ORTKT_PROVIDER_* in priority order */
} ortkt_options;

/*
 * Makes ONNX Runtime ready to use. No-op on Apple platforms, where it is linked statically.
 *
 * On Linux and Windows the shared library is loaded from, in order:
 *   1. `path`, when not NULL;
 *   2. the ONNXRUNTIME_LIBRARY_PATH environment variable;
 *   3. the official library embedded in this build, unpacked once into a per-user cache
 *      ($XDG_CACHE_HOME or ~/.cache on Linux, %LOCALAPPDATA% on Windows; under onnxrt-kt/).
 * Calling it again after a successful load does nothing.
 */
char* ortkt_load_library(const char* path);

/* 1 when ONNX Runtime is ready to use (statically linked, or loaded). */
int32_t ortkt_is_loaded(void);

/* 1 when ONNX Runtime is linked statically into this binary. */
int32_t ortkt_is_static(void);

const char* ortkt_version(void);

void ortkt_free_string(char* s);

char* ortkt_create_session(const void* model, size_t model_size, const ortkt_options* options, ortkt_session** out);
void ortkt_release_session(ortkt_session* session);

size_t ortkt_input_count(const ortkt_session* session);
size_t ortkt_output_count(const ortkt_session* session);
const char* ortkt_input_name(const ortkt_session* session, size_t index);
const char* ortkt_output_name(const ortkt_session* session, size_t index);

/* Creates a tensor that owns a copy of `data` (`byte_size` bytes). */
char* ortkt_create_tensor(int32_t element_type, const void* data, size_t byte_size,
                          const int64_t* shape, size_t rank, ortkt_value** out);
char* ortkt_create_string_tensor(const char* const* strings, size_t count,
                                 const int64_t* shape, size_t rank, ortkt_value** out);
void ortkt_release_value(ortkt_value* value);

/* `outputs` must have room for `output_count` values; each one must be released by the caller. */
char* ortkt_run(ortkt_session* session,
                const char* const* input_names, ortkt_value* const* inputs, size_t input_count,
                const char* const* output_names, size_t output_count, ortkt_value** outputs);

/* Fails for values that are not tensors (sequences, maps, ...). */
char* ortkt_tensor_info(const ortkt_value* value, int32_t* element_type, size_t* rank, size_t* element_count);
/* `shape` must have room for `rank` dimensions. */
char* ortkt_tensor_shape(const ortkt_value* value, int64_t* shape, size_t rank);
/* Pointer to the tensor's data, valid until the value is released. Not for string tensors. */
char* ortkt_tensor_data(ortkt_value* value, const void** data);

/* Total UTF-8 byte length of all strings in a string tensor. */
char* ortkt_string_tensor_length(const ortkt_value* value, size_t* total_bytes);
/* Copies the strings back to back into `buffer` and each string's start into `offsets`. */
char* ortkt_string_tensor_content(const ortkt_value* value, char* buffer, size_t buffer_size,
                                  size_t* offsets, size_t count);

#ifdef __cplusplus
}
#endif

#endif /* ORTKT_H */
