/*
 * Implementation of ortkt.h on top of the ONNX Runtime C API.
 *
 * Apple targets link ONNX Runtime statically (ORTKT_STATIC is defined by the build plugin).
 * Linux and Windows load the official shared library at runtime and look up the single
 * exported entry point, OrtGetApiBase; every other call goes through the OrtApi table.
 */
#include "ortkt.h"
#include "onnxruntime_c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#include <pthread.h>
#endif

typedef const OrtApiBase*(ORT_API_CALL* ortkt_get_api_base_fn)(void);

static const OrtApiBase* g_api_base = NULL;
static const OrtApi* g_api = NULL;
static OrtEnv* g_env = NULL;
static char* g_env_error = NULL;

struct ortkt_session {
    OrtSession* session;
    size_t input_count;
    size_t output_count;
    char** input_names;
    char** output_names;
};

struct ortkt_value {
    OrtValue* value;
};

/* ---------------------------------------------------------------- helpers */

static char* dup_string(const char* s) {
    size_t n = strlen(s) + 1;
    char* copy = (char*)malloc(n);
    if (copy) memcpy(copy, s, n);
    return copy;
}

static char* format_error(const char* prefix, const char* detail) {
    size_t n = strlen(prefix) + strlen(detail) + 3;
    char* msg = (char*)malloc(n);
    if (msg) snprintf(msg, n, "%s: %s", prefix, detail);
    return msg;
}

/* Converts an OrtStatus into an error string (and releases it). NULL means success. */
static char* check(OrtStatus* status) {
    if (status == NULL) return NULL;
    char* msg = dup_string(g_api->GetErrorMessage(status));
    g_api->ReleaseStatus(status);
    return msg;
}

#define CHECK(expr)                       \
    do {                                  \
        char* ortkt_err_ = check(expr);   \
        if (ortkt_err_) return ortkt_err_; \
    } while (0)

/* -------------------------------------------------------- library loading */

#ifdef ORTKT_STATIC

int32_t ortkt_is_static(void) { return 1; }

char* ortkt_load_library(const char* path) {
    (void)path;
    return NULL;
}

static const OrtApiBase* api_base(void) {
    if (g_api_base == NULL) g_api_base = OrtGetApiBase();
    return g_api_base;
}

#else

/* Appended by the build plugin: the official library bytes, via #embed. */
extern const unsigned char ortkt_library_bytes[];
extern const size_t ortkt_library_size;

int32_t ortkt_is_static(void) { return 0; }

static const OrtApiBase* api_base(void) { return g_api_base; }

/* Cache directory name: FNV-1a hash of the embedded library, so each build unpacks its own copy. */
static void library_hash(char out[17]) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < ortkt_library_size; i++) {
        h ^= ortkt_library_bytes[i];
        h *= 1099511628211ULL;
    }
    snprintf(out, 17, "%016llx", (unsigned long long)h);
}

#if defined(_WIN32)

#define LIBRARY_FILE_NAME L"onnxruntime.dll"

static char* to_utf8(const wchar_t* w) {
    int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, NULL, 0, NULL, NULL);
    char* s = (char*)malloc(n > 0 ? (size_t)n : 1);
    if (s) {
        if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w, -1, s, n, NULL, NULL);
        else s[0] = 0;
    }
    return s;
}

static wchar_t* to_wide(const char* s) {
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, NULL, 0);
    if (n <= 0) return NULL;
    wchar_t* w = (wchar_t*)malloc(sizeof(wchar_t) * (size_t)n);
    if (w) MultiByteToWideChar(CP_UTF8, 0, s, -1, w, n);
    return w;
}

static char* win_error(const char* what, const wchar_t* path) {
    char* p = to_utf8(path);
    char buf[512];
    snprintf(buf, sizeof(buf), "%s %s failed with Windows error %lu", what, p ? p : "?", (unsigned long)GetLastError());
    free(p);
    return dup_string(buf);
}

static char* load_from(const wchar_t* path) {
    /* Resolve the DLL's own dependencies from its directory first. */
    HMODULE module = LoadLibraryExW(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (module == NULL && GetLastError() == ERROR_INVALID_PARAMETER) module = LoadLibraryW(path); /* relative path */
    if (module == NULL) {
        char* p = to_utf8(path);
        char buf[700];
        snprintf(buf, sizeof(buf),
                 "Could not load %s (Windows error %lu). onnxruntime.dll needs the Microsoft Visual C++ "
                 "Redistributable (MSVCP140.dll, VCRUNTIME140.dll, VCRUNTIME140_1.dll).",
                 p ? p : "?", (unsigned long)GetLastError());
        free(p);
        return dup_string(buf);
    }
    ortkt_get_api_base_fn get = (ortkt_get_api_base_fn)(void*)GetProcAddress(module, "OrtGetApiBase");
    if (get == NULL) return dup_string("OrtGetApiBase not found: not an ONNX Runtime library");
    g_api_base = get();
    return NULL;
}

static char* unpack_embedded(wchar_t* out, size_t out_len) {
    wchar_t base[MAX_PATH];
    DWORD n = GetEnvironmentVariableW(L"LOCALAPPDATA", base, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) n = GetTempPathW(MAX_PATH, base);
    if (n == 0 || n >= MAX_PATH) return dup_string("Could not find LOCALAPPDATA or TEMP to unpack onnxruntime.dll");
    char hash[17];
    library_hash(hash);
    wchar_t dir[MAX_PATH];
    _snwprintf(dir, MAX_PATH, L"%ls\\onnxrt-kt", base);
    CreateDirectoryW(dir, NULL);
    _snwprintf(dir, MAX_PATH, L"%ls\\onnxrt-kt\\%hs", base, hash);
    if (!CreateDirectoryW(dir, NULL) && GetLastError() != ERROR_ALREADY_EXISTS) return win_error("Creating", dir);
    _snwprintf(out, out_len, L"%ls\\%ls", dir, LIBRARY_FILE_NAME);

    WIN32_FILE_ATTRIBUTE_DATA attrs;
    if (GetFileAttributesExW(out, GetFileExInfoStandard, &attrs) &&
        (((uint64_t)attrs.nFileSizeHigh << 32) | attrs.nFileSizeLow) == ortkt_library_size) {
        return NULL; /* already unpacked */
    }
    wchar_t tmp[MAX_PATH];
    _snwprintf(tmp, MAX_PATH, L"%ls.%lu.tmp", out, (unsigned long)GetCurrentProcessId());
    HANDLE f = CreateFileW(tmp, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) return win_error("Creating", tmp);
    size_t written = 0;
    while (written < ortkt_library_size) {
        DWORD chunk = (DWORD)((ortkt_library_size - written) > (1u << 30) ? (1u << 30) : (ortkt_library_size - written));
        DWORD did = 0;
        if (!WriteFile(f, ortkt_library_bytes + written, chunk, &did, NULL) || did == 0) {
            CloseHandle(f);
            DeleteFileW(tmp);
            return win_error("Writing", tmp);
        }
        written += did;
    }
    CloseHandle(f);
    if (!MoveFileExW(tmp, out, MOVEFILE_REPLACE_EXISTING)) {
        char* err = NULL;
        /* Another process may have just put the same file in place and loaded it. */
        if (GetFileAttributesExW(out, GetFileExInfoStandard, &attrs) &&
            (((uint64_t)attrs.nFileSizeHigh << 32) | attrs.nFileSizeLow) == ortkt_library_size) {
            err = NULL;
        } else {
            err = win_error("Moving", tmp);
        }
        DeleteFileW(tmp);
        return err;
    }
    return NULL;
}

static char* load_library_impl(const char* path) {
    if (path == NULL || path[0] == 0) {
        wchar_t env[MAX_PATH];
        DWORD n = GetEnvironmentVariableW(L"ONNXRUNTIME_LIBRARY_PATH", env, MAX_PATH);
        if (n > 0 && n < MAX_PATH) return load_from(env);
        wchar_t unpacked[MAX_PATH];
        char* err = unpack_embedded(unpacked, MAX_PATH);
        return err ? err : load_from(unpacked);
    }
    wchar_t* w = to_wide(path);
    if (!w) return format_error("Invalid library path", path);
    char* err = load_from(w);
    free(w);
    return err;
}

static CRITICAL_SECTION* load_lock(void) {
    static CRITICAL_SECTION cs;
    static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
    BOOL pending = FALSE;
    InitOnceBeginInitialize(&once, 0, &pending, NULL);
    if (pending) {
        InitializeCriticalSection(&cs);
        InitOnceComplete(&once, 0, NULL);
    }
    return &cs;
}

char* ortkt_load_library(const char* path) {
    CRITICAL_SECTION* cs = load_lock();
    EnterCriticalSection(cs);
    char* err = g_api_base ? NULL : load_library_impl(path);
    LeaveCriticalSection(cs);
    return err;
}

#else /* Linux */

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#define LIBRARY_FILE_NAME "libonnxruntime.so.1"

static char* load_from(const char* path) {
    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (handle == NULL) {
        const char* detail = dlerror();
        return format_error("Could not load ONNX Runtime", detail ? detail : path);
    }
    ortkt_get_api_base_fn get = (ortkt_get_api_base_fn)dlsym(handle, "OrtGetApiBase");
    if (get == NULL) return format_error(path, "OrtGetApiBase not found: not an ONNX Runtime library");
    g_api_base = get();
    return NULL;
}

static char* errno_error(const char* what, const char* path) {
    char buf[1200];
    snprintf(buf, sizeof(buf), "%s %s failed: %s", what, path, strerror(errno));
    return dup_string(buf);
}

static char* unpack_embedded(char* out, size_t out_len) {
    char base[1024];
    const char* xdg = getenv("XDG_CACHE_HOME");
    const char* home = getenv("HOME");
    if (xdg && xdg[0] == '/') snprintf(base, sizeof(base), "%s", xdg);
    else if (home && home[0]) snprintf(base, sizeof(base), "%s/.cache", home);
    else snprintf(base, sizeof(base), "/tmp");
    char hash[17];
    library_hash(hash);
    char dir[1100];
    mkdir(base, 0700);
    snprintf(dir, sizeof(dir), "%s/onnxrt-kt", base);
    mkdir(dir, 0700);
    snprintf(dir, sizeof(dir), "%s/onnxrt-kt/%s", base, hash);
    if (mkdir(dir, 0700) != 0 && errno != EEXIST) return errno_error("Creating", dir);
    snprintf(out, out_len, "%s/%s", dir, LIBRARY_FILE_NAME);

    struct stat st;
    if (stat(out, &st) == 0 && (size_t)st.st_size == ortkt_library_size) return NULL; /* already unpacked */

    char tmp[1300];
    snprintf(tmp, sizeof(tmp), "%s.%ld.tmp", out, (long)getpid());
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0700);
    if (fd < 0) return errno_error("Creating", tmp);
    size_t written = 0;
    while (written < ortkt_library_size) {
        ssize_t n = write(fd, ortkt_library_bytes + written, ortkt_library_size - written);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            char* err = errno_error("Writing", tmp);
            close(fd);
            unlink(tmp);
            return err;
        }
        written += (size_t)n;
    }
    if (close(fd) != 0) {
        unlink(tmp);
        return errno_error("Writing", tmp);
    }
    if (rename(tmp, out) != 0) {
        char* err = errno_error("Moving", tmp);
        unlink(tmp);
        return err;
    }
    return NULL;
}

static char* load_library_impl(const char* path) {
    if (path != NULL && path[0] != 0) return load_from(path);
    const char* env = getenv("ONNXRUNTIME_LIBRARY_PATH");
    if (env != NULL && env[0] != 0) return load_from(env);
    char unpacked[1400];
    char* err = unpack_embedded(unpacked, sizeof(unpacked));
    return err ? err : load_from(unpacked);
}

static pthread_mutex_t g_load_lock = PTHREAD_MUTEX_INITIALIZER;

char* ortkt_load_library(const char* path) {
    pthread_mutex_lock(&g_load_lock);
    char* err = g_api_base ? NULL : load_library_impl(path);
    pthread_mutex_unlock(&g_load_lock);
    return err;
}

#endif /* _WIN32 */

#endif /* ORTKT_STATIC */

int32_t ortkt_is_loaded(void) { return api_base() != NULL; }

const char* ortkt_version(void) {
    const OrtApiBase* base = api_base();
    return base ? base->GetVersionString() : NULL;
}

/* Creates the OrtApi table and the process-wide OrtEnv exactly once. */
static void init_env(void) {
    const OrtApiBase* base = api_base();
    if (base == NULL) {
        g_env_error = dup_string("ONNX Runtime library is not loaded");
        return;
    }
    g_api = base->GetApi(ORT_API_VERSION);
    if (g_api == NULL) {
        char detail[128];
        snprintf(detail, sizeof(detail), "the library (version %s) does not support C API version %d",
                 base->GetVersionString(), ORT_API_VERSION);
        g_env_error = format_error("Incompatible ONNX Runtime", detail);
        return;
    }
    g_env_error = check(g_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnxrt-kt", &g_env));
}

#if defined(_WIN32)
static INIT_ONCE g_init_once = INIT_ONCE_STATIC_INIT;
static BOOL CALLBACK init_env_once(PINIT_ONCE once, PVOID param, PVOID* context) {
    (void)once; (void)param; (void)context;
    init_env();
    return TRUE;
}
static char* ensure_env(void) {
    InitOnceExecuteOnce(&g_init_once, init_env_once, NULL, NULL);
    return g_env_error ? dup_string(g_env_error) : NULL;
}
#else
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;
static char* ensure_env(void) {
    if (api_base() == NULL) return dup_string("ONNX Runtime library is not loaded");
    pthread_once(&g_init_once, init_env);
    return g_env_error ? dup_string(g_env_error) : NULL;
}
#endif

void ortkt_free_string(char* s) { free(s); }

/* ---------------------------------------------------------------- sessions */

static char* apply_options(OrtSessionOptions* so, const ortkt_options* o) {
    if (o->intra_op_threads > 0) CHECK(g_api->SetIntraOpNumThreads(so, o->intra_op_threads));
    if (o->inter_op_threads > 0) CHECK(g_api->SetInterOpNumThreads(so, o->inter_op_threads));
    CHECK(g_api->SetSessionGraphOptimizationLevel(so, (GraphOptimizationLevel)o->optimization_level));
    CHECK(g_api->SetSessionLogSeverityLevel(so, o->log_level));
    for (int32_t i = 0; i < o->provider_count; i++) {
        switch (o->providers[i]) {
            case ORTKT_PROVIDER_COREML:
                CHECK(g_api->SessionOptionsAppendExecutionProvider(so, "CoreML", NULL, NULL, 0));
                break;
            case ORTKT_PROVIDER_XNNPACK:
                CHECK(g_api->SessionOptionsAppendExecutionProvider(so, "XNNPACK", NULL, NULL, 0));
                break;
            default:
                return dup_string("Unknown execution provider");
        }
    }
    return NULL;
}

static char* read_names(OrtSession* session, size_t count, int outputs, char*** out) {
    OrtAllocator* allocator = NULL;
    CHECK(g_api->GetAllocatorWithDefaultOptions(&allocator));
    char** names = (char**)calloc(count ? count : 1, sizeof(char*));
    if (!names) return dup_string("Out of memory");
    *out = names;
    for (size_t i = 0; i < count; i++) {
        char* name = NULL;
        OrtStatus* status = outputs ? g_api->SessionGetOutputName(session, i, allocator, &name)
                                    : g_api->SessionGetInputName(session, i, allocator, &name);
        CHECK(status);
        names[i] = dup_string(name);
        allocator->Free(allocator, name);
    }
    return NULL;
}

void ortkt_release_session(ortkt_session* s) {
    if (s == NULL) return;
    if (s->input_names) {
        for (size_t i = 0; i < s->input_count; i++) free(s->input_names[i]);
        free(s->input_names);
    }
    if (s->output_names) {
        for (size_t i = 0; i < s->output_count; i++) free(s->output_names[i]);
        free(s->output_names);
    }
    if (s->session) g_api->ReleaseSession(s->session);
    free(s);
}

char* ortkt_create_session(const void* model, size_t model_size, const ortkt_options* options, ortkt_session** out) {
    *out = NULL;
    char* err = ensure_env();
    if (err) return err;

    OrtSessionOptions* so = NULL;
    CHECK(g_api->CreateSessionOptions(&so));
    err = apply_options(so, options);
    if (err) {
        g_api->ReleaseSessionOptions(so);
        return err;
    }

    ortkt_session* s = (ortkt_session*)calloc(1, sizeof(ortkt_session));
    if (!s) {
        g_api->ReleaseSessionOptions(so);
        return dup_string("Out of memory");
    }
    err = check(g_api->CreateSessionFromArray(g_env, model, model_size, so, &s->session));
    g_api->ReleaseSessionOptions(so);
    if (!err) err = check(g_api->SessionGetInputCount(s->session, &s->input_count));
    if (!err) err = check(g_api->SessionGetOutputCount(s->session, &s->output_count));
    if (!err) err = read_names(s->session, s->input_count, 0, &s->input_names);
    if (!err) err = read_names(s->session, s->output_count, 1, &s->output_names);
    if (err) {
        ortkt_release_session(s);
        return err;
    }
    *out = s;
    return NULL;
}

size_t ortkt_input_count(const ortkt_session* s) { return s->input_count; }
size_t ortkt_output_count(const ortkt_session* s) { return s->output_count; }
const char* ortkt_input_name(const ortkt_session* s, size_t i) { return s->input_names[i]; }
const char* ortkt_output_name(const ortkt_session* s, size_t i) { return s->output_names[i]; }

/* ----------------------------------------------------------------- tensors */

static char* wrap_value(OrtValue* v, ortkt_value** out) {
    ortkt_value* w = (ortkt_value*)malloc(sizeof(ortkt_value));
    if (!w) {
        g_api->ReleaseValue(v);
        return dup_string("Out of memory");
    }
    w->value = v;
    *out = w;
    return NULL;
}

char* ortkt_create_tensor(int32_t element_type, const void* data, size_t byte_size,
                          const int64_t* shape, size_t rank, ortkt_value** out) {
    *out = NULL;
    char* err = ensure_env();
    if (err) return err;
    OrtAllocator* allocator = NULL;
    CHECK(g_api->GetAllocatorWithDefaultOptions(&allocator));
    OrtValue* v = NULL;
    CHECK(g_api->CreateTensorAsOrtValue(allocator, shape, rank, (ONNXTensorElementDataType)element_type, &v));
    if (byte_size > 0) {
        void* dst = NULL;
        err = check(g_api->GetTensorMutableData(v, &dst));
        if (err) {
            g_api->ReleaseValue(v);
            return err;
        }
        memcpy(dst, data, byte_size);
    }
    return wrap_value(v, out);
}

char* ortkt_create_string_tensor(const char* const* strings, size_t count,
                                 const int64_t* shape, size_t rank, ortkt_value** out) {
    *out = NULL;
    char* err = ensure_env();
    if (err) return err;
    OrtAllocator* allocator = NULL;
    CHECK(g_api->GetAllocatorWithDefaultOptions(&allocator));
    OrtValue* v = NULL;
    CHECK(g_api->CreateTensorAsOrtValue(allocator, shape, rank, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &v));
    if (count > 0) {
        err = check(g_api->FillStringTensor(v, strings, count));
        if (err) {
            g_api->ReleaseValue(v);
            return err;
        }
    }
    return wrap_value(v, out);
}

void ortkt_release_value(ortkt_value* value) {
    if (value == NULL) return;
    if (value->value) g_api->ReleaseValue(value->value);
    free(value);
}

char* ortkt_run(ortkt_session* s,
                const char* const* input_names, ortkt_value* const* inputs, size_t input_count,
                const char* const* output_names, size_t output_count, ortkt_value** outputs) {
    const OrtValue** in = (const OrtValue**)calloc(input_count ? input_count : 1, sizeof(OrtValue*));
    OrtValue** out = (OrtValue**)calloc(output_count ? output_count : 1, sizeof(OrtValue*));
    if (!in || !out) {
        free((void*)in);
        free(out);
        return dup_string("Out of memory");
    }
    for (size_t i = 0; i < input_count; i++) in[i] = inputs[i]->value;
    char* err = check(g_api->Run(s->session, NULL, input_names, in, input_count, output_names, output_count, out));
    free((void*)in);
    for (size_t i = 0; i < output_count; i++) outputs[i] = NULL;
    if (!err) {
        for (size_t i = 0; i < output_count && !err; i++) {
            err = wrap_value(out[i], &outputs[i]);
            out[i] = NULL;
        }
    }
    if (err) {
        for (size_t i = 0; i < output_count; i++) {
            if (out[i]) g_api->ReleaseValue(out[i]);
            ortkt_release_value(outputs[i]);
            outputs[i] = NULL;
        }
    }
    free(out);
    return err;
}

char* ortkt_tensor_info(const ortkt_value* value, int32_t* element_type, size_t* rank, size_t* element_count) {
    int is_tensor = 0;
    CHECK(g_api->IsTensor(value->value, &is_tensor));
    if (!is_tensor) return dup_string("Value is not a tensor; only tensor outputs are supported");
    OrtTensorTypeAndShapeInfo* info = NULL;
    CHECK(g_api->GetTensorTypeAndShape(value->value, &info));
    ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    char* err = check(g_api->GetTensorElementType(info, &type));
    if (!err) err = check(g_api->GetDimensionsCount(info, rank));
    if (!err) err = check(g_api->GetTensorShapeElementCount(info, element_count));
    g_api->ReleaseTensorTypeAndShapeInfo(info);
    *element_type = (int32_t)type;
    return err;
}

char* ortkt_tensor_shape(const ortkt_value* value, int64_t* shape, size_t rank) {
    OrtTensorTypeAndShapeInfo* info = NULL;
    CHECK(g_api->GetTensorTypeAndShape(value->value, &info));
    char* err = rank > 0 ? check(g_api->GetDimensions(info, shape, rank)) : NULL;
    g_api->ReleaseTensorTypeAndShapeInfo(info);
    return err;
}

char* ortkt_tensor_data(ortkt_value* value, const void** data) {
    void* ptr = NULL;
    CHECK(g_api->GetTensorMutableData(value->value, &ptr));
    *data = ptr;
    return NULL;
}

char* ortkt_string_tensor_length(const ortkt_value* value, size_t* total_bytes) {
    CHECK(g_api->GetStringTensorDataLength(value->value, total_bytes));
    return NULL;
}

char* ortkt_string_tensor_content(const ortkt_value* value, char* buffer, size_t buffer_size,
                                  size_t* offsets, size_t count) {
    if (count == 0) return NULL;
    CHECK(g_api->GetStringTensorContent(value->value, buffer, buffer_size, offsets, count));
    return NULL;
}
