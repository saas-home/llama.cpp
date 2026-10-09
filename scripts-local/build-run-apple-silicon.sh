#!/bin/bash
set -euo pipefail

# ====================== USAGE ==============================
# ./scripts-local/build-run-apple-silicon.sh                      -> run default .conf
# ./scripts-local/build-run-apple-silicon.sh [config]             -> run specific .conf
# ./scripts-local/build-run-apple-silicon.sh --build              -> rebuild binaries before running
# ./scripts-local/build-run-apple-silicon.sh --build-only         -> build binaries and exit (no server)
# ./scripts-local/build-run-apple-silicon.sh --clean --build-only -> clean build directory and rebuild
# ./scripts-local/build-run-apple-silicon.sh [config] -f          -> run in foreground
# ./scripts-local/build-run-apple-silicon.sh [config] --stop      -> stop server for config
# ./scripts-local/build-run-apple-silicon.sh --stop               -> stop all running llama-server instances
# ============================================================

BUILD=false
BUILD_ONLY=false
CLEAN=false
BUILD_TYPE="Release"
USE_METAL="ON"
BUILD_DIR=""
JOBS=""

BENCH=false
BENCH_ONLY=false
BENCH_COUNT=1
BENCH_PARALLEL=1
BENCH_BUDGET=500
FOREGROUND=false
STOP=false
CONFIG_OVERRIDE=""
EXPLICIT_SERVICE=""
EXPLICIT_PORT=""
EXPLICIT_HOST=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Runtime directory for PIDs and logs
RUN_DIR="$HOME/.cache/llama-run"
mkdir -p "$RUN_DIR"

# Auto-detect Apple Silicon CPU topology
# Performance cores (P-cores) avoid efficiency core synchronization barrier stalls
NCPU=$(sysctl -n hw.logicalcpu 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
NPERF=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo "$NCPU")
MEM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
MEM_GB=$(( MEM_BYTES / 1024 / 1024 / 1024 ))

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build) BUILD=true; shift ;;
        --build-only|--no-run) BUILD=true; BUILD_ONLY=true; shift ;;
        --clean) CLEAN=true; BUILD=true; shift ;;
        --debug) BUILD_TYPE="Debug"; BUILD=true; shift ;;
        --no-metal) USE_METAL="OFF"; BUILD=true; shift ;;
        --build-dir=*) BUILD_DIR="${1#*=}"; shift ;;
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        -j|--jobs) JOBS="$2"; shift 2 ;;
        -j*) JOBS="${1#-j}"; shift ;;
        --bench) BENCH=true; shift ;;
        --bench-only) BENCH_ONLY=true; BENCH=true; shift ;;
        --bench-count=*) BENCH_COUNT="${1#*=}"; shift ;;
        --bench-count) BENCH_COUNT="$2"; shift 2 ;;
        --bench-parallel=*) BENCH_PARALLEL="${1#*=}"; shift ;;
        --bench-parallel) BENCH_PARALLEL="$2"; shift 2 ;;
        --bench-budget=*) BENCH_BUDGET="${1#*=}"; shift ;;
        --bench-budget) BENCH_BUDGET="$2"; shift 2 ;;
        --foreground|-f) FOREGROUND=true; shift ;;
        --stop|--disable|--stop-service|--stop-disable|--down) STOP=true; shift ;;
        --service=*) EXPLICIT_SERVICE="${1#*=}"; shift ;;
        --service) EXPLICIT_SERVICE="$2"; shift 2 ;;
        --port=*) EXPLICIT_PORT="${1#*=}"; shift ;;
        --port) EXPLICIT_PORT="$2"; shift 2 ;;
        --host=*) EXPLICIT_HOST="${1#*=}"; shift ;;
        --host) EXPLICIT_HOST="$2"; shift 2 ;;
        *.conf) CONFIG_OVERRIDE="$1"; shift ;;
        -h|--help)
            cat << EOF
Usage: $(basename "$0") [config.conf] [options]

Unified build and run script for llama.cpp on Apple Silicon.

Build options:
  --build               Rebuild binaries before launching
  --build-only, --no-run Build binaries and exit without starting server
  --clean               Clean build directory before building
  --debug               Build with Debug configuration (default: Release)
  --no-metal            Disable Metal backend (CPU-only build)
  --build-dir=<path>    Custom build directory (default: ./build)
  -j, --jobs <N>        Number of parallel compilation jobs (default: $NPERF)

Runtime options:
  --foreground, -f      Run attached to current terminal
  --stop, --down        Stop running server instance(s)
  --service=<name>      Explicit service name identifier
  --host=<host>         Override listening host (default: 0.0.0.0)
  --port=<port>         Override listening port (default: 8080)
  --bench               Run benchmark after startup
  --bench-only          Run benchmark without launching new server
  --bench-count=<N>     Number of benchmark requests (default: 1)
  --bench-parallel=<N>  Number of concurrent benchmark streams (default: 1)
  --bench-budget=<N>    Token budget per benchmark stream (default: 500)
  -h, --help            Show this help message
EOF
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Resolve default build directory and jobs
BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build}"
JOBS="${JOBS:-$NPERF}"

# Helper to find compiled server binary
find_server_bin() {
    for candidate in \
        "$BUILD_DIR/bin/llama-server" \
        "$BUILD_DIR/bin/$BUILD_TYPE/llama-server" \
        "$BUILD_DIR/bin/Release/llama-server" \
        "$BUILD_DIR/bin/Debug/llama-server"; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# Helper to find built Web UI dist
find_ui_dist() {
    for candidate in \
        "$BUILD_DIR/tools/ui/dist" \
        "$BUILD_DIR/tools/ui/$BUILD_TYPE/dist" \
        "$REPO_DIR/tools/ui/dist" \
        "$REPO_DIR/build/tools/ui/dist"; do
        if [[ -d "$candidate" ]]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

# Build function
build_llama() {
    cd "$REPO_DIR"
    echo "=== Building llama.cpp for Apple Silicon ==="

    OS_NAME=$(uname -s)
    ARCH_NAME=$(uname -m)
    if [[ "$OS_NAME" != "Darwin" ]]; then
        echo "Warning: Current OS is $OS_NAME (expected Darwin/macOS)."
    fi
    if [[ "$ARCH_NAME" != "arm64" ]]; then
        echo "Warning: Current architecture is $ARCH_NAME (expected arm64 for Apple Silicon)."
    fi

    if ! command -v cmake >/dev/null 2>&1; then
        echo "Error: cmake is required but not installed."
        echo "Install via Homebrew: brew install cmake"
        exit 1
    fi

    GENERATOR_ARGS=()
    if command -v ninja >/dev/null 2>&1; then
        GENERATOR_ARGS=("-G" "Ninja")
        echo "Using build generator: Ninja"
    else
        echo "Using build generator: Unix Makefiles"
    fi

    CCACHE_OPT="OFF"
    if command -v ccache >/dev/null 2>&1; then
        CCACHE_OPT="ON"
        echo "ccache found: enabling GGML_CCACHE"
    fi

    if [[ "$CLEAN" == true && -d "$BUILD_DIR" ]]; then
        echo "Cleaning build directory: $BUILD_DIR"
        rm -rf "$BUILD_DIR"
    fi

    echo "Build configuration:"
    echo "  Build type  : $BUILD_TYPE"
    echo "  Build dir   : $BUILD_DIR"
    echo "  Metal GPU   : $USE_METAL"
    echo "  Accelerate  : ON"
    echo "  Jobs        : $JOBS"

    CMAKE_ARGS=(
        "-B" "$BUILD_DIR"
        "-S" "$REPO_DIR"
        "${GENERATOR_ARGS[@]}"
        "-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
        "-DCMAKE_OSX_ARCHITECTURES=arm64"
        "-DGGML_NATIVE=ON"
        "-DGGML_METAL=$USE_METAL"
        "-DGGML_METAL_EMBED_LIBRARY=$USE_METAL"
        "-DGGML_METAL_NDEBUG=ON"
        "-DGGML_ACCELERATE=ON"
        "-DGGML_BLAS=ON"
        "-DGGML_BLAS_VENDOR=Apple"
        "-DGGML_CCACHE=$CCACHE_OPT"
        "-DGGML_CURL=ON"
        "-DLLAMA_BUILD_SERVER=ON"
        "-DLLAMA_BUILD_EXAMPLES=ON"
        "-DLLAMA_BUILD_TOOLS=ON"
    )

    echo "Running CMake configuration..."
    cmake "${CMAKE_ARGS[@]}"

    echo "Compiling targets (jobs: $JOBS)..."
    cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" -j "$JOBS"
    echo "Build completed successfully."

    BIN_DIR="$BUILD_DIR/bin"
    [[ ! -d "$BIN_DIR" && -d "$BUILD_DIR/bin/$BUILD_TYPE" ]] && BIN_DIR="$BUILD_DIR/bin/$BUILD_TYPE"

    if [[ -d "$BIN_DIR" ]]; then
        echo "Compiled binaries in $BIN_DIR:"
        for bin_name in llama-cli llama-server llama-bench llama-quantize; do
            if [[ -x "$BIN_DIR/$bin_name" ]]; then
                echo "  - $bin_name"
            fi
        done
    fi
}

# If --build-only requested, compile and exit immediately
if [[ "$BUILD_ONLY" == true ]]; then
    build_llama
    echo "Build complete. Exiting (--build-only mode)."
    exit 0
fi

# Graceful termination helper
stop_pid() {
    local pid="$1"
    local name="$2"
    if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping $name (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "Stopped $name successfully."
                return 0
            fi
            sleep 0.5
        done
        echo "Force killing $name (PID: $pid)..."
        kill -9 "$pid" 2>/dev/null || true
    fi
}

# Stop logic
if [[ "$STOP" == true ]]; then
    echo "=== Stopping llama-server ==="

    if [[ -n "$CONFIG_OVERRIDE" ]]; then
        if [[ -f "$CONFIG_OVERRIDE" ]]; then
            RESOLVED_CONF="$CONFIG_OVERRIDE"
        elif [[ -f "$SCRIPT_DIR/$CONFIG_OVERRIDE" ]]; then
            RESOLVED_CONF="$SCRIPT_DIR/$CONFIG_OVERRIDE"
        else
            echo "Error: Config file not found: $CONFIG_OVERRIDE"
            exit 1
        fi

        PORT=8080
        SERVICE_NAME="llama-server"
        source "$RESOLVED_CONF"
        [[ -n "$EXPLICIT_SERVICE" ]] && SERVICE_NAME="$EXPLICIT_SERVICE"
        [[ -n "$EXPLICIT_PORT" ]] && PORT="$EXPLICIT_PORT"

        PID_FILE="$RUN_DIR/${SERVICE_NAME}.pid"
        STOPPED=false

        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                stop_pid "$PID" "$SERVICE_NAME"
                STOPPED=true
            fi
            rm -f "$PID_FILE"
        fi

        PORT_PIDS=$(lsof -ti :"$PORT" 2>/dev/null || true)
        if [[ -n "$PORT_PIDS" ]]; then
            for p in $PORT_PIDS; do
                stop_pid "$p" "process on port $PORT"
                STOPPED=true
            done
        fi

        if [[ "$STOPPED" == false ]]; then
            echo "No running server found for $SERVICE_NAME (port $PORT)."
        fi
        exit 0
    elif [[ -n "$EXPLICIT_SERVICE" ]]; then
        PID_FILE="$RUN_DIR/${EXPLICIT_SERVICE}.pid"
        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                stop_pid "$PID" "$EXPLICIT_SERVICE"
            fi
            rm -f "$PID_FILE"
        else
            echo "No PID file found for service $EXPLICIT_SERVICE."
        fi
        exit 0
    else
        RUNNING_PIDS=$(pgrep -x "llama-server" 2>/dev/null || true)
        if [[ -n "$RUNNING_PIDS" ]]; then
            for p in $RUNNING_PIDS; do
                stop_pid "$p" "llama-server"
            done
        else
            echo "No running llama-server processes found."
        fi
        rm -f "$RUN_DIR"/*.pid
        echo "All instances stopped."
        exit 0
    fi
fi

# Default configuration values
MODEL_PATH=""
MMPRJ_PATH=""
MODEL_ALIAS=""
SERVICE_NAME="llama-server"

THREADS="$NPERF"
THREADS_BATCH="$NPERF"
THREADS_HTTP=4
PRIORITY=2
PRIORITY_BATCH=1
FLASH_ATTN="on"

N_GPU_LAYERS=999
N_CPU_MOE=0
MOE_CACHE_MIB=0
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"
CTX_SIZE=32768
PARALLEL=1
BATCH_SIZE=2048
UBATCH_SIZE=512

CACHE_RAM=32768
CACHE_REUSE=256
KV_UNIFIED="true"
CLEAR_IDLE="true"
CACHE_IDLE_SLOTS="true"
CONTEXT_SHIFT="true"
SLOT_SAVE_PATH="$HOME/.cache/llama-slots"
KV_OFFLOAD="true"
CONT_BATCHING="true"

# Speculative decoding & draft model settings
MODEL_DRAFT=""
SPEC_TYPE=""
SPEC_DRAFT_N_MAX=""
SPEC_DRAFT_N_MIN=""
SPEC_DRAFT_P_MIN=""
GPU_LAYERS_DRAFT=""

TEMP=""
MIN_P=""
XTC_PROBABILITY=""
XTC_THRESHOLD=""
TOP_P=""
TOP_K=""
REPEAT_PENALTY=""
REPEAT_LAST_N=""
PRESENCE_PENALTY=""

DRY_MULTIPLIER=""
DRY_BASE=""
DRY_ALLOWED_LENGTH=""
DRY_PENALTY_LAST_N=""
DRY_SEQUENCE_BREAKERS=""

REASONING="auto"
REASONING_FORMAT="auto"
REASONING_BUDGET=-1
REASONING_BUDGET_MESSAGE=""
REASONING_EFFORT=""
REASONING_PRESERVE=""
JINJA=false
JINJA_KWARGS=""
CHAT_TEMPLATE_FILE=""
EXTRA_ARGS=""

LAZY_MODE=""
BACKEND_SAMPLING=""
API_KEY=""
TIMEOUT=""

SAMPLERS=""
HOST="0.0.0.0"
PORT=8080
LOG_DISABLE=false
MLOCK=false
MMAP=true
MMPRJ_OFFLOAD=true
IMAGE_MAX_TOKENS=""
IMAGE_MIN_TOKENS=""

# Default to an existing Apple Silicon config if none provided
if [[ -z "$CONFIG_OVERRIDE" ]]; then
    if [[ -f "$SCRIPT_DIR/qwen-3.8-flash-next-iq3_xxs.conf" ]]; then
        CONFIG_OVERRIDE="$SCRIPT_DIR/qwen-3.8-flash-next-iq3_xxs.conf"
    elif [[ -f "$SCRIPT_DIR/qwen-3.8-flash-next-q2.conf" ]]; then
        CONFIG_OVERRIDE="$SCRIPT_DIR/qwen-3.8-flash-next-q2.conf"
    elif [[ -f "$SCRIPT_DIR/qwen-3.8-27b-gsq.conf" ]]; then
        CONFIG_OVERRIDE="$SCRIPT_DIR/qwen-3.8-27b-gsq.conf"
    fi
fi

if [[ -n "$CONFIG_OVERRIDE" ]]; then
    if [[ -f "$CONFIG_OVERRIDE" ]]; then
        RESOLVED_CONF="$CONFIG_OVERRIDE"
    elif [[ -f "$SCRIPT_DIR/$CONFIG_OVERRIDE" ]]; then
        RESOLVED_CONF="$SCRIPT_DIR/$CONFIG_OVERRIDE"
    else
        echo "Error: Config file not found: $CONFIG_OVERRIDE"
        exit 1
    fi
    echo "Loading profile: $RESOLVED_CONF"
    source "$RESOLVED_CONF"
fi

# Command-line overrides
[[ -n "$EXPLICIT_SERVICE" ]] && SERVICE_NAME="$EXPLICIT_SERVICE"
[[ -n "$EXPLICIT_HOST" ]] && HOST="$EXPLICIT_HOST"
[[ -n "$EXPLICIT_PORT" ]] && PORT="$EXPLICIT_PORT"

# Resolve model path on macOS (handles absolute, relative, and translated user paths)
resolve_path() {
    local p="$1"
    [[ -z "$p" ]] && return 0
    if [[ -f "$p" ]]; then
        echo "$p"
        return 0
    fi
    if [[ -f "$REPO_DIR/$p" ]]; then
        echo "$REPO_DIR/$p"
        return 0
    fi
    if [[ "$p" =~ ^/home/[^/]+/(.*) || "$p" =~ ^/Users/[^/]+/(.*) ]]; then
        local rel="${BASH_REMATCH[1]}"
        if [[ -f "$HOME/$rel" ]]; then
            echo "$HOME/$rel"
            return 0
        fi
    fi
    local base
    base=$(basename "$p")
    for dir in "$REPO_DIR/models" "$REPO_DIR/gguf" "$HOME/models" "$HOME/model_dir"; do
        if [[ -f "$dir/$base" ]]; then
            echo "$dir/$base"
            return 0
        fi
    done
    echo "$p"
}

RESOLVED_MODEL=$(resolve_path "$MODEL_PATH")
RESOLVED_MMPRJ=$(resolve_path "$MMPRJ_PATH")

if [[ ! -f "$RESOLVED_MODEL" ]]; then
    echo "Warning: Model file not found at: $RESOLVED_MODEL"
    echo "Please update MODEL_PATH in your configuration."
fi

# Resolve server binary path
SERVER_BIN=$(find_server_bin || echo "$BUILD_DIR/bin/llama-server")

# Rebuild if requested or if binary is missing
if [[ "$BUILD" == true || ! -x "$SERVER_BIN" ]]; then
    build_llama
    SERVER_BIN=$(find_server_bin || echo "")
fi

if [[ -z "$SERVER_BIN" || ! -x "$SERVER_BIN" ]]; then
    echo "Error: Binary not found at $SERVER_BIN"
    exit 1
fi

if [[ "$BENCH_ONLY" == true ]]; then
    echo "Running in benchmark-only mode."
else
    # Check if existing instance is on the target port
    EXISTING_PID=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [[ -n "$EXISTING_PID" ]]; then
        echo "Found running process on port $PORT (PID: $EXISTING_PID). Stopping it..."
        stop_pid "$EXISTING_PID" "previous instance"
    fi

    # Build command line
    CMD=("$SERVER_BIN")
    CMD+=("--model" "$RESOLVED_MODEL")
    [[ -n "${MODEL_ALIAS:-}" ]] && CMD+=("--alias" "$MODEL_ALIAS")

    UI_DIST=$(find_ui_dist || true)
    [[ -n "$UI_DIST" ]] && CMD+=("--path" "$UI_DIST")

    [[ -n "${RESOLVED_MMPRJ:-}" && -f "$RESOLVED_MMPRJ" ]] && CMD+=("--mmproj" "$RESOLVED_MMPRJ")
    [[ "${MMPRJ_OFFLOAD:-true}" == "false" ]] && CMD+=("--no-mmproj-offload")
    [[ -n "${IMAGE_MAX_TOKENS:-}" ]] && CMD+=("--image-max-tokens" "$IMAGE_MAX_TOKENS")
    [[ -n "${IMAGE_MIN_TOKENS:-}" ]] && CMD+=("--image-min-tokens" "$IMAGE_MIN_TOKENS")
    CMD+=("--n-gpu-layers" "$N_GPU_LAYERS")
    [[ -n "${N_CPU_MOE:-}" && "$N_CPU_MOE" -gt 0 ]] && CMD+=("--n-cpu-moe" "$N_CPU_MOE")
    [[ -n "${MOE_CACHE_MIB:-}" && "$MOE_CACHE_MIB" -gt 0 ]] && CMD+=("--moe-cache-mib" "$MOE_CACHE_MIB")
    CMD+=("--cache-type-k" "$CACHE_TYPE_K")
    CMD+=("--cache-type-v" "$CACHE_TYPE_V")

    if [[ "${MLOCK:-false}" == "true" && "${MMAP:-true}" == "true" ]]; then
        CMD+=("--load-mode" "mmap+mlock")
    elif [[ "${MLOCK:-false}" == "true" ]]; then
        CMD+=("--load-mode" "mlock")
    elif [[ "${MMAP:-true}" == "false" ]]; then
        CMD+=("--load-mode" "none")
    fi

    # Speculative decoding & draft models
    [[ -n "${MODEL_DRAFT:-}" ]] && CMD+=("--model-draft" "$MODEL_DRAFT")
    [[ -n "${SPEC_TYPE:-}" ]] && CMD+=("--spec-type" "$SPEC_TYPE")
    [[ -n "${SPEC_DRAFT_N_MAX:-}" ]] && CMD+=("--spec-draft-n-max" "$SPEC_DRAFT_N_MAX")
    [[ -n "${SPEC_DRAFT_N_MIN:-}" ]] && CMD+=("--spec-draft-n-min" "$SPEC_DRAFT_N_MIN")
    [[ -n "${SPEC_DRAFT_P_MIN:-}" ]] && CMD+=("--spec-draft-p-min" "$SPEC_DRAFT_P_MIN")
    [[ -n "${GPU_LAYERS_DRAFT:-}" ]] && CMD+=("--gpu-layers-draft" "$GPU_LAYERS_DRAFT")

    CMD+=("--parallel" "$PARALLEL")
    CMD+=("--cache-ram" "$CACHE_RAM")
    [[ -n "${CACHE_REUSE:-}" ]] && CMD+=("--cache-reuse" "$CACHE_REUSE")
    if [[ "${KV_UNIFIED:-true}" == "true" ]]; then
        CMD+=("--kv-unified")
    elif [[ "${KV_UNIFIED:-true}" == "false" ]]; then
        CMD+=("--no-kv-unified")
    fi
    [[ "${KV_OFFLOAD:-true}" == "false" ]] && CMD+=("--no-kv-offload")
    if [[ "${CLEAR_IDLE:-true}" == "true" || "${CACHE_IDLE_SLOTS:-true}" == "true" ]]; then
        CMD+=("--cache-idle-slots")
    elif [[ "${CLEAR_IDLE:-true}" == "false" || "${CACHE_IDLE_SLOTS:-true}" == "false" ]]; then
        CMD+=("--no-cache-idle-slots")
    fi
    if [[ "${CONTEXT_SHIFT:-true}" == "true" ]]; then
        CMD+=("--context-shift")
    elif [[ "${CONTEXT_SHIFT:-true}" == "false" ]]; then
        CMD+=("--no-context-shift")
    fi
    if [[ -n "${SLOT_SAVE_PATH:-}" ]]; then
        mkdir -p "$SLOT_SAVE_PATH" 2>/dev/null || true
        CMD+=("--slot-save-path" "$SLOT_SAVE_PATH")
    fi
    if [[ "${CONT_BATCHING:-true}" == "true" ]]; then
        CMD+=("--cont-batching")
    elif [[ "${CONT_BATCHING:-true}" == "false" ]]; then
        CMD+=("--no-cont-batching")
    fi

    CMD+=("--threads" "$THREADS")
    CMD+=("--threads-batch" "$THREADS_BATCH")
    CMD+=("--threads-http" "$THREADS_HTTP")
    CMD+=("--prio" "$PRIORITY")
    CMD+=("--prio-batch" "$PRIORITY_BATCH")
    CMD+=("--flash-attn" "$FLASH_ATTN")
    CMD+=("--ctx-size" "$CTX_SIZE")
    CMD+=("--batch-size" "$BATCH_SIZE")
    CMD+=("--ubatch-size" "$UBATCH_SIZE")

    CMD+=("--reasoning" "$REASONING")
    CMD+=("--reasoning-format" "$REASONING_FORMAT")
    CMD+=("--reasoning-budget" "$REASONING_BUDGET")
    [[ -n "${REASONING_BUDGET_MESSAGE:-}" ]] && CMD+=("--reasoning-budget-message" "$REASONING_BUDGET_MESSAGE")
    [[ -n "${REASONING_EFFORT:-}" ]] && CMD+=("--reasoning-effort" "$REASONING_EFFORT")
    if [[ "${REASONING_PRESERVE:-}" == "true" ]]; then
        CMD+=("--reasoning-preserve")
    elif [[ "${REASONING_PRESERVE:-}" == "false" ]]; then
        CMD+=("--no-reasoning-preserve")
    fi
    [[ "${JINJA:-false}" == "true" ]] && CMD+=("--jinja")
    [[ -n "${JINJA_KWARGS:-}" ]] && CMD+=("--chat-template-kwargs" "$JINJA_KWARGS")
    [[ -n "${CHAT_TEMPLATE_FILE:-}" && -f "$CHAT_TEMPLATE_FILE" ]] && CMD+=("--chat-template-file" "$CHAT_TEMPLATE_FILE")

    [[ -n "${TEMP:-}" ]] && CMD+=("--temp" "$TEMP")
    [[ -n "${MIN_P:-}" ]] && CMD+=("--min-p" "$MIN_P")
    [[ -n "${XTC_PROBABILITY:-}" ]] && CMD+=("--xtc-probability" "$XTC_PROBABILITY")
    [[ -n "${XTC_THRESHOLD:-}" ]] && CMD+=("--xtc-threshold" "$XTC_THRESHOLD")
    [[ -n "${TOP_P:-}" ]] && CMD+=("--top-p" "$TOP_P")
    [[ -n "${TOP_K:-}" ]] && CMD+=("--top-k" "$TOP_K")
    [[ -n "${REPEAT_LAST_N:-}" ]] && CMD+=("--repeat-last-n" "$REPEAT_LAST_N")
    [[ -n "${REPEAT_PENALTY:-}" ]] && CMD+=("--repeat-penalty" "$REPEAT_PENALTY")
    [[ -n "${PRESENCE_PENALTY:-}" ]] && CMD+=("--presence-penalty" "$PRESENCE_PENALTY")
    [[ -n "${DRY_MULTIPLIER:-}" ]] && CMD+=("--dry-multiplier" "$DRY_MULTIPLIER")
    [[ -n "${DRY_BASE:-}" ]] && CMD+=("--dry-base" "$DRY_BASE")
    [[ -n "${DRY_ALLOWED_LENGTH:-}" ]] && CMD+=("--dry-allowed-length" "$DRY_ALLOWED_LENGTH")
    [[ -n "${DRY_PENALTY_LAST_N:-}" ]] && CMD+=("--dry-penalty-last-n" "$DRY_PENALTY_LAST_N")

    if [[ -n "${DRY_SEQUENCE_BREAKERS:-}" ]]; then
        for breaker in $DRY_SEQUENCE_BREAKERS; do
            fixed_breaker=$(printf '%b' "$breaker")
            CMD+=("--dry-sequence-breaker" "$fixed_breaker")
        done
    fi

    [[ -n "${SAMPLERS:-}" ]] && CMD+=("--samplers" "$SAMPLERS")
    [[ -n "${LAZY_MODE:-}" ]] && CMD+=("--lazy-mode" "$LAZY_MODE")
    [[ "${BACKEND_SAMPLING:-false}" == "true" ]] && CMD+=("--backend-sampling")
    [[ -n "${API_KEY:-}" ]] && CMD+=("--api-key" "$API_KEY")
    [[ -n "${TIMEOUT:-}" ]] && CMD+=("--timeout" "$TIMEOUT")

    CMD+=("--host" "$HOST")
    CMD+=("--port" "$PORT")
    [[ "${LOG_DISABLE:-false}" == "true" ]] && CMD+=("--log-disable")

    if [[ -n "${EXTRA_ARGS:-}" ]]; then
        eval "extra_array=($EXTRA_ARGS)"
        CMD+=("${extra_array[@]}")
    fi
    CMD+=("--metrics")

    LOG_FILE="$RUN_DIR/${SERVICE_NAME}.log"
    PID_FILE="$RUN_DIR/${SERVICE_NAME}.pid"

    CHIP_NAME=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    echo "=== Starting llama-server on Apple Silicon ==="
    echo "Hardware     : $CHIP_NAME (${MEM_GB}GB Unified Memory, ${NPERF} P-cores / ${NCPU} total)"
    echo "Service name : $SERVICE_NAME"
    echo "Host / Port  : http://$HOST:$PORT"
    echo "Model        : $RESOLVED_MODEL"
    echo "Threads      : $THREADS (batch: $THREADS_BATCH)"

    # Metal residency optimization (keep active sets wired for 30m)
    export GGML_METAL_RESIDENCY_KEEP_ALIVE_S="${GGML_METAL_RESIDENCY_KEEP_ALIVE_S:-1800}"

    if [[ "$FOREGROUND" == true ]]; then
        echo "Running in foreground (Ctrl+C to exit)..."
        exec "${CMD[@]}"
    else
        echo "Starting in background..."
        echo "Logs: $LOG_FILE"
        nohup "${CMD[@]}" > "$LOG_FILE" 2>&1 &
        SERVER_PID=$!
        echo "$SERVER_PID" > "$PID_FILE"

        echo "Waiting for server to initialize..."
        CHECK_HOST="${HOST}"
        [[ "$CHECK_HOST" == "0.0.0.0" ]] && CHECK_HOST="127.0.0.1"
        INITIALIZED=false
        for i in {1..20}; do
            sleep 1
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "Error: Server failed to start (crashed). Last log lines:"
                tail -n 30 "$LOG_FILE"
                rm -f "$PID_FILE"
                exit 1
            fi
            if command -v curl >/dev/null 2>&1; then
                HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://${CHECK_HOST}:${PORT}/health" 2>/dev/null || true)
                if [[ "$HTTP_CODE" == "200" || "$HTTP_CODE" == "503" ]]; then
                    INITIALIZED=true
                    break
                fi
            fi
        done
        echo "Server running with PID: $SERVER_PID"
        echo "Endpoint: http://$HOST:$PORT"
    fi
fi

if [[ "$BENCH" == true ]]; then
    echo "Checking server readiness before benchmarking..."
    CHECK_HOST="${HOST}"
    [[ "$CHECK_HOST" == "0.0.0.0" ]] && CHECK_HOST="127.0.0.1"
    for i in {1..60}; do
        HEALTH_STATUS=$(curl -s "http://${CHECK_HOST}:${PORT}/health" 2>/dev/null || true)
        if echo "$HEALTH_STATUS" | grep -q '"status":"ok"\|"ok"'; then
            break
        fi
        sleep 1
    done
    echo "Running benchmark..."
    MODEL_NAME="${MODEL_ALIAS:-$(basename "$RESOLVED_MODEL")}"
    if [[ -f "$SCRIPT_DIR/bench-llama.py" ]]; then
        OUTPUT=$(python3 "$SCRIPT_DIR/bench-llama.py" "$BENCH_COUNT" -p "$BENCH_PARALLEL" --port "$PORT" --model "$MODEL_NAME" --budget "$BENCH_BUDGET")
        echo "$OUTPUT"

        BASELINES_FILE="$SCRIPT_DIR/baselines.json"
        CONFIG_NAME=$(basename "${CONFIG_OVERRIDE:-script_defaults}")
        if [[ -f "$BASELINES_FILE" ]]; then
            CUR_AVG_GEN=$(echo "$OUTPUT" | grep "Generation" | tail -n 1 | awk '{print $4}')
            CUR_AGG_THR=$(echo "$OUTPUT" | grep "Throughput" | tail -n 1 | awk '{print $3}')

            if [[ -n "$CUR_AVG_GEN" && -n "$CUR_AGG_THR" ]]; then
                python3 - << EOF
import json

file_path = "$BASELINES_FILE"
config_name = "$CONFIG_NAME"

try:
    with open(file_path, "r") as f:
        data = json.load(f)

    if config_name in data:
        base = data[config_name]
        cur_gen = float("$CUR_AVG_GEN")
        cur_thr = float("$CUR_AGG_THR")

        gen_diff = ((cur_gen - base['avg_gen']) / base['avg_gen']) * 100
        thr_diff = ((cur_thr - base['agg_thr']) / base['agg_thr']) * 100

        print(f"\nPerformance Comparison vs Golden Baseline ({base.get('timestamp', 'N/A')}):")

        def fmt_diff(diff):
            color = "\033[92m" if diff >= -2 else ("\033[93m" if diff >= -10 else "\033[91m")
            reset = "\033[0m"
            return f"{color}{diff:+.1f}%{reset}"

        print(f"   - Avg Generation: {base['avg_gen']:.1f} -> {cur_gen:.1f} tok/s ({fmt_diff(gen_diff)})")
        print(f"   - Agg Throughput: {base['agg_thr']:.1f} -> {cur_thr:.1f} tok/s ({fmt_diff(thr_diff)})")

        if thr_diff < -10:
            print("\nWARNING: PERFORMANCE REGRESSION DETECTED (>10% drop in throughput!)")
    else:
        print(f"\nNo baseline found for {config_name}.")
except Exception as e:
    print(f"\nError comparing baselines: {e}")
EOF
            fi
        fi
    else
        echo "bench-llama.py not found in $SCRIPT_DIR."
    fi
fi
