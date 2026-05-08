#!/bin/bash
set -euo pipefail

# ====================== USAGE ==============================
# ./rebuild-llama.sh                      → use script defaults
# ./rebuild-llama.sh [config]             → override with specific .conf
# ./rebuild-llama.sh [config] --build     → build from source
# ./rebuild-llama.sh [config] --bench     → run benchmark
# ./rebuild-llama.sh [config] --generate-ui → regenerate OpenWeb-UI profiles
# ./rebuild-llama.sh [config] --no-deploy   → stop and build, but do not deploy/restart
# ============================================================

BUILD=false
BENCH=false
BENCH_ONLY=false
BENCH_COUNT=1
BENCH_PARALLEL=1
BENCH_BUDGET=500
GEN_UI=false
DEPLOY=true
CONFIG_OVERRIDE=""

# --- Default Current Configuration (Mythos-26B-A4B-PRISM Optimized) ---
# Optimized for: AMD 7950X3D | RTX 4070 Ti Super (16GB) | Ubuntu 26.04
# Goal: Maximum Context (256K) with 4-slot stability.

# 0. Dependency check for Ubuntu
if [[ -f /etc/lsb-release ]] && grep -q "Ubuntu" /etc/lsb-release; then
    echo "🔍 Checking for Ubuntu dependencies..."
    DEPS=(build-essential cmake ninja-build git libcurl4-openssl-dev pkg-config)
    # Only auto-install Ubuntu's nvidia-cuda-toolkit if NVIDIA's official toolkit isn't present
    if [[ ! -x /usr/local/cuda/bin/nvcc ]]; then
        DEPS+=(nvidia-cuda-toolkit)
    fi
    MISSING_DEPS=()
    for dep in "${DEPS[@]}"; do
        if ! dpkg -s "$dep" >/dev/null 2>&1; then
            MISSING_DEPS+=("$dep")
        fi
    done
    if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
        echo "📦 Installing missing dependencies: ${MISSING_DEPS[*]}"
        sudo apt update && sudo apt install -y "${MISSING_DEPS[@]}"
    fi
fi

# Ensure nvcc is in path
if [[ -d "/usr/local/cuda/bin" ]]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
fi

MODEL_PATH="/home/siva/models/gemma-4-26B-A4B-it/Ex0bit/mythos-26b-a4b-prism-pro-dq.gguf"
MMPRJ_PATH="/home/siva/models/gemma-4-26B-A4B-it/Ex0bit/mmproj-mythos-26b-a4b-prism-pro.gguf"
MODEL_ALIAS="${MODEL_ALIAS:-}"
SERVICE_NAME="llama-server.service"

# CPU & Scheduling Optimization
# Full V-Cache CCD Focus: Cores 0-7 + SMT 16-23.
CPU_AFFINITY="0-15"
THREADS=16
THREADS_BATCH=16
THREADS_HTTP=4
PRIORITY=2
PRIORITY_BATCH=1

# GPU & Memory
N_GPU_LAYERS=999
# Balanced offloading: 12 experts on CPU (4 on GPU) to utilize the 16GB VRAM headroom.
N_CPU_MOE=12
CACHE_TYPE_K="q8_0"
CACHE_TYPE_V="q8_0"
# Stable Context: 128K total (~42K per slot x 3 slots).
CTX_SIZE=131072
PARALLEL=3
BATCH_SIZE=4096
UBATCH_SIZE=512

# KV Cache & Persistence Optimizations
CACHE_RAM=32768
CACHE_REUSE=256
KV_UNIFIED="true"
CLEAR_IDLE="true"
CONTEXT_SHIFT="true"
SLOT_SAVE_PATH="/home/siva/.cache/llama-slots"

# Sampling & Logic
TEMP=0.4
MIN_P=0.02
XTC_PROBABILITY=0.1
XTC_THRESHOLD=0.1
TOP_P=0.95
TOP_K=50
REPEAT_PENALTY=1.1
REPEAT_LAST_N=64
PRESENCE_PENALTY=0.0

# DRY Sampler
DRY_MULTIPLIER=0.8
DRY_BASE=1.75
DRY_ALLOWED_LENGTH=2
DRY_PENALTY_LAST_N=4096

# Reasoning / Thinking
REASONING="auto"
REASONING_FORMAT="deepseek"
REASONING_BUDGET=-1
REASONING_BUDGET_MESSAGE=""
JINJA="${JINJA:-false}"

SAMPLERS="dry;top_k;top_p;xtc;min_p;temperature"
HOST="0.0.0.0"
PORT=8080
SLEEP_IDLE_SECONDS=300

# Model Persistence & Mapping
MLOCK="${MLOCK:-false}"
MMAP="${MMAP:-true}"

# 1. Environment & Base Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME_DEFAULT="llama-server.service"
SERVICE_NAME="${SERVICE_NAME:-$SERVICE_NAME_DEFAULT}"

# 2. Parse arguments and check for overrides
for arg in "$@"; do
    case "$arg" in
        --build) BUILD=true ;;
        --bench) BENCH=true ;;
        --bench-only) BENCH_ONLY=true; BENCH=true ;;
        --bench-count=*) BENCH_COUNT="${arg#*=}" ;;
        --bench-parallel=*) BENCH_PARALLEL="${arg#*=}" ;;
        --bench-budget=*) BENCH_BUDGET="${arg#*=}" ;;
        --generate-ui) GEN_UI=true ;;
        --no-deploy) DEPLOY=false ;;
        --service=*) SERVICE_NAME="${arg#*=}" ;;
        *.conf) CONFIG_OVERRIDE="$arg" ;;
    esac
done

if [[ -n "$CONFIG_OVERRIDE" ]]; then
    if [[ -f "$CONFIG_OVERRIDE" ]]; then
        echo "📂 Overriding defaults with: $CONFIG_OVERRIDE"
        source "$CONFIG_OVERRIDE"
        # --- VRAM Pre-flight Check ---
        if [[ -f "$SCRIPT_DIR/vram-linter.py" ]]; then
            python3 "$SCRIPT_DIR/vram-linter.py" "$CONFIG_OVERRIDE" || true
        fi
    else
        echo "❌ Override config file not found: $CONFIG_OVERRIDE"
        exit 1
    fi
fi

# Ensure slot save path exists
if [[ -n "$SLOT_SAVE_PATH" ]]; then
    mkdir -p "$SLOT_SAVE_PATH"
fi

# =========================================================

echo "=== llama.cpp Rebuild Script (Gemma-4 Q6_K Speed Optimized) ==="

# Auto-detect llama.cpp directory
if [[ -n "${LLAMA_CPP_DIR:-}" ]]; then
    LLAMA_DIR="$LLAMA_CPP_DIR"
elif [[ -f "$SCRIPT_DIR/../CMakeLists.txt" ]]; then
    LLAMA_DIR="$SCRIPT_DIR/.."
else
    LLAMA_DIR="$(pwd)"
fi

cd "$LLAMA_DIR"
LLAMA_DIR="$(pwd)"

if [[ "$BENCH_ONLY" == true ]]; then
    echo "⏭️  Skipping restart (--bench-only mode)"
else

# 3. Graceful shutdown
echo "🔄 Stopping service..."
sudo systemctl stop "$SERVICE_NAME" || true

if [[ "$BUILD" == true ]]; then
    # 4. Update to latest
    git pull || true

    # 6. Build with best optimizations
    echo "🛠️ Building with maximum optimizations (CUDA 13.2 optimized)..."
    
    CUDA_ARGS=()
    if [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        CUDA_ARGS+=("-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc")
    fi

    cmake -B build -S . -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      "${CUDA_ARGS[@]}" \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
      -DGGML_NATIVE=ON \
      -DGGML_AVX512=ON \
      -DGGML_AVX512_VNNI=ON \
      -DGGML_AVX512_BF16=ON \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_FA=ON \
      -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DGGML_CUDA_GRAPHS=ON \
      -DGGML_CUDA_NO_PEER_COPY=OFF \
      -DGGML_CUDA_PEER_MAX_BATCH_SIZE=128 \
      -DGGML_CUDA_COMPRESSION_MODE=speed \
      -DGGML_CUDA_NO_VMM=ON \
      -DGGML_CURL=ON \
      -DGGML_OPENMP=ON \
      -DCMAKE_CUDA_ARCHITECTURES="89"

    cmake --build build --config Release -j$(nproc)
fi

if [[ "$DEPLOY" == true ]]; then
    # 7. Update systemd service
    echo "📝 Updating systemd service: $SERVICE_NAME"
    CONFIG_NAME=$(basename "${CONFIG_OVERRIDE:-script_defaults}")

    # Build ExecStart command array for robust generation
    CMD=("$LLAMA_DIR/build/bin/llama-server")
    CMD+=("--model $MODEL_PATH")
    [[ -n "$MODEL_ALIAS" ]] && CMD+=("--alias $MODEL_ALIAS")
    CMD+=("--path $LLAMA_DIR/tools/server/public")
    [[ -n "$MMPRJ_PATH" ]] && CMD+=("--mmproj $MMPRJ_PATH")
    CMD+=("--n-gpu-layers $N_GPU_LAYERS")
    CMD+=("--n-cpu-moe $N_CPU_MOE")
    CMD+=("--cache-type-k $CACHE_TYPE_K")
    CMD+=("--cache-type-v $CACHE_TYPE_V")
    [[ "$MLOCK" == "true" ]] && CMD+=("--mlock")
    [[ "$MMAP" == "false" ]] && CMD+=("--no-mmap")
    CMD+=("--parallel $PARALLEL")
    CMD+=("--cache-ram $CACHE_RAM")
    CMD+=("--cache-reuse $CACHE_REUSE")
    [[ "${KV_UNIFIED:-}" == "true" ]] && CMD+=("--kv-unified")
    [[ "${CLEAR_IDLE:-}" == "true" ]] && CMD+=("--cache-idle-slots")
    [[ "${CONTEXT_SHIFT:-}" == "true" ]] && CMD+=("--context-shift")
    CMD+=("--slot-save-path $SLOT_SAVE_PATH")
    CMD+=("--cont-batching")
    CMD+=("--threads $THREADS")
    CMD+=("--threads-batch $THREADS_BATCH")
    CMD+=("--threads-http $THREADS_HTTP")
    CMD+=("--prio $PRIORITY")
    CMD+=("--prio-batch $PRIORITY_BATCH")
    CMD+=("--numa isolate")
    CMD+=("--flash-attn on")
    CMD+=("--ctx-size $CTX_SIZE")
    CMD+=("--batch-size $BATCH_SIZE")
    CMD+=("--ubatch-size $UBATCH_SIZE")
    CMD+=("--reasoning $REASONING")
    CMD+=("--reasoning-format $REASONING_FORMAT")
    CMD+=("--reasoning-budget $REASONING_BUDGET")
    [[ -n "$REASONING_BUDGET_MESSAGE" ]] && CMD+=("--reasoning-budget-message \"$REASONING_BUDGET_MESSAGE\"")
    [[ "$JINJA" == "true" ]] && CMD+=("--jinja")
    CMD+=("--temp $TEMP")
    CMD+=("--min-p $MIN_P")
    CMD+=("--xtc-probability $XTC_PROBABILITY")
    CMD+=("--xtc-threshold $XTC_THRESHOLD")
    CMD+=("--top-p $TOP_P")
    CMD+=("--top-k $TOP_K")
    CMD+=("--repeat-last-n $REPEAT_LAST_N")
    CMD+=("--repeat-penalty $REPEAT_PENALTY")
    CMD+=("--presence-penalty $PRESENCE_PENALTY")
    CMD+=("--dry-multiplier $DRY_MULTIPLIER")
    CMD+=("--dry-base $DRY_BASE")
    CMD+=("--dry-allowed-length $DRY_ALLOWED_LENGTH")
    CMD+=("--dry-penalty-last-n $DRY_PENALTY_LAST_N")
    CMD+=("--samplers \"$SAMPLERS\"")
    CMD+=("--host $HOST")
    CMD+=("--port $PORT")
    CMD+=("--log-disable")
    [[ -n "${EXTRA_ARGS:-}" ]] && CMD+=("${EXTRA_ARGS:-}")
    CMD+=("--metrics")

    # Join array with backslashes
    EXEC_START=$(printf "  %s \\\\\n" "${CMD[@]}")
    EXEC_START=${EXEC_START% \\\\$'\n'} # Remove last backslash

    cat > /tmp/$SERVICE_NAME << EOF

[Unit]
Description=Llama.cpp Server - Config: $CONFIG_NAME
After=network.target

[Service]
Type=simple
User=siva
CPUAffinity=$CPU_AFFINITY
LimitMEMLOCK=infinity

ExecStart=$EXEC_START

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

    sudo cp /tmp/$SERVICE_NAME /etc/systemd/system/$SERVICE_NAME
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    sudo systemctl restart $SERVICE_NAME

    echo "✅ Service restarted."
else
    echo "⏭️  Skipping deployment (--no-deploy mode)"
fi

if [[ "$GEN_UI" == true ]]; then
    echo "🎨 Regenerating OpenWeb-UI profiles..."
    python3 "$SCRIPT_DIR/generate-ui-profiles.py"
fi

echo "Binary path : $LLAMA_DIR/build/bin/llama-server"
echo "Model       : $MODEL_PATH"
fi

if [[ "$BENCH" == true ]]; then
    echo "🏎️ Running benchmark..."
    MODEL_NAME="${MODEL_ALIAS:-$(basename "$MODEL_PATH")}"
    OUTPUT=$(python3 "$SCRIPT_DIR/bench-llama.py" "$BENCH_COUNT" -p "$BENCH_PARALLEL" --port "$PORT" --model "$MODEL_NAME" --budget "$BENCH_BUDGET")
    echo "$OUTPUT"

    # --- Performance Comparison Logic ---
    BASELINES_FILE="$SCRIPT_DIR/baselines.json"
    CONFIG_NAME=$(basename "${CONFIG_OVERRIDE:-script_defaults}")

    if [[ -f "$BASELINES_FILE" ]]; then
        # Parse current results
        CUR_AVG_GEN=$(echo "$OUTPUT" | grep "Generation" | tail -n 1 | awk '{print $4}')
        CUR_AGG_THR=$(echo "$OUTPUT" | grep "Throughput" | tail -n 1 | awk '{print $3}')

        if [[ -n "$CUR_AVG_GEN" && -n "$CUR_AGG_THR" ]]; then
            python3 - << EOF
import json
import os

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
        
        print(f"\n📈 Performance Comparison vs Golden Baseline ({base['timestamp']}):")
        
        def fmt_diff(diff):
            color = "\033[92m" if diff >= -2 else ("\033[93m" if diff >= -10 else "\033[91m")
            reset = "\033[0m"
            return f"{color}{diff:+.1f}%{reset}"

        print(f"   - Avg Generation: {base['avg_gen']:.1f} -> {cur_gen:.1f} tok/s ({fmt_diff(gen_diff)})")
        print(f"   - Agg Throughput: {base['agg_thr']:.1f} -> {cur_thr:.1f} tok/s ({fmt_diff(thr_diff)})")
        
        if thr_diff < -10:
            print("\n⚠️  WARNING: PERFORMANCE REGRESSION DETECTED (>10% drop in throughput!)")
            print("   Review recent upstream changes or hardware thermal state.")
    else:
        print(f"\nℹ️  No golden baseline found for {config_name}. Run ./scripts-local/save-baseline.sh to set one.")
except Exception as e:
    print(f"\n❌ Error comparing baselines: {e}")
EOF
        fi
    fi
fi
