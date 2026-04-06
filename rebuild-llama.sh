#!/bin/bash
set -euo pipefail

# ====================== CONFIGURATION ======================
SERVICE_NAME="llama-server.service"
#MODEL_PATH="/home/siva/models/Nemotron-Cascade-2/Nemotron-Cascade-2-30B-A3B-MXFP4_MOE_BF16.gguf"
#MODEL_PATH="/home/siva/models/Nemotron-Cascade-2/Nemotron-Cascade-2-30B-A3B.i1-Q5_K_M.gguf"
#MODEL_PATH="/home/siva/models/gemma-4-26B-A4B-unsloth/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf"
#MMPRJ_PATH="/home/siva/models/gemma-4-26B-A4B-noctrex-apex/gemma-4-26B-A4B-it-mmproj-F32.gguf"
#MODEL_PATH="/home/siva/models/gemma-4-26B-A4B-noctrex-apex/gemma-4-26B-A4B-it-MXFP4_MOE_F16.gguf"

MMPRJ_PATH="/home/siva/models/gemma-4-26B-A4B-mudler-apex/mmproj-F16.gguf"
MODEL_PATH="/home/siva/models/gemma-4-26B-A4B-mudler-apex/gemma-4-26B-A4B-APEX-I-Quality.gguf"

# =========================================================

echo "=== llama.cpp Rebuild Script (Final Version) ==="

# 1. Auto-detect llama.cpp directory
if [[ -n "${LLAMA_CPP_DIR:-}" ]]; then
    LLAMA_DIR="$LLAMA_CPP_DIR"
elif [[ -f "$(dirname "$0")/CMakeLists.txt" ]]; then
    LLAMA_DIR="$(dirname "$0")"
elif [[ -f "$(pwd)/CMakeLists.txt" ]]; then
    LLAMA_DIR="$(pwd)"
else
    echo "❌ Could not detect llama.cpp directory."
    echo "   Run from inside llama.cpp or set LLAMA_CPP_DIR=/path/to/llama.cpp"
    exit 1
fi

echo "✅ Using llama.cpp directory: $LLAMA_DIR"
cd "$LLAMA_DIR"
LLAMA_DIR="$(pwd)"    # canonicalize to absolute path

# 2. CPU affinity (0-31 tested: same perf as 16-31, less constrained for 34 MoE threads)
CPU_AFFINITY="0-15"

# 3. Graceful shutdown
echo "🔄 Stopping service..."
sudo systemctl stop "$SERVICE_NAME" || true

# 4. Update to latest
DEFAULT_BRANCH=$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null | sed 's@^refs/remotes/origin/@@') || DEFAULT_BRANCH="master"
echo "📥 Pulling latest $DEFAULT_BRANCH..."
git checkout "$DEFAULT_BRANCH"
git pull origin "$DEFAULT_BRANCH"

# 5. Build tools (install only if missing)
MISSING_PKGS=()
for pkg in ninja ccache; do
    pacman -Q "$pkg" &>/dev/null || MISSING_PKGS+=("$pkg")
done
if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
    echo "📦 Installing missing packages: ${MISSING_PKGS[*]}"
    sudo pacman -Sy --needed --noconfirm "${MISSING_PKGS[@]}"
fi
export PATH="/usr/lib/ccache/bin:$PATH"
export CCACHE_DIR=~/.ccache

# 6. Build with best optimizations
echo "🛠️ Building with maximum optimizations..."

# Navigate to source and wipe everything
rm -rf build/
find . -name "CMakeCache.txt" -delete

cmake -B build -S . -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_AVX512=ON \
  -DGGML_AVX512_VNNI=ON \
  -DGGML_AVX512_VBMI=ON \
  -DGGML_NATIVE=ON \
  -DGGML_CURL=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89"

cmake --build build --config Release -j$(nproc)

# 7. Update systemd service
echo "📝 Updating systemd service..."
cat > /tmp/llama-server.service << EOF

[Unit]
Description=Llama.cpp Server - Gemma 4 26B-A4B
After=network.target

[Service]
Type=simple
User=siva
CPUAffinity=0-7
LimitMEMLOCK=infinity

ExecStart=$LLAMA_DIR/build/bin/llama-server \\
  --model $MODEL_PATH \\
  --mmproj $MMPRJ_PATH \\
  --fit on \\
  --fit-target 1536 \\
  --mmap \\
  --mlock \\
  --parallel 2 \\
  --cache-ram 8192 \\
  --cont-batching \\
  --cache-type-k q4_0 \\
  --cache-type-v q4_0 \\
  --threads 12 \\
  --numa isolate \\
  --flash-attn on \\
  --ctx-size 65536 \\
  --batch-size 512 \\
  --ubatch-size 512 \\
  --reasoning-budget 1024 \\
  --reasoning-budget-message " [Thinking budget reached, concluding logic...] " \\
  --jinja \\
  --temp 1.0 \\
  --min-p 0.05 \\
  --top-p 0.95 \\
  --top_k 64 \\
  --repeat_last_n 64 \\
  --repeat_penalty 1.05 \\
  --samplers "top_k;top_p;min_p;temperature" \\
  --host 0.0.0.0 \\
  --port 8080 \\
  --metrics \\
  --sleep-idle-seconds 300

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo cp /tmp/llama-server.service /etc/systemd/system/$SERVICE_NAME
sudo systemctl daemon-reload
sudo systemctl restart $SERVICE_NAME

echo "✅ Rebuild complete! Service restarted."
echo "Binary path : $LLAMA_DIR/build/bin/llama-server"
echo "CPU Affinity: $CPU_AFFINITY"
echo ""
echo "Check status : systemctl status $SERVICE_NAME"
echo "Check logs   : journalctl -u $SERVICE_NAME -f"
