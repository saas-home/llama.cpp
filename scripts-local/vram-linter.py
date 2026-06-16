#!/usr/bin/env python3
"""VRAM pre-flight checker for llama.cpp services.

Reads actual GPU VRAM and model file size, then estimates KV cache and
CUDA overhead to determine if the service will fit on the target GPU.
"""
import sys
import os
import subprocess
import re

# --- Model constants ---
GEMMA4_LAYERS = 30
GEMMA4_KV_HEADS = 8
GEMMA4_HEAD_DIM = 128
GEMMA4_EXPERTS = 128

QWEN36_LAYERS = 40
QWEN36_KV_HEADS = 2
QWEN36_HEAD_DIM = 256
QWEN36_EXPERTS = 256

# Safety threshold: fraction of total VRAM to leave free for system/OS
# 90% leaves ~1.6 GB headroom on 16 GB GPU for display server, other processes
SAFETY_FRACTION = 0.90

# CUDA allocator overhead multiplier (workspace buffers, fragmentation)
CUDA_OVERHEAD = 1.08


def get_gpu_vram_mb(gpu_id=0):
    """Query actual GPU total VRAM via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
            capture_output=True, text=True, check=True
        )
        total = result.stdout.strip()
        if total:
            return float(total)
    except Exception:
        pass
    # Fallback: 16 GB if nvidia-smi unavailable
    return 16384.0


def get_model_file_size_mb(model_path):
    """Read actual model file size from disk."""
    if not model_path or not os.path.isfile(model_path):
        return 0.0
    return os.path.getsize(model_path) / (1024 * 1024)


def parse_bash_conf(file_path):
    try:
        cmd = f"set -a; source {file_path}; env"
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True)
        return {line.split("=", 1)[0]: line.split("=", 1)[1] for line in result.stdout.splitlines() if "=" in line}
    except Exception:
        return {}


def estimate_kv_cache_mb(ctx_size, layers, kv_heads, head_dim, cache_type_k, parallel, kv_unified, cache_idle_slots):
    """Estimate KV cache VRAM in MB.

    llama.cpp does NOT pre-allocate full KV for all parallel slots.
    With kv-unified + cache-idle-slots, idle slots are offloaded to RAM,
    so only the active slot(s) consume VRAM at any given time.
    """
    precision = 1.0  # q8_0 default
    if "f16" in cache_type_k:
        precision = 2.0
    elif "q4_0" in cache_type_k:
        precision = 0.5
    elif "q6_k" in cache_type_k:
        precision = 0.75

    kv_per_token_bytes = 2 * layers * kv_heads * head_dim * precision
    total_kv_bytes = ctx_size * kv_per_token_bytes

    # With kv-unified + cache-idle-slots: only 1 slot active in VRAM at a time
    # The rest are offloaded to RAM via prompt cache
    if kv_unified == "true" and cache_idle_slots == "true":
        slot_multiplier = 1
    else:
        # Without idle-slot offload, all parallel slots need VRAM
        slot_multiplier = max(parallel, 1)

    total_kv_bytes *= slot_multiplier
    return total_kv_bytes / (1024 * 1024)


def estimate_vram(conf, gpu_vram_mb):
    model_path = conf.get("MODEL_PATH", "")
    n_gpu_layers = int(conf.get("N_GPU_LAYERS", 999))
    n_cpu_moe = int(conf.get("N_CPU_MOE", 0))

    # Architecture detection from model path
    model_lower = model_path.lower()
    if "qwen3.6" in model_lower or "qwen3" in model_lower:
        layers = QWEN36_LAYERS
        kv_heads = QWEN36_KV_HEADS
        head_dim = QWEN36_HEAD_DIM
        total_experts = QWEN36_EXPERTS
        arch_name = "Qwen3.6-35B-A3B"
    else:
        layers = GEMMA4_LAYERS
        kv_heads = GEMMA4_KV_HEADS
        head_dim = GEMMA4_HEAD_DIM
        total_experts = GEMMA4_EXPERTS
        arch_name = "Gemma-4-26B-A4B"

    # 1. Actual model file size on disk
    model_size_mb = get_model_file_size_mb(model_path)

    # 2. Model VRAM: apply n-gpu-layers and n-cpu-moe ratio
    # If n_gpu_layers == 999, all layers go to GPU
    if n_gpu_layers >= layers:
        weights_vram_mb = model_size_mb
    elif n_gpu_layers > 0:
        # Proportional: GPU gets n_gpu_layers/layers of the model
        weights_vram_mb = model_size_mb * n_gpu_layers / layers
    else:
        weights_vram_mb = 0.0

    # 3. KV cache VRAM (accounts for parallel slots + idle-slot offload)
    ctx_size = int(conf.get("CTX_SIZE", 131072))
    cache_type_k = conf.get("CACHE_TYPE_K", "q8_0")
    parallel = int(conf.get("PARALLEL", 1))
    kv_unified = conf.get("KV_UNIFIED", "false")
    cache_idle_slots = conf.get("CACHE_IDLE_SLOTS", "false")
    kv_vram_mb = estimate_kv_cache_mb(ctx_size, layers, kv_heads, head_dim, cache_type_k, parallel, kv_unified, cache_idle_slots)

    # 4. Vision (mmproj) + CUDA overhead
    vision_vram = 800 if conf.get("MMPRJ_PATH") else 0
    overhead = 200  # base overhead (server, HTTP, etc.)

    # CUDA overhead applies to KV/workspace buffers, NOT model weights
    # (quantized GGUF maps 1:1 to GPU memory)
    kv_with_overhead = kv_vram_mb * CUDA_OVERHEAD

    # Base GPU usage: model + minimal KV (incremental allocation ~25% of full)
    base_kv_mb = kv_vram_mb * 0.25
    base_vram_mb = weights_vram_mb + base_kv_mb * CUDA_OVERHEAD + vision_vram + overhead

    # Peak GPU usage: model + full KV + overhead
    peak_vram_mb = weights_vram_mb + kv_with_overhead + vision_vram + overhead

    safe_limit_mb = gpu_vram_mb * SAFETY_FRACTION

    return {
        "base": base_vram_mb,
        "peak": peak_vram_mb,
        "weights": weights_vram_mb,
        "kv": kv_vram_mb,
        "cuda_overhead": kv_vram_mb * (CUDA_OVERHEAD - 1),
        "vision": vision_vram,
        "overhead": overhead,
        "gpu_vram": gpu_vram_mb,
        "safe_limit": safe_limit_mb,
        "arch": arch_name,
        "model_file_mb": model_size_mb,
        "parallel": parallel
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: vram-linter.py <config.conf>")
        sys.exit(1)

    conf = parse_bash_conf(sys.argv[1])
    gpu_vram = get_gpu_vram_mb()
    est = estimate_vram(conf, gpu_vram)

    print(f"VRAM Pre-flight Check: {os.path.basename(sys.argv[1])}")
    print(f"   - GPU               : {est['gpu_vram']:.0f} MB total")
    print(f"   - Arch Detection    : {est['arch']}")
    print(f"   - Model File        : {est['model_file_mb']:.0f} MB ({os.path.basename(conf.get('MODEL_PATH', ''))})")
    print(f"   - Weights on GPU    : {est['weights']:.0f} MB")
    print(f"   - KV Cache (1 slot) : {est['kv']:.0f} MB (Ctx: {conf.get('CTX_SIZE')}, Parallel: {est['parallel']})")
    print(f"   - CUDA Overhead     : {est['cuda_overhead']:.0f} MB (+{int((CUDA_OVERHEAD-1)*100)}%)")
    print(f"   - Vision/Overhead   : {est['vision'] + est['overhead']:.0f} MB")
    print(f"   ---------------------------")
    print(f"   - Base GPU Use      : {est['base']:.0f} MB / {est['gpu_vram']:.0f} MB ({est['base']/est['gpu_vram']*100:.1f}%)")
    print(f"   - Peak GPU Use      : {est['peak']:.0f} MB / {est['gpu_vram']:.0f} MB ({est['peak']/est['gpu_vram']*100:.1f}%)")
    print(f"   - Safety Threshold  : {est['safe_limit']:.0f} MB ({SAFETY_FRACTION*100}%)")

    if est['base'] > est['safe_limit']:
        print(f"\nFATAL: Model weights alone exceed {SAFETY_FRACTION*100}% safety threshold!")
        print(f"   Excess: {est['base'] - est['safe_limit']:.0f} MB")
        print(f"   Service will fail to load. Use a smaller quantization or reduce --n-gpu-layers.")
        sys.exit(1)
    elif est['peak'] > est['safe_limit']:
        print(f"\nWARNING: Peak GPU usage ({est['peak']:.0f} MB) exceeds {SAFETY_FRACTION*100}% threshold ({est['safe_limit']:.0f} MB).")
        print(f"   Excess: {est['peak'] - est['safe_limit']:.0f} MB")
        print(f"   Service will start but may OOM with long contexts. Consider reducing CTX_SIZE.")
        sys.exit(1)
    else:
        print(f"\nOK: GPU usage within {SAFETY_FRACTION*100}% safety threshold.")


if __name__ == "__main__":
    main()
