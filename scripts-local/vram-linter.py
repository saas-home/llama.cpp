#!/usr/bin/env python3
import sys
import os
import subprocess

# Constants for Gemma-4 26B-A4B MoE
GEMMA4_LAYERS = 30
GEMMA4_KV_HEADS = 8
GEMMA4_HEAD_DIM = 128
GEMMA4_EXPERTS = 128

# Constants for Qwen3.6 35B-A3B MoE
QWEN36_LAYERS = 40
QWEN36_KV_HEADS = 2
QWEN36_HEAD_DIM = 256
QWEN36_EXPERTS = 256

VRAM_LIMIT_MB = 15872 # 15.5 GB Safety Threshold

# Approx Weights (MB) for Gemma-4 26B-A4B
GEMMA4_WEIGHTS = {
    "q6_k": {"base": 3000, "expert": 15500},
    "q4_0": {"base": 2100, "expert": 12500},
    "prism-dq": {"base": 2000, "expert": 12000},
    "apex-i": {"base": 3500, "expert": 16500},
    "default": {"base": 2500, "expert": 14000}
}

# Approx Weights (MB) for Qwen3.6 35B-A3B (Scaled for ~22GB APEX-I)
QWEN36_WEIGHTS = {
    "q6_k": {"base": 4200, "expert": 21000},
    "q4_0": {"base": 3000, "expert": 17000},
    "prism-dq": {"base": 2800, "expert": 16500},
    "apex-i": {"base": 5000, "expert": 23000},
    "default": {"base": 3500, "expert": 19000}
}

def parse_bash_conf(file_path):
    try:
        cmd = f"set -a; source {file_path}; env"
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True)
        return {line.split("=", 1)[0]: line.split("=", 1)[1] for line in result.stdout.splitlines() if "=" in line}
    except: return {}

def estimate_vram(conf):
    model_path = conf.get("MODEL_PATH", "").lower()
    
    # Architecture Detection
    if "qwen3.6" in model_path or "qwen3" in model_path:
        layers = QWEN36_LAYERS
        kv_heads = QWEN36_KV_HEADS
        head_dim = QWEN36_HEAD_DIM
        total_experts = QWEN36_EXPERTS
        weight_table = QWEN36_WEIGHTS
        arch_name = "Qwen3.6-35B-A3B"
    else:
        layers = GEMMA4_LAYERS
        kv_heads = GEMMA4_KV_HEADS
        head_dim = GEMMA4_HEAD_DIM
        total_experts = GEMMA4_EXPERTS
        weight_table = GEMMA4_WEIGHTS
        arch_name = "Gemma-4-26B-A4B"

    # 1. Determine Quant Type
    quant = "default"
    if "q6_k" in model_path: quant = "q6_k"
    elif "q4_0" in model_path: quant = "q4_0"
    elif "prism-pro-dq" in model_path: quant = "prism-dq"
    elif "apex-i" in model_path: quant = "apex-i"
    
    weights = weight_table[quant]
    
    # 2. Model Weight VRAM (Offloaded)
    n_cpu_moe = int(conf.get("N_CPU_MOE", 12)) # Experts on CPU
    
    # Base layers + Experts on GPU ratio
    gpu_base = weights["base"]
    gpu_experts = weights["expert"] * (total_experts - n_cpu_moe) / total_experts
    total_weights_vram = gpu_base + gpu_experts
    # 3. KV Cache VRAM
    ctx_size = int(conf.get("CTX_SIZE", 131072))
    cache_type_k = conf.get("CACHE_TYPE_K", "q8_0")
    
    # Precision bytes (Approx)
    precision = 1.0 # q8_0
    if "f16" in cache_type_k: precision = 2.0
    elif "q4_0" in cache_type_k: precision = 0.5
    
    # KV per token = 2 * layers * kv_heads * head_dim * precision
    kv_per_token_bytes = 2 * layers * kv_heads * head_dim * precision
    total_kv_vram_mb = (ctx_size * kv_per_token_bytes) / (1024 * 1024)
    
    # 4. Vision (mmproj) + Overhead
    vision_vram = 800 if conf.get("MMPRJ_PATH") else 0
    overhead = 600
    
    total_mb = total_weights_vram + total_kv_vram_mb + vision_vram + overhead
    
    return {
        "total": total_mb,
        "weights": total_weights_vram,
        "kv": total_kv_vram_mb,
        "vision": vision_vram,
        "quant": quant,
        "limit": VRAM_LIMIT_MB,
        "arch": arch_name
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: vram-linter.py <config.conf>")
        sys.exit(1)
        
    conf = parse_bash_conf(sys.argv[1])
    est = estimate_vram(conf)
    
    print(f"🔍 VRAM Pre-flight Check: {os.path.basename(sys.argv[1])}")
    print(f"   - Arch Detection  : {est['arch']}")
    print(f"   - Quant Detection : {est['quant'].upper()}")
    print(f"   - Model Weights   : {est['weights']:.0f} MB")
    print(f"   - KV Cache        : {est['kv']:.0f} MB (Ctx: {conf.get('CTX_SIZE')})")
    print(f"   - Vision/Overhead : {est['vision'] + 600:.0f} MB")
    print(f"   ---------------------------")
    print(f"   - Estimated Total : {est['total']:.0f} MB / {est['limit']} MB")
    
    if est['total'] > est['limit']:
        print(f"\n❌ FATAL: Estimated VRAM usage exceeds safety limit!")
        print(f"   Difference: {est['total'] - est['limit']:.0f} MB")
        sys.exit(1)
    else:
        print(f"✅ VRAM Safety Check Passed.")

if __name__ == "__main__":
    main()
