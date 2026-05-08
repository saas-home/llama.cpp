# Technical Optimization Profile: Qwen 3.6 35B-A3B-APEX-I-Balanced

## 1. Documentation & Primary Sources
This profile is synthesized from the following official technical specifications:
*   **APEX Implementation:** [mudler/Qwen3.6-35B-A3B-APEX-GGUF](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/raw/main/README.md)
*   **Model Weights/Specs:** [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/README.md)
*   **APEX Methodology:** [mudler/apex-quant](https://github.com/mudler/apex-quant/blob/main/README.md)

---

## 2. Model Architecture & Deployment Rationale

### **Model Overview**
*   **Base:** Qwen 3.6 35B-A3B (256 Experts).
*   **Specialization:** Optimized for "Thinking Mode" reasoning, agentic coding (terminal/repo-level), and Model Context Protocol (MCP) tool-calling.
*   **Context Limit:** 262,144 (Native). 
*   **Deployment Goal:** Stable 128K context window on 16GB VRAM (RTX 4070 Ti Super) with peak generation speed.

### **Hardware Calibration (16GB RTX 4070 Ti Super + 7950X3D)**
*   **CCD Focus:** `CPU_AFFINITY="0-7,24-31"` (CCD0 V-Cache) ensures logic-heavy MoE branch routing stays on the high-cache CCD.
*   **Expert Offloading:** `N_CPU_MOE=192` (64 experts on GPU, 192 on CPU).
    *   *Finding:* Values < 192 are silently overridden by `llama-server` to fit in 16GB VRAM.
*   **Memory Locks:** `MLOCK=true` is mandatory to eliminate page-fault latency spikes when the router hits CPU-offloaded experts.
*   **KV Cache:** `CTX_SIZE=131072` (128K) with `q4_0` K/V types fits within the ~15GB safety limit.

---

## 3. APEX (Adaptive Precision for EXpert Models) Strategy

### **Quantization Details (Extracted from mudler/apex-quant)**
*   **Layer-wise Precision Gradient:** Not all layers are equal. Sensitivity is non-linear across the 40 blocks.
*   **Edge Layers:** First and last 5 layers are most sensitive; kept at higher precision (e.g., Q6_K).
*   **Shared Experts:** Must be at least **Q8_0** to maintain routing stability and prevent "logic drift."
*   **Imatrix (APEX-I):** Uses a diverse calibration dataset (chat, code, reasoning, tool-calling) to trade negligible perplexity for significant gains in real-world accuracy.
*   **Tiering:** APEX-I Balanced (24GB GGUF) is the recommended tier for high-accuracy local deployment on 16GB-24GB cards.

---

## 4. llama.cpp Server Configuration & Findings

### **Thinking Mode Parameters (Official Qwen 3.6 Specs)**
*   **Thinking mode** is enabled by default to generate reasoning traces within `<think>` blocks.
*   **Reasoning Format:** `deepseek` (Default). Maps `<think>` blocks to `reasoning_content` in API responses.
*   **Agentic Continuity:** `preserve_thinking: true` and `enable_thinking: true` must be enabled via `--chat-template-kwargs`.
    *   *Source Detail:* `preserve_thinking` allows the model to retain reasoning context from historical messages, which is critical for decision consistency in multi-turn agentic workflows.
*   **Reasoning Budget:** `REASONING_BUDGET=-1` (Infinite) to prevent truncating complex reasoning traces.

### **Sampling (Official Qwen 3.6 "General Thinking" Specs)**
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `TEMP` | 1.0 | Standard randomness for MoE diversity. |
| `TOP_P` | 0.95 | Nucleus sampling. |
| `TOP_K` | 20 | Tight top-K for reasoning precision. |
| `MIN_P` | 0.0 | Official recommendation for general thinking tasks. |
| `REPEAT_PENALTY` | 1.0 | Official spec recommendation (no penalty). |
| `PRESENCE_PENALTY` | 1.5 | High penalty to force creative topic expansion. |
| `SAMPLERS` | `dry;top_k;top_p;min_p;temperature` | Optimized chain order. |

---

## 5. Benchmark Baselines (128K Context)

| Slot Configuration | Generation Speed (avg) | Aggregate Throughput |
| :--- | :--- | :--- |
| **1-Slot (Peak)** | **55.9 tok/s** | 55.9 tok/s |
| **3-Slot (Concurrent)** | **26.7 tok/s** | **75.4 tok/s** |

---

## 6. Maintenance & Sync Notes
*   **Update Script:** Always use `./scripts-local/rebuild-llama.sh scripts-local/qwen-3.6-35b-a3b.conf`.
*   **UI Sync:** The script automatically generates the OpenWeb-UI JSON profile with the mandatory `preserve_thinking` flags.
*   **Baseline File:** Baselines are stored in `scripts-local/baselines.json`.
