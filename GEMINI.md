# Workspace Core Mandates & Engineering Standards

This file contains foundational mandates, architectural patterns, and hard-won lessons specific to this `llama.cpp` and OpenWeb-UI deployment repository. These rules take absolute precedence over general defaults.

## 1. System Architecture & Deployment

- **Deployment Script**: Always use `./scripts-local/rebuild-llama.sh [CONFIG_FILE]` to deploy or update the server. Never launch `llama-server` manually without the wrapper script.
  - **VRAM Pre-flight**: Automatically checks if your config fits in the 16GB RTX 4070 Ti Super before starting. Aborts if >15.5GB.
  - **Manual UI Sync**: OpenWeb-UI JSON profiles are no longer regenerated automatically. Use the `--generate-ui` flag in `rebuild-llama.sh` to trigger updates.
- **Active Configurations**:
  - `qwen-3.6-35b-a3b.conf`: APEX-I-Balanced (Port 8080 | CCD0 focus).
  - `gemma-4-26b-a4b.conf`: PRISM-PRO-DQ (Port 8081 | CCD1 focus).
- **Upstream Sync**: Use `./scripts-local/sync-fork.sh` to synchronize the local fork with the upstream repository and rebase active branches.

## 2. Hardware Calibration (16GB VRAM | AMD 7950X3D)

When tuning `.conf` files for MoE models, respect these physical limits and findings:
- **CPU Affinity Strategy**:
  - **Mixed Affinity (Preferred for Burst)**: Use cross-CCD strings like `0-7,24-31` (Qwen) and `16-23,8-15` (Mythos) to maximize single-model burst performance (up to 71 tok/s for 35B).
  - **Strict Isolation (Preferred for Multitasking)**: Use strict ranges like `0-7,16-23` (CCD0) and `8-15,24-31` (CCD1) to reduce prompt evaluation stutter by ~24% during simultaneous load.
- **CUDA Optimization**:
  - **Peer Copy**: Always build with `-DGGML_CUDA_NO_PEER_COPY=OFF`. This is critical for MoE models when experts are split between VRAM and System RAM (DDR5), significantly reducing expert-switching latency.
  - **Binary Compression**: For CUDA 12.8+, use `-DGGML_CUDA_COMPRESSION_MODE=speed` to optimize link times and kernel performance.
  - **Architectures**: Set `CMAKE_CUDA_ARCHITECTURES="89"` for the RTX 4070 Ti Super.
- **VRAM Safeguards**:
  - Max combined Context + Weights must not exceed 15.5GB.
  - Use `CACHE_TYPE_K="q4_0"` and `CACHE_TYPE_V="q4_0"` for maximum context (128K+) on 16GB cards.
  - For Qwen 35B, use `N_CPU_MOE=32` to keep most experts on GPU while fitting in VRAM.

## 3. The 10/10 UI/UX Standard for OpenWeb-UI Profiles

When generating or editing JSON profiles, you MUST strictly adhere to these standards:

1. **Header Scaling**: Use `###` (Header 3) as the maximum header size. Never use `##` or `#`.
2. **F-Scan Anchoring**: Always place emojis at the **absolute start** of the header (e.g., `### 🏗️ Engineering`) to enable vertical marginal scanning.
3. **Mandatory Breathing Room**: Every system prompt must include: **"Use double-newlines and keep paragraphs under 3 lines."**
4. **The FIM Mission**: Instant/FIM profiles must include: `"Predict the next most likely tokens. No conversational filler."`
5. **Type Standardization**: All numerical parameters (budgets, penalties, samplers) must be explicitly defined as **floats** (e.g., `1024.0`, `1.05`) for OpenWeb-UI parser robustness.
6. **ID Hygiene**: Use short, lowercase, hyphenated IDs starting with model type (e.g., `qw3-35b-fast`, `gemma4-26b-pro`).
7. **The "No-Leak" Mandate**: For reasoning personas, use the phrase: **"Deliver the final solution only; strictly exclude internal reasoning traces from this response block."**

## 4. Reasoning & Sampler Mechanics (MoE / APEX Optimization)

- **APEX-I-Balanced Sampling (Qwen 3.6)**:
  - **Temperature**: Maintain `0.40 - 0.55`. Lowering below `0.25` causes MoE "Expert Stagnation."
  - **Min-P**: Use `0.05 - 0.1` to filter quantization noise while preserving the APEX imatrix diversity.
  - **Repetition**: Use `repeat_penalty: 1.05` with `repeat_last_n: 256.0` to prevent MoE "End-of-Thought" stutter.
- **Reasoning Budget Enforcement**:
  - **Format**: Always use `REASONING_FORMAT="auto"` for non-DeepSeek models to ensure native tag detection.
  - **Stop-Anchor**: Use `REASONING_BUDGET_MESSAGE=" [Logic Finalized] "` in `.conf` files to provide a graceful termination sequence.
  - **Jinja Hygiene**: Set `preserve_thinking: false` in chat template kwargs (or omit it) to allow the C++ engine to natively manage and hide thinking tokens. This is required for `thinking_budget_tokens` to be respected.
- **Standard Sampler Chains**:
  - **Creative/Pro**: `"samplers": "dry;top_p;temperature"`
  - **Logic/Coder**: `"samplers": "dry;top_k;min_p"`
  - **DRY Sampler**: Always include DRY (`multiplier=0.8`, `base=1.75`) to prevent repetitive MoE loops.
