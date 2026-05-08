---
name: llm-stack-optimizer
description: Generate optimized llama-server configs and OpenWeb-UI JSON profiles by researching model specs, benchmarks, and cloning from established high-quality templates.
---

# LLM Stack Optimizer

This skill performs deep research and analysis to generate a perfectly tuned configuration for any new LLM on your hardware, using established high-quality templates as a base.

## Input
- **Base Model Name**: (e.g., `google_gemma-4-26B-A4B-it-Q6_K.gguf`)

## Workflow

### 1. Preparation Phase (Upstream Sync)
- **Sync Check**: Ask the user if they want to synchronize the local repository with upstream before proceeding.
- **Action**: If yes, execute `./scripts-local/sync-fork.sh`. 
- **Build Requirement**: If a sync was performed, the deployment step **MUST** include the `--build` flag in `rebuild-llama.sh` to update the server binary.

### 2. Intelligence Phase (Deep Research & Architecture)
- **Search Objective**: Perform a comprehensive Google search for the model's official release notes, HuggingFace model card, and benchmark reports (PPL, MMLU, HumanEval).
- **Architecture Analysis**: Identify if the model is Dense or MoE (Mixture of Experts), number of layers, and attention mechanisms (GQA/MQA).
- **Research Summary**: Provide a detailed technical report including:
    - **Model Details**: Architecture, parameter count, expert count (if MoE).
    - **Performance**: Key benchmark scores (HumanEval, MMLU).
    - **Sources**: List the URLs/citations used for the data.
    - **Llama-Server Specs**: Proposed `llama-server` flags based on the architecture (e.g., `--reasoning`, `--chat-template`).
- **User Input**: **Ask the user if they have additional resources** (local documents, URLs, or notes) to refine the config.
- **Confirmation**: **Ask for explicit user confirmation** to proceed to creating the `.conf` and `.json` files.

### 3. Physical Phase (Hardware Constraints)
- **VRAM Verification**: Use `python3 scripts-local/vram-linter.py [TEMP_CONFIG]` to validate usage for the 16GB RTX 4070 Ti Super.
- **Safety Threshold**: Target < 15.5GB total usage. If the linter fails, increase `N_CPU_MOE` or decrease `CTX_SIZE`.
- **MoE Expert Offloading**: 
    - **Stability baseline (Gemma-4/128-expert)**: 112-120 experts on CPU (out of 128).
    - **Stability baseline (Qwen 3.6/256-expert)**: 240-244 experts on CPU (out of 256).
- **Context Scaling (DeltaNet)**: For Hybrid DeltaNet models (Qwen 3.6), `CTX_SIZE=262144` is achievable on 16GB VRAM due to linear-attention efficiency (75% of layers).
- **Affinity**: Always use `CPU_AFFINITY="0-15"` and `THREADS=16` for the 7950X3D.

### 4. Generation Phase (Templated Cloning)
- **llama-server Config**: 
    - **Base**: Clone from `scripts-local/gemma-4-26b-a4b-prism-pro-dq.gguf.conf`.
    - **Mandatory Flags**: `--cache-reuse 256`, `--prio 2`, `--context-shift`, `--kv-unified`.
    - **Qwen 3.6 / Agentic Specifics**:
        - Include `--chat-template-kwargs '{"preserve_thinking":true}'` for agentic preservation.
        - **Pristine History**: Set `REASONING_BUDGET_MESSAGE=""` (empty) to prevent non-native filler from disrupting recursive logic.
        - **Watchdog**: Set a global `REASONING_BUDGET=2048` to prevent infinite thinking loops.
- **OpenWeb-UI Profiles**: 
    - **Automation**: Do NOT manually create JSON profiles. Execute `python3 scripts-local/generate-ui-profiles.py` after saving the `.conf` file to generate all 8 standard personas (FIM, Coder, Pro, etc.).
    - **Standards**: The script automatically enforces the **10/10 UI Standard** (H3 headers, emoji anchoring, float types).
    - **Recursive Logic**: Ensure high-logic personas (Pro/Coder/Research) instruct the model to "recursively validate against previous reasoning traces."
    - **Tool Calling**: For Qwen 3.6 Coder models, ensure the persona uses the XML-based `qwen3_coder` format (detected automatically by modern llama-server).

### 5. Review & Confirmation Phase (Pre-Deployment)
- **Technical Summary**: Present a final summary of the proposed stack:
    - **Model**: [Model Name & Quant]
    - **VRAM**: [Estimated Usage] MB / 15.5 GB
    - **Compute**: [N_CPU_MOE] experts on CPU, [N_GPU_LAYERS] layers on GPU.
    - **Context**: [CTX_SIZE] (Total) / [PARALLEL] (Slots).
    - **Reasoning**: [Budget/Format settings].
- **Build Notification**: State if a full rebuild is triggered (mandatory if sync was performed).
- **Mandatory Stop**: **Ask for explicit user confirmation** before proceeding to deployment.

### 6. Integration Phase (Deployment & Health Check)
- **Deployment**: Execute `./scripts-local/rebuild-llama.sh scripts-local/[CONFIG_NAME].conf` (include `--build` if sync was performed).
- **Initialization Delay**: **Wait 10 seconds** to allow the model to load into VRAM and the KV cache to initialize.
- **Health Check**: Perform a health check via `curl -s http://localhost:8080/health`. 
- **Verification**: Ensure the response returns `{"status": "ok"}`. If the service is not healthy, check `systemctl status llama-server` and report the error.

### 7. Deployment & Validation Phase (Benchmarking & Baseline)
- **Single-Slot Benchmark**: Run `python3 scripts-local/bench-llama.py 1 -p 1` to measure peak single-stream generation speed.
- **Parallel-Slot Benchmark**: Run `python3 scripts-local/bench-llama.py 1 -p [PARALLEL]` (where `PARALLEL` matches the value in the `.conf` file) to measure aggregate throughput under load.
- **Establish Baseline**: Once performance is optimized and stable across both tests, run `./scripts-local/save-baseline.sh scripts-local/[CONFIG_NAME].conf` to set the **Golden Baseline**.

## Hard-Won Lessons (Reference)

### The 10/10 UI/UX Standard (Mandatory)
- **Header Scaling**: Use `###` (Header 3) as the maximum header size. Never use `##` or `#` (prevents oversized text on mobile).
- **F-Scan Anchoring**: Always place emojis at the **absolute start** of the header (e.g., `### 🏗️ Engineering`) to enable vertical marginal scanning.
- **Breathing Room**: Every system prompt must include the instruction: **"Use double-newlines and keep paragraphs under 3 lines."**
- **Surgical Capabilities**: Only enable metadata `capabilities` (Vision, Search, Code Interpreter) that are directly relevant to the persona. Prune others to keep the UI clean.
- **ID Hygiene**: Use short, lowercase, hyphenated IDs (e.g., `mythos-coder`) instead of long filenames or quant-heavy strings.
- **Technical Rigor**: For Math/Logic profiles, explicitly define symbols in the system prompt (e.g., "Use $\therefore$ for 'therefore' and $\implies$ for 'implies'") to force formal derivation logic.
- **The "No-Filler" Mandate**: For tool-like profiles (Extractor, FIM), use the phrase: **"Return ONLY the data. No conversational filler or introductory text."**
- **Type Standardization**: All numerical parameters (e.g., `repeat_penalty: 1.0`) must be explicitly defined as **floats**, never integers, for parser robustness.

### Optimization Baseline
- **MoE Ratio for Q6_K**: For **26B-A4B-Q6_K**, `N_CPU_MOE=14` + `CTX_SIZE=65536` (32K per slot for 2 slots) is the optimized spot for speed and stability on 16GB VRAM.
- **OOM Prevention**: If the server fails with `status=11/SEGV`, increase `N_CPU_MOE` by 2 or decrease `CTX_SIZE` by 16,384.
- **Sampler Stability**: Use `TEMP=0.7`, `MIN_P=0.02`, and `REPEAT_PENALTY=1.08` for Gemma-4 MoE models.
- **DRY Sampler**: Always include DRY (`multiplier=0.8`, `base=1.75`) to prevent loops.
- **KV Cache**: Use `CACHE_TYPE_K="q4_0"` and `CACHE_TYPE_V="q4_0"` by default.
- **Autocomplete/FIM Profiles**: For high-speed "Turbo" Autocomplete/FIM profiles:
    - **Omit Parameters**: Completely omit `thinking_budget_tokens` and `reasoning`.
    - **Disable Thinking**: Set `"chat_template_kwargs": { "enable_thinking": false }`.
    - **Thin Sampler Chain**: Use `"samplers": "top_k;top_p"` to minimize latency.
- **Reasoning Budget Message**: When setting `REASONING_BUDGET_MESSAGE` in `.conf`, avoid using control tokens like `<channel|>` as they can prematurely interrupt the model's output stream. Use a clean string like `" [Logic Finalized] "`.
