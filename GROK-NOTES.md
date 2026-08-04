# llama.cpp — Complete Repository Analysis

> Generated: 2026-07-29
> Branch: `saashome-dev` (fork of `ggml-org/llama.cpp`, diverged at `571d0d540`)
> Local commits: 32 (all additions, nothing removed from upstream)

---

## 1. Repository Overview

llama.cpp is a high-performance inference engine for LLMs written in plain C/C++. It runs efficiently on consumer hardware with no external dependencies beyond the bundled `ggml` tensor library.

| Metric | Value |
|--------|-------|
| Total model implementations | 139 (`src/models/*.cpp`) |
| Total model lines | ~29,400 |
| Largest model file | `deepseek4.cpp` (1,199 lines) |
| Public API header | `include/llama.h` (1,604 lines) |
| Core source files | ~20 `.cpp` + `.h` in `src/` |
| ggml tensor library | `ggml/` (not a submodule) |
| Build system | CMake (Makefile is a stub) |
| Language | C++17, plain C for ggml core |

---

## 2. Architecture

### 2.1 Core Inference Pipeline

```
llama_model_load()          → read GGUF, allocate weights, dispatch to src/models/
llama_init_from_model()     → create context, allocate KV cache, init backend buffers
llama_batch                 → token batch for parallel decoding across sequences
llama_decode(ctx, batch)    → build ggml compute graph, execute on backend
llama_sampler_sample()      → sample next token from logits (temp, top-p, grammar, penalties)
llama_vocab                 → tokenizer (SPM, BPE, WPM, UGM)
```

### 2.2 Key Types

| Type | Purpose |
|------|---------|
| `llama_model` | Loaded from GGUF; holds weights + architecture metadata |
| `llama_context` | Runtime state: KV cache, batches, compute graph |
| `llama_batch` | Token batch for parallel decoding across sequences |
| `llama_sampler` | Composable token-sampling chain |
| `llama_vocab` | Tokenizer (SPM, BPE, WPM, UGM) |
| `llama_cparams` | Context parameters (ctx size, batch size, etc.) |
| `llama_hparams` | Hyperparameters (layers, heads, dim, etc.) |
| `llama-kv-cache` | Unified KV cache (DSA, ISWA variants); hybrid and recurrent memory |

### 2.3 ggml Tensor Library

Located in `ggml/`. Provides:
- Tensor operations: `mul_mat`, `norm`, `rope`, `concat`, `softmax`, etc.
- Backend abstraction: CPU/SIMD, CUDA, Metal, Vulkan, SYCL, HIP, OpenCL, WebGPU
- Compute graph builder

**Important:** `ggml_mul_mat(ctx, A, B)` computes $C = B A^T$ (transposed B convention).

Backends are **dynamic** — multiple compiled in, selected at runtime via `--device`. Backend dynamic loading builds `.so` files (`GGML_BACKEND_DL`).

### 2.4 Server Architecture (`tools/server/`)

`llama-server` is an OpenAI-compatible HTTP server.

**Core components:**
- `server_context` — main inference state; single-threaded; holds `llama_context` + all active slots
- `server_slot` — one per parallel request; manages prompt, generation, state per sequence
- `server_queue` — thread-safe task queue (HTTP workers → `server_context`)
- `server_response` — thread-safe result queue (`server_context` → HTTP workers)
- `server_task` / `server_task_result` — units of work and results
- `server_tokens` — unified token representation (text + multimodal)

**Request flow:** HTTP handler → parse JSON → `server_task` → queue → `server_context` → `server_slot` → `update_slots()` → `llama_decode()` → results back via `server_task_result`

**Thread model:** `server_context` runs on one thread. HTTP workers handle JSON parsing, chat templates, tokenization. Keep post-processing light to avoid blocking multi-sequence throughput.

**Server scope (in-scope):**
- Basic inference (text completion, embeddings)
- Chat features (completion, tool calling)
- Third-party API compatibility (OAI, Anthropic)
- Multimodal I/O
- Memory management (save/load state, checkpoints)
- Model management

**Out-of-scope:**
- Server-side agentic loops (external API calls in C++ are costly)
- Exposing internal model state via API
- Model-specific features (all API calls must be model-agnostic)

---

## 3. Directory Structure

| Path | Contents |
|------|----------|
| `src/` | Core llama library (model, context, graph, KV cache, sampler, vocab) |
| `src/models/` | Model-specific architecture implementations (139 models) |
| `ggml/` | Tensor library (ggml), upstream dependency |
| `include/llama.h` | Public C API header |
| `common/` | Shared utilities (chat, sampling, jinja, peg-parser, arg parsing, HTTP helpers) |
| `tools/` | CLI tools (server, cli, quantize, bench, imatrix, gguf-split, perplexity, tokenize, tts, rpc, ui) |
| `tools/server/` | OpenAI-compatible HTTP server |
| `tools/ui/` | SvelteKit web UI |
| `examples/` | Example programs |
| `tests/` | C++ unit tests, Python tokenizer tests, shell integration tests |
| `grammars/` | GBNF grammar files for structured output |
| `conversion/` | Python model conversion scripts (100+ model converters) |
| `scripts-local/` | Local workflow scripts (rebuild, sync, benchmarks) |
| `docs/development/` | HOWTO-add-model.md, parsing.md, debugging-tests.md |
| `docs/backend/` | Backend-specific docs (BLIS, CANN, CUDA, SYCL, Vulkan, etc.) |
| `models/templates/` | Jinja chat templates (60+ model templates) |
| `cmake/` | CMake modules and toolchains |
| `ci/` | CI scripts (`run.sh`) |
| `vendor/` | Third-party dependencies (cpp-httplib, nlohmann/json, stb, miniaudio) |

---

## 4. Build System

### 4.1 CMake Presets

Presets in `CMakePresets.json`:
- `x64-linux-gcc-debug` / `x64-linux-gcc-release` / `x64-linux-gcc-reldbg`
- `x64-linux-gcc+static-release`
- `x64-windows-msvc-debug` / `x64-windows-msvc-release`
- `x64-windows-llvm-debug` / `x64-windows-llvm-release`
- `arm64-apple-clang-debug` / `arm64-apple-clang-release`
- `x64-windows-sycl-debug` / `x64-windows-sycl-release`
- `x64-windows-vulkan-debug` / `x64-windows-vulkan-release`

### 4.2 Build Commands

```bash
# Basic build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# With CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build build -j$(nproc)

# Using preset
cmake --preset x64-linux-gcc-release
cmake --build build-x64-linux-gcc-release
```

**Key CMake options:**
- `GGML_CUDA` — Enable CUDA backend
- `GGML_METAL` — Enable Metal backend (on by default on macOS)
- `GGML_VULKAN` — Enable Vulkan backend
- `GGML_HIP` — Enable HIP backend
- `GGML_SYCL` — Enable SYCL backend
- `GGML_BLAS` — Enable BLAS backend
- `GGML_NATIVE` — Enable native CPU optimizations
- `GGML_STATIC` — Static linking
- `GGML_CUDA_PEER_MAX_BATCH_SIZE` — CUDA peer memory batch size
- `GGML_CUDA_FA` — Enable FlashAttention
- `GGML_CUDA_GRAPHS` — Enable CUDA graphs
- `GGML_CUDA_COMPRESSION_MODE` — `speed` or `memory`

**Output binaries:** `build/bin/llama-*`, `build/bin/ggml-*`

### 4.3 Local Build Script (CachyOS)

`scripts-local/llama-rebuild-cachyos.sh` — **Use this, NOT `rebuild-llama.sh`** (Ubuntu variant).

Flags: `--build`, `--install-deps`, `--dry-run`, `--bench`, `--no-deploy`

Auto-detects:
- `g++-15/14/13` for CUDA host compiler
- Pacman CUDA package path (`/usr/bin/nvcc`)
- Generates systemd unit with CPU affinity, `LimitMEMLOCK=infinity`, `StandardOutput=journal`

---

## 5. Model Implementations

### 5.1 Model Directory (`src/models/`)

139 model architecture implementations. Largest files:

| File | Lines | Model |
|------|-------|-------|
| `deepseek4.cpp` | 1,199 | DeepSeek 4 |
| `qwen35moe.cpp` | 741 | Qwen3.5 MoE |
| `qwen35.cpp` | 644 | Qwen3.5 |
| `delta-net-base.cpp` | 606 | DeltaNet |
| `qwen3next.cpp` | 595 | Qwen3 Next |
| `step35.cpp` | 556 | Step 3.5 |
| `kimi-linear.cpp` | 550 | Kimi Linear |
| `gemma4.cpp` | 508 | Gemma 4 |
| `deepseek32.cpp` | 506 | DeepSeek 32 |
| `gemma3n.cpp` | 459 | Gemma 3n |

### 5.2 Adding a New Model

See `docs/development/HOWTO-add-model.md` for full instructions.

**Steps:**
1. Convert model to GGUF (Python script in `conversion/`)
2. Define model architecture in `llama.cpp`:
   - Add `llm_arch` enum in `src/llama-arch.h`
   - Add arch name to `LLM_ARCH_NAMES` in `src/llama-arch.cpp`
   - Add metadata loading in `src/llama-model-loader.cpp`
   - Add RoPE type in `src/llama-model.cpp`
3. Build GGML graph in `src/models/<arch>.cpp`
4. Optional: Add multimodal encoder in `tools/mtmd/`

**GGUF conventions:**
- Tensor names must end with `.weight` or `.bias`
- `bid` keyword substitutes layer index in repetitive blocks
- `blk.{bid}.attn_norm` is the standard attention norm naming

### 5.3 Chat Templates (`models/templates/`)

60+ Jinja chat templates for various models. Examples:
- `meta-llama-Llama-3.1-8B-Instruct.jinja`
- `google-gemma-4-31B-it.jinja`
- `deepseek-ai-DeepSeek-V4.jinja`
- `Qwen3.5-4B.jinja`

---

## 6. Tools

| Tool | Purpose |
|------|---------|
| `llama-server` | OpenAI-compatible HTTP server |
| `llama-cli` | CLI inference |
| `llama-quantize` | Quantize GGUF models |
| `llama-bench` | Benchmark inference performance |
| `llama-perplexity` | Perplexity evaluation |
| `llama-imatrix` | Importance matrix for quantization |
| `llama-gguf-split` | Split GGUF files |
| `llama-tokenize` | Tokenize text |
| `llama-tts` | Text-to-speech |
| `llama-rpc` | RPC server |
| `llama-ui` | Web UI (SvelteKit) |
| `llama-mtmd` | Multimodal inference |

---

## 7. Tests

```bash
# Build tests
cmake --build build --target tests

# Run all tests
ctest --test-dir build --output-on-failure

# Skip GPU tests
ctest --test-dir build --label-exclude gpu --output-on-failure

# Quick smoke test
ctest --test-dir build --label-exclude "main|gpu" --output-on-failure

# Individual test
./build/bin/test-<name>

# Full CI
bash ci/run.sh ./tmp/results ./tmp/mnt
```

**Key tests:**
- `test-backend-ops` — Backend operator tests (run with ≥2 backends)
- `test-tokenizer-0.py` / `test-tokenizer-random.py` — Python tokenizer tests
- `test-peg-parser.cpp` — PEG parser tests
- `test-chat.cpp` — Chat template tests
- `test-sampling.cpp` — Sampling tests
- `test-grammar-*.cpp` — Grammar tests

---

## 8. Code Style

- **clang-format** (v15+): 4-space indent, 120 chars max, `BraceWrapping.AfterCaseLabel: true`, `BreakBeforeBraces: Attach`
- **clang-tidy**: bugprone, readability, performance, portability, misc
- **Naming:** `snake_case` for functions/variables, `<class>_<method>` for API, UPPER_CASE enums
- **Language:** C++17, no fancy STL, basic for-loops
- **Python:** flake8 (125 chars, excluding `examples`/`tools`), mypy strict, ty (`ty.toml`)
- **Pre-commit:** `pre-commit run --all-files` (trailing-whitespace, end-of-file, yaml, flake8)

---

## 9. PR Guidelines

From `CONTRIBUTING.md`:

1. Check existing issues/PRs before submitting; avoid duplicates
2. **One feature per PR**
3. For new models/features, add **CPU support only** in initial PR; backends follow in separate PRs
4. New quantization types must provide:
   - Conversion to HF
   - Perplexity comparisons vs FP16/BF16
   - KL divergence data
   - Performance data on pure CPU
5. If modifying ggml, run `test-backend-ops` with at least two backends
6. Verify perplexity and performance (`llama-perplexity`, `llama-bench`)
7. Check `ci/run.sh` locally before publishing

---

## 10. Local Workflow Scripts (`scripts-local/`)

### 10.1 `llama-rebuild-cachyos.sh`

**Use this, NOT `rebuild-llama.sh`** (Ubuntu variant uses `dpkg`/`apt` and will fail on CachyOS).

Flags: `--build`, `--install-deps`, `--dry-run`, `--bench`, `--no-deploy`

Auto-detects `g++-15/14/13`, pacman CUDA package path (`/usr/bin/nvcc`), and adds `cuda` package if missing.

Generates systemd unit with:
- CPU affinity
- `LimitMEMLOCK=infinity`
- `StandardOutput=journal`

### 10.2 `sync-fork.sh`

**Always use for upstream sync.** Flags: `--branch`, `--force`, `--dry-run`

Workflow:
1. Save current branch to fork
2. Sync `master` with upstream (fast-forward or rebase)
3. Rebase current branch on `master`
4. Push (force-push if history changed)

### 10.3 `vram-linter.py`

VRAM validation tool. Reads `.conf` file and estimates GPU memory usage.

**Known bug:** `estimate_vram()` only recognizes "qwen3", "qwopus", and "ornith" architecture names. Gemma-4, 12B, and 9B models are misdetected as Gemma-4-26B-A4B (30 layers, 8 KV heads, 128 experts), inflating KV estimates. Treat linter percentages above 100% for non-Qwen3 models as approximate only.

### 10.4 Model Configurations

| Config | Model | Weights | KV Cache | Peak Est. | Risk |
|--------|-------|---------|----------|-----------|------|
| `qwen-3.6-35b-a3b.conf` | MISSING | — | q4_0, 256K | ~11.3GB (69%) | LOW |
| `qwen3.6-35b-a3b-mtp.conf` | I-Balanced | ~17GB | q8_0, 262K | ~34GB (209%) | CRITICAL |
| `gemma-4-12b.conf` | Q6_K | 9.1GB | q4_0, 256K | ~26GB (inflated) | MODERATE |
| `gemma-4-26b-a4b.conf` | MISSING | — | q8_0, 256K | ~34GB (209%) | CRITICAL |
| `gemma-4-26b-a4b-qat.conf` | Q4_K_XL | 13.6GB | q4_0, 131K | ~18.7GB (114%) | FATAL |
| `qwythos-9b.conf` | Q8_0 | 9.1GB | Q8_0, 256K | ~26GB (163%) | MODERATE |
| `ornith-1.0-35b.conf` | — | — | — | — | — |

**Common fixes:**
- Default config (`MODEL_PATH=""`) has empty model path — service will fail unless override `.conf` is passed
- `N_GPU_LAYERS=999` means "all layers," but for MoE models use `N_CPU_MOE` (CPU expert ratio)
- MTP draft args in `EXTRA_ARGS` are commented out in most configs — uncomment to enable speedup
- Q8_0 KV cache (1.0 bytes/token) vs q4_0 (0.5) doubles KV cost. Use q4_0 for >10B models.

---

## 11. CUDA Backend (RTX 4070 Ti Super, Ada Lovelace, AD107, 16GB)

- Build flag: `-DCMAKE_CUDA_ARCHITECTURES="89"` (correct for Ada)
- CachyOS script enables:
  - `GGML_CUDA_FA=ON` — FlashAttention
  - `GGML_CUDA_FA_ALL_QUANTS=ON` — All quant variants
  - `GGML_CUDA_GRAPHS=ON` — CUDA graphs
  - `GGML_CUDA_COMPRESSION_MODE=speed`
- `CUDA_VISIBLE_DEVICES` not set — works for single-GPU
- Server uses `PARALLEL=4` logical slots sharing GPU via batched inference
- `--prio 2` (high) + CachyOS BORE scheduler
- IPv6 client URLs must use brackets: `http://[::1]:8080`
- Server binds to `127.0.0.1` by default (`common/common.h:627`); use `--host 0.0.0.0` for dual-stack
- CPU affinity `0-31` (16 threads on CCD0 and CCD2) for 7950X3D

---

## 12. Gotchas

- **ggml is a dependency**, not a submodule. Changes to ggml ops must be synced upstream.
- **Dynamic backends**: multiple compiled in, selected at runtime via `--device`. Use `--device none` for CPU-only.
- **`ggml_mul_mat` transposes B**: `C = B A^T`.
- **GGML backend `.so` loading**: enable with `GGML_BACKEND_DL`.
- **No git submodules**.
- **Avoid unicode characters**: use ASCII equivalents (`-` not `—`, `->` not `→`, `x` not `×`, `...` not `…`).
- **Before writing code, read all relevant files and understand existing patterns.** If the change is large or introduces a new pattern, PAUSE and ask for confirmation.

---

## 13. Useful References

| Resource | Path/URL |
|----------|----------|
| AGENTS.md | `AGENTS.md` — contributor and AI-agent guidelines |
| CONTRIBUTING.md | `CONTRIBUTING.md` — project contribution rules |
| HOWTO-add-model | `docs/development/HOWTO-add-model.md` — adding new model support |
| Server README-dev | `tools/server/README-dev.md` — server development scope |
| Parsing docs | `docs/development/parsing.md` — PEG parser for model output |
| Jinja README | `common/jinja/README.md` — template engine |
| Build docs | `docs/build.md` |
| PR template | `.github/pull_request_template.md` |
| libllama changelog | https://github.com/ggml-org/llama.cpp/issues/9289 |
| llama-server REST changelog | https://github.com/ggml-org/llama.cpp/issues/9291 |
| GGUF spec | https://github.com/ggml-org/ggml/blob/master/docs/gguf.md |

---

## 14. Branch Status

- **Current branch:** `saashome-dev`
- **Upstream divergence:** `571d0d540` ("model: rotate injected K/V cache for DFlash (#25823)")
- **Local commits:** 32 (all additions)
- **Upstream commits ahead:** 0 (master has 0 commits ahead of this branch)

### Local Diffs (32 commits)

**`scripts-local/`** — 15 files:
- Model configs: `gemma-4`, `qwen3.6`, `ornith`, `qwythos`
- Rebuild/sync/bench scripts: `llama-rebuild-cachyos.sh`, `sync-fork.sh`, `bench-llama.py`, `save-baseline.sh`
- VRAM linter: `vram-linter.py`

**`.gemini/`** — 3 files:
- Skills, models, settings

**`.github/copilot-instructions.md`**

**Root-level files:** `AGENTS.md`, `CLAUDE.md`, `.gitignore`

---

## 15. Key Files Quick Reference

| File | Purpose |
|------|---------|
| `include/llama.h` | Public C API (1,604 lines) |
| `src/llama-model.cpp` | Model loading and graph building |
| `src/llama-context.cpp` | Runtime context (4,157 lines) |
| `src/llama-kv-cache.cpp` | KV cache implementation |
| `src/llama-sampler.cpp` | Token sampling (3,883 lines) |
| `src/llama-vocab.cpp` | Tokenizer (4,355 lines) |
| `src/llama-graph.cpp` | Compute graph (3,514 lines) |
| `src/llama-quant.cpp` | Quantization (1,413 lines) |
| `ggml/src/ggml.c` | Core tensor ops (8,023 lines) |
| `ggml/src/ggml-quants.c` | Quantization ops (5,667 lines) |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | CUDA backend (5,425 lines) |
| `tools/server/server-context.cpp` | Server inference state |
| `tools/server/server-slot.cpp` | Per-request slot management |
| `common/common.h` | Shared utilities (1,143 lines) |
| `CMakePresets.json` | Pre-configured build presets |
| `conversion/convert_hf_to_gguf.py` | HF to GGUF conversion |
