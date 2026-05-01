# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read first

> **IMPORTANT:** Read [AGENTS.md](AGENTS.md) before any code changes. It defines the project's AI usage policy and prohibited actions for AI agents (no PR descriptions / commit messages / reviewer responses; no committing without explicit human approval; no implementing features the contributor doesn't understand).

This is a personal fork on branch `saashome-dev`. `scripts-local/`, `GEMINI.md`, `.gemini/`, and `server*.log` are local-only and not upstreamed. Default to upstream `master` for any contribution-style work.

## Local environment state (as of 2026-05-01)

### Branch state — deliberate reverts at HEAD

This branch carries two **deliberate reverts** to work around an upstream regression. Do **not** drop them without confirming a fix has landed upstream:

- `08d8c6fe8 Revert "ggml-cuda: refactor fusion code (#22468)"`
- `1c03bc8de Revert "CUDA: fuse SSM_CONV + ADD(bias) + SILU (#22478)"`

Symptom without these reverts: `ggml_cuda_compute_forward: SOFT_MAX failed / CUDA error: invalid argument` at warmup decode, on Qwen3.6‑35B‑A3B's gated‑delta‑net path. Affects RTX Ada (sm_89) on driver R595+. Reproduces on CUDA 12.4 / 12.8.2 / 13.2 — toolkit version is irrelevant; the bug is in upstream llama.cpp commits. A daily upstream watcher routine reports when the fix lands (https://claude.ai/code/routines/trig_01PCWcBH4TUjwivXxa7TzCG2) — when it returns `drop reverts`, run `git revert --no-edit 08d8c6fe8 1c03bc8de` and rebuild.

### Toolchain (verified working)

- **CUDA toolkit**: NVIDIA's `cuda-toolkit-13-2` at `/usr/local/cuda` (symlink → `/usr/local/cuda-13.2`)
- **Driver**: 595.58.03 (CUDA 13.2 era)
- **OS**: Ubuntu 26.04 LTS, glibc 2.43
- **Host compiler for CUDA**: `g++-14` (CUDA 13.2 doesn't accept Ubuntu 26.04's default gcc-15)
- **Hardware**: RTX 4070 Ti SUPER (16 GiB, sm_89), AMD 7950X3D

### Toolchain rules (enforced by lessons learned)

- **DO NOT install Ubuntu's `nvidia-cuda-toolkit` apt package** — it stomps on `/usr/lib/x86_64-linux-gnu/libcudart.so.12` and produces a hybrid linkage with NVIDIA's toolkit. `scripts-local/rebuild-llama.sh` was patched to skip auto-installing it when `/usr/local/cuda/bin/nvcc` exists.
- **DO NOT downgrade to CUDA 12.8.2 on Ubuntu 26.04** — its `crt/math_functions.h` conflicts with glibc 2.43's stricter `noexcept` specs on `cospi`/`sinpi`/`rsqrt` etc. Build won't compile. CUDA 13.x ships fixed headers.
- **The phantom `cudaMemGetInfo` bug** (reports ~58 GiB free on a 16 GiB card) is a CUDA 13.x driver issue, not a VMM issue. `-DGGML_CUDA_NO_VMM=ON` doesn't fully fix it but is harmless and worth keeping.

### Build flags in `rebuild-llama.sh` (non-default, load-bearing)

```bash
-DGGML_CUDA_FA_ALL_QUANTS=ON      # required for q8_0/q4_0 KV cache flash-attn
-DGGML_CUDA_NO_VMM=ON             # CUDA 13 VMM workaround (harmless, kept for safety)
-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14
```

### Active services

Two systemd units, both enabled and running:

| Unit | Port | Model | VRAM (typical) |
|---|---|---|---:|
| `llama-qwen.service` | 8080 | Qwen3.6‑35B‑A3B (APEX‑I‑Balanced) | ~10 GiB |
| `llama-gemma4.service` | 8081 | Mythos‑26B‑A4B PRISM‑PRO‑DQ (gemma‑4) | ~5 GiB |

Configs live at `scripts-local/qwen-3.6-35b-a3b.conf` and `scripts-local/gemma-4-26b-a4b.conf`. Deploy via `./scripts-local/rebuild-llama.sh <conf>` (omit `--build` if `build/bin/llama-server` is current). The script's VRAM linter (`scripts-local/vram-linter.py`) doesn't account for `--n-cpu-moe` offload, so its FATAL warnings on MoE configs are bogus — script proceeds anyway.

The four formerly-coexisting alternative units (`llama-server.service`, `llama-qwen-uc.service`, `llama-qwen27b.service`, `llama-qwen27b-ud.service`) were deleted; only the active two remain. The model `.gguf` files are still on disk under `/home/siva/models/` if needed.

### Validated sampler/reasoning config (do not regress)

Per Qwen team's official spec for Qwen3.6 thinking-mode general:

```
TEMP=1.0  TOP_P=0.95  TOP_K=20  MIN_P=0.0  PRESENCE_PENALTY=1.5  REPEAT_PENALTY=1.0
SAMPLERS="top_k;top_p;min_p;temperature"
REASONING_BUDGET=-1               # MUST be -1 — see "Reasoning budget gating" below
REASONING_BUDGET_MESSAGE=" [Logic Finalized] "
```

Per Google's spec for Gemma-4:

```
TEMP=1.0  TOP_P=0.95  TOP_K=64  MIN_P=0.02  REPEAT_PENALTY=1.1
SAMPLERS="top_p;temperature"
REASONING_BUDGET=-1               # MUST be -1 — see "Reasoning budget gating" below
REASONING_BUDGET_MESSAGE=" [Logic Finalized] "
```

**Critical sampler rule**: do **not** put `dry` in either `samplers` chain. DRY destroys code generation (forces continuation tokens off identifier/bracket repetition patterns — produces broken Python like `or` → `ever`, `nums[pivot_idx` with no closing bracket). Also degrades non-Latin script generation (Tamil agglutinative endings get penalized as repetition). Validated empirically by 6-prompt bench in this session. The `dry_*` config values can stay in conf files since they're inert when `dry` isn't in the chain.

### Reasoning budget gating — three‑layer call chain (code‑verified)

Three files cooperate to enforce reasoning budgets. **Getting one wrong silently breaks per‑persona budgets.**

**Layer 1 — `tools/server/server-common.cpp:1132–1144` (OpenAI‑compat translation):**

```cpp
int reasoning_budget = opt.reasoning_budget;                      // from conf (--reasoning-budget)
if (reasoning_budget == -1 && body.contains("thinking_budget_tokens")) {
    reasoning_budget = json_value(body, "thinking_budget_tokens", -1);
}
// ... then:
llama_params["reasoning_budget_tokens"] = reasoning_budget;
```

The OAI‑compat layer reads `thinking_budget_tokens` from the request body (this is the OpenWebUI/community field name), but **only when `opt.reasoning_budget == -1`**. If conf is `8192`, the gating fails and the per‑request value is silently ignored — every request uses the conf default and per‑persona budgets are dead. **This is why `REASONING_BUDGET` MUST be `-1` in both confs.**

**Layer 2 — `tools/server/server-task.cpp:480–484` (internal task setup):**

```cpp
const int32_t budget = json_value(data, "reasoning_budget_tokens", (int32_t) -1);
// ...
params.sampling.reasoning_budget_tokens = budget;
```

By this point the field has been translated. Internally the canonical name is `reasoning_budget_tokens`. The `data` here is the `llama_params` written by Layer 1, not the original request body.

**Layer 3 — `common/sampling.cpp:296` (sampler initialization):**

```cpp
if (!start.empty() && !end.empty() && (grammar_lazy || reasoning_budget_tokens >= 0)) {
    rbudget = common_reasoning_budget_init(...);
}
```

The budget sampler initializes only if start/end tags are present (chat template provides them when thinking is enabled) AND (lazy grammar OR positive budget). With `reasoning_budget_tokens == -1` and no lazy grammar, the sampler doesn't initialize and the budget message never fires.

**End-to-end consequences for the OpenWebUI persona JSONs:**

1. **Field name in persona JSON must be `thinking_budget_tokens`** (the OAI‑compat name read at Layer 1). Don't rename to `reasoning_budget_tokens` in the persona — that's the *internal* post‑translation name and won't be picked up by the OAI‑compat layer.
2. **`REASONING_BUDGET=-1` in conf is required** for the gating in Layer 1 to forward the per‑request value.
3. **`REASONING_BUDGET_MESSAGE` is concatenated with the end tag** and tokenized as `forced_tokens`, injected at budget exhaustion (`server-task.cpp:491`). Set it on both confs even though it only fires when budget enforcement triggers.
4. **Per‑persona `thinking_budget_tokens`** in the JSON (e.g. coder=1024, pro=2048, research=3072, math=2048, creative=1024, extractor=256) gets honored end‑to‑end now. Verified empirically: qwen‑pro on a hard prompt thought ~2352 tokens (cap=2048, slight overrun is UTF‑8 boundary handling), produced full answer. Same prompt before fix: 9 100 tokens of thinking, 0‑char answer.

**Persona rule for non‑thinking personas (FIM, Fast, Extractor)**: set `chat_template_kwargs: {"enable_thinking": false}` in the persona JSON. The script (`scripts-local/generate-ui-profiles.py`) sets this for all three. **Don't omit it for extractor** — qwen extractor would otherwise enable thinking and consume the entire `max_tokens=1024` budget on reasoning, producing an empty `content` block (the JSON answer gets generated *inside* the thinking block but never escapes). Found this exact bug during a session bench; gemma extractor only worked by accident because gemma is naturally brief.

### OpenWebUI persona system

Each model has 8 OpenWebUI personas defined in JSON files generated by `scripts-local/generate-ui-profiles.py` from the `.conf` files. The JSON files are **outputs**, not sources of truth — re-running `--generate-ui` overwrites manual edits. Source-of-truth changes go in the script.

```
scripts-local/openweb-ui_models_qwen-3.6-35b-a3b.json    # 8 personas
scripts-local/openweb-ui_models_gemma-4-26b-a4b.json     # 8 personas
```

Personas: `fim` (autocomplete, no thinking), `fast` (direct factual, no thinking), `coder`, `pro` (deep thinking), `research`, `creative`, `math`, `extractor` (no thinking).

**Persona rules learned (do not regress)**:

- **Don't add `<think>`/`</think>` tag directives to gemma system prompts.** Gemma uses `<|think|>` and `<|channel>thought` markers natively, NOT `<think>` tags. Forcing the wrong tag in the system prompt corrupts gemma's reasoning extraction. Trust gemma's chat template.
- **Don't add tag-wrapping directives to non-thinking personas** (FIM, Fast, Extractor). They have `enable_thinking: false`; telling the model to wrap thinking in tags contradicts the template.
- **Qwen's `preserve_thinking: true`** in `chat_template_kwargs` is intentional on agentic personas (coder, pro, research) — Qwen team's official feature for retaining reasoning across multi-turn agent workflows. Per Qwen docs: "particularly beneficial for agent scenarios."
- **Gemma's `chat_template_kwargs: {}`** (empty) is intentional. Google's spec explicitly forbids: "Thoughts from previous model turns must not be added before the next user turn begins." Don't add `preserve_thinking: true` for "consistency."
- **All 16 personas have `access_control: null`** which marks them public in OpenWebUI. The change in JSON does not auto-sync to OpenWebUI's database — a re-import (Admin Panel → Settings → Models) or `BYPASS_MODEL_ACCESS_CONTROL=True` env var is needed.
- **Frame-precise temporal analysis directive** is gemma-only on the Creative persona (gemma supports video at 60s/1fps; qwen3.6 doesn't have video).

### Lessons learned (avoid these mistakes)

- **`--log-disable` in the systemd unit** masks CUDA errors and made the SOFT_MAX failure undiagnosable until we relaunched manually with logs on. Keep it for routine production noise reduction, but the first thing to do when debugging a service failure is to launch the binary in a foreground terminal without it.
- **Don't apply nash_su's `<think>` GOAL/APPROACH/EDGE GBNF grammar** to qwen for general use. The 22× token reduction comes at the cost of suppressing the model's trained reasoning capability — the APEX‑I quant's imatrix calibration on reasoning traces is wasted, the architecture's hybrid GDN+attention design for long deliberation is bypassed. The grammar's `[\x09\x0A\x0D\x20-\x7E]+` charset also forbids non-ASCII output (no Tamil, no Unicode). Use it only for benchmark-style code-only batch tasks.
- **Both OpenWebUI persona JSON regen and conf-level changes** affect runtime. JSON personas override conf-level samplers/budget/reasoning per request. Fixing only conf without fixing personas leaves the bug alive in OWUI.
- **CUDA toolkit version is mostly a red herring** for runtime stability — driver and llama.cpp commits matter much more. Verified by running same workload on apt 12.4, NVIDIA 12.8.2, and NVIDIA 13.2 with the reverts in place: identical performance and stability across all three.
- **Don't rename `thinking_budget_tokens` to `reasoning_budget_tokens` in OpenWebUI persona JSONs.** Reading only `server-task.cpp:480` (which uses `reasoning_budget_tokens`) without checking `server-common.cpp:1132–1144` (the OAI‑compat translation layer) misses the fact that the *external* request body field is `thinking_budget_tokens` and the *internal* `llama_params` field after translation is `reasoning_budget_tokens`. Renaming in the persona breaks the OAI‑compat path. See "Reasoning budget gating" above for the full call chain. **Always read all three layers (server-common → server-task → sampling) before reasoning about request‑body field semantics.**

## Build

The Makefile is intentionally a stub — the build is CMake. The canonical reference is [docs/build.md](docs/build.md).

```bash
# Plain CPU build
cmake -B build
cmake --build build --config Release -j$(nproc)

# CUDA build (this fork's typical config — see scripts-local/rebuild-llama.sh for full flags)
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_CURL=ON
cmake --build build --config Release -j$(nproc)
```

Binaries land in `build/bin/` (`llama-cli`, `llama-server`, `llama-bench`, `llama-perplexity`, `llama-quantize`, `test-*`, ...).

For this fork's full deploy/rebuild/bench workflow (systemd service, VRAM linter, baseline comparison), use `./scripts-local/rebuild-llama.sh [config.conf] [--build|--bench|--no-deploy|--generate-ui]`. Active configs live in `scripts-local/*.conf`.

CMake presets for cross builds: see `CMakePresets.json` (e.g. `arm64-apple-clang-release`, `x64-windows-llvm-release`, `x64-windows-vulkan-release`).

## Test

Tests are CTest targets defined in `tests/CMakeLists.txt`.

```bash
# Run the full suite
ctest --test-dir build --output-on-failure -j$(nproc)

# Run one test (label or name)
ctest --test-dir build -R test-grammar-integration --output-on-failure
ctest --test-dir build -L main           # the "main" label is the default fast subset
ctest --test-dir build -L model          # tests requiring downloaded models (slow / opt-in)

# Run a test binary directly for fine-grained control / gdb
./build/bin/test-tokenizer-0 ./models/ggml-vocab-llama-bpe.gguf

# ggml backend op correctness (run when modifying any ggml op or backend)
./build/bin/test-backend-ops
```

Some tests (tokenizer regressions, lora-conversion, server e2e) shell out to Python and need vocab/model files — they're skipped automatically if the file is missing. Server e2e lives under `tools/server/tests/` (pytest-based, see `tools/server/tests/README.md`).

Linting / formatting: `pre-commit run --all-files` (config at `.pre-commit-config.yaml`); C/C++ uses `.clang-format` + `.clang-tidy`; Python uses flake8 + mypy + pyright (`mypy.ini`, `pyrightconfig.json`).

## Architecture

The codebase has a strict layering. Knowing which layer you're in determines what you can depend on.

```
include/llama.h          ← stable public C API (only file external consumers should include)
        │
        ▼
src/llama-*.{cpp,h}      ← libllama: model loading, KV cache, sampling, grammar, vocab, graph
src/models/<arch>.cpp    ← per-architecture model graph builders (one file per model family)
        │
        ▼
ggml/include + ggml/src  ← tensor library + per-backend kernels (cpu, cuda, metal, vulkan,
                           sycl, hip, musa, opencl, openvino, cann, hexagon, rpc, virtgpu,
                           webgpu, blas, zdnn). `ggml-backend-reg.cpp` is the registry.
```

On top of libllama:

- `common/` — shared helpers used by every CLI/tool but **not** part of the public API: `arg.cpp` (CLI parser), `chat.cpp` + `chat-peg-parser.cpp` + `chat-auto-parser*.cpp` (chat template + tool/reasoning output parsing), `sampling.cpp`, `reasoning-budget.cpp`, `speculative.cpp`, `json-schema-to-grammar.cpp`, `download.cpp` (HF cache), `jinja/` (template engine), `log.cpp`, `ngram-*.cpp`. Built as `libllama-common`.
- `tools/` — end-user binaries. Each subdir is its own CMake target. The **server** (`tools/server/`) is the largest and has a dedicated architecture documented in `tools/server/README-dev.md` — read it before touching server code. Other tools: `cli`, `bench`, `perplexity`, `quantize`, `imatrix`, `gguf-split`, `mtmd` (multimodal projector), `rpc`, `tts`, `tokenize`, `parser`.
- `examples/` — minimal demos (`simple/`, `batched/`, `embedding/`, `lookahead/`, ...). Treat as reference, not infrastructure.

Other directories:

- `convert_hf_to_gguf.py` + `convert_lora_to_gguf.py` — Python tools to convert HF checkpoints to GGUF. The big `convert_hf_to_gguf.py` (~650KB) maps HF arch names to our `LLM_ARCH_*` enums in `src/llama-arch.cpp`; whenever you add a model in `src/models/`, you almost always also touch `llama-arch.{cpp,h}`, the `convert_*` script, and possibly `gguf-py/`.
- `gguf-py/` — Python package (`gguf`) used by the converters; mirrors GGUF tensor names + metadata keys.
- `grammars/` — sample GBNF files; the GBNF spec is in `grammars/README.md`. Note the alternative PEG parser at `common/peg-parser.cpp` (docs: `docs/development/parsing.md`).
- `vendor/` — single-header dependencies (cpp-httplib, nlohmann/json, stb-image, miniaudio, subprocess.h). Do **not** add new third-party deps without discussion (see CONTRIBUTING.md §"Coding guidelines").

### Server architecture (key abstractions)

When working in `tools/server/`, the core types form this dataflow:

```
HTTP (cpp-httplib)  →  server_http_context  →  server_routes (JSON ↔ task)
                                                    │
                                       server_task ▼              ▲ server_task_result
                                                server_queue  server_response
                                                    │              │
                                                server_context ── server_slot[*]
                                                       (one slot per parallel sequence)
```

`server_tokens` is the unified token sequence representation (text + multimodal tokens). `server_prompt_checkpoint` snapshots KV state for SWA / recurrent models so prefix-shared requests can skip recompute. **Router mode** (multiple backend instances behind one endpoint) lives in `server_models.cpp` and is independent of `server_context`.

Read `tools/server/README-dev.md` §"Scope of features" before proposing any new server feature — there are explicit out-of-scope categories (server-side agentic loops, model-specific endpoints, frontend plugins, etc.).

## Conventions that bite if ignored

From [CONTRIBUTING.md](CONTRIBUTING.md):

- `snake_case` everywhere; enum values `UPPER_SNAKE` prefixed with the enum name (`LLAMA_VOCAB_TYPE_BPE`).
- Naming optimizes for **longest common prefix**: `number_small`/`number_big`, not `small_number`/`big_number`. Functions are `<class>_<action>_<noun>` — `llama_sampler_chain_remove`, not `remove_from_sampler_chain`.
- 4-space indent, brackets on the same line, `void * ptr`, `int & a`. Avoid templates and "fancy" STL — keep it close to C with classes.
- Tensors are row-major; `ggml_mul_mat(ctx, A, B)` computes `C = B Aᵀ` (yes, transposed — see CONTRIBUTING.md for the diagram).
- New model / new feature PRs: **CPU only first**, GPU backends in follow-ups. New `ggml_type` quants require perplexity + KL-div + perf data per CONTRIBUTING.md.
- If you modify a ggml op, add a case to `tests/test-backend-ops.cpp`.

## Adding a new model (high-level checklist)

Authoritative guide: [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md). The touch-list is roughly:

1. `src/llama-arch.{cpp,h}` — register `LLM_ARCH_<NAME>` and its tensor names.
2. `src/models/<name>.cpp` — graph builder (copy a structurally-similar model and adapt).
3. `src/llama-model.cpp` — wire architecture detection / hyperparameter loading.
4. `convert_hf_to_gguf.py` — add the HF → GGUF mapper class.
5. `gguf-py/gguf/constants.py` (and friends) — add metadata constants if needed.
6. `tests/test-llama-archs.cpp` and tokenizer tests if a new tokenizer is involved.
7. Mention in `README.md` model list.

## Useful documentation pointers

- [docs/build.md](docs/build.md) — every backend's build flags
- [docs/development/HOWTO-add-model.md](docs/development/HOWTO-add-model.md)
- [docs/development/parsing.md](docs/development/parsing.md) — PEG parser (preferred over regex for model output)
- [docs/autoparser.md](docs/autoparser.md) — auto-detect model-specific output formats
- [common/jinja/README.md](common/jinja/README.md) — chat template engine
- [grammars/README.md](grammars/README.md) — GBNF reference
- [tools/server/README-dev.md](tools/server/README-dev.md) — server internals + feature scope
- [ci/README.md](ci/README.md) — running full CI locally before publishing
