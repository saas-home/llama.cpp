# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with this repository.

## Critical Rules

- **Do NOT commit or push without explicit human approval.** If user asks you to commit, use `Assisted-by: <name>`, NOT `Co-authored-by:`.
- **Do NOT run `git push` or `gh pr create` on the user's behalf.** Automated PR submissions can result in a contributor ban.
- **Do NOT write PR descriptions, commit messages, or reviewer responses.**
- **Do NOT implement features the user does not fully understand.**
- **Avoid unicode characters:** use ASCII equivalents (`-` not `—`, `->` not `→`, `x` not `×`, `...` not `…`).
- **Before writing code, read all relevant files and understand existing patterns.** If the change is large or introduces a new pattern, PAUSE and ask for confirmation.
- **Majority of code must be human-authored.** AI may assist with corrections, formatting, repetitive patterns, or documentation drafts for understood components.
- Code comments must be concise — only state a constraint the code itself can't show. Never restate what the code already says, and never write comments meaningful only out of context (e.g., "this fixes the problem you mentioned").

## Architecture

### Core Library

Plain C/C++ with no dependencies beyond ggml. Public API is `include/llama.h`.

**Key types:**

- **`llama_model`** — loaded from GGUF; holds weights + architecture metadata. Model implementations live in `src/models/` (133 models).
- **`llama_context`** — runtime state: KV cache, batches, compute graph. Created via `llama_init_from_model()`.
- **`llama_batch`** — token batch for parallel decoding across sequences.
- **`llama_sampler`** — composable token-sampling chain (temp, top-p, grammar, penalties).
- **`llama_vocab`** — tokenizer (SPM, BPE, WPM, UGM).
- **`llama_cparams` / `llama_hparams`** — context and hyperparameter configs.
- **`llama-kv-cache`** — unified KV cache (DSA, ISWA variants); hybrid and recurrent memory.

**Inference pipeline:**

1. `llama_model_load()` — read GGUF, allocate weights, dispatch to model loader in `src/models/`.
2. `llama_init_from_model()` — create context, allocate KV cache, initialize backend buffers.
3. Build `llama_batch` with prompt tokens.
4. `llama_decode(ctx, batch)` — build ggml compute graph, execute on backend.
5. `llama_sampler_sample(sampler, ctx, -1)` — sample next token from logits.
6. Repeat 3-5 for generation.

### ggml — tensor library

ggml lives in `ggml/` and is a **dependency, not a submodule** (upstream: `github.com/ggml-org/ggml`). Provides tensor ops (mul_mat, norm, rope), backend abstraction (CPU/SIMD, CUDA, Metal, Vulkan, SYCL, HIP), and the compute graph builder.

**Matrix multiplication is unconventional:** `C = ggml_mul_mat(ctx, A, B)` means $C = B A^T$. See `CONTRIBUTING.md` for the diagram.

Backends are dynamic — multiple compiled in, selected at runtime via `--device`. Backend dynamic loading builds `.so` files (`GGML_BACKEND_DL`). Metal is enabled by default on macOS (`-DGGML_METAL=OFF` to disable).

### Server Architecture

`llama-server` is an OpenAI-compatible HTTP server (`tools/server/`). Read `tools/server/README-dev.md` before implementing server features.

**Core components:**

- **`server_context`** — main inference state; single-threaded; holds `llama_context` + all active slots.
- **`server_slot`** — one per parallel request; manages prompt, generation, and state per sequence.
- **`server_queue`** — thread-safe task queue (HTTP workers -> `server_context`).
- **`server_response`** — thread-safe result queue (`server_context` -> HTTP workers).
- **`server_task` / `server_task_result`** — units of work and results.
- **`server_tokens`** — unified token representation (text + multimodal).

**Request flow:** HTTP handler parses JSON -> `server_task` -> queue -> `server_context` dispatches to `server_slot` -> slot calls `update_slots()` which batches across slots and calls `llama_decode()` -> results back via `server_task_result`.

**Thread model:** `server_context` runs on one thread. Heavy post-processing blocks multi-sequence throughput. HTTP workers handle JSON parsing, chat templates, tokenization — keep these concerns separate.

### Key Directories

| Path | Contents |
|------|----------|
| `src/` | Core llama library (model, context, graph, KV cache, sampler, vocab) |
| `src/models/` | Model-specific architecture implementations (133 models) |
| `ggml/` | Tensor library (ggml), upstream dependency |
| `include/llama.h` | Public C API header |
| `common/` | Shared utilities (chat, sampling, jinja, peg-parser, arg parsing, HTTP helpers) |
| `tools/` | CLI tools (server, cli, quantize, bench, imatrix, gguf-split, perplexity, tokenize, tts, rpc, ui) |
| `tools/server/` | OpenAI-compatible HTTP server |
| `tools/ui/` | SvelteKit web UI |
| `examples/` | Example programs |
| `tests/` | C++ unit tests, Python tokenizer tests, shell integration tests |
| `grammars/` | GBNF grammar files for structured output |
| `conversion/` | Python model conversion scripts |
| `scripts-local/` | Local workflow scripts (rebuild, sync, benchmarks) |

## Build

llama.cpp uses **CMake**. The Makefile is a stub that errors out.

```bash
# Basic build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# With CUDA (not preset-based — enable manually)
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# Presets (Ninja, pre-configured toolchains)
cmake --preset x64-linux-gcc-release
cmake --build build-x64-linux-gcc-release
```

Presets in `CMakePresets.json`: Linux GCC (debug/release/reldbg, static), Windows MSVC/LLVM, SYCL, Vulkan (Windows only). Key options: `GGML_CUDA`, `GGML_METAL`, `GGML_VULKAN`, `GGML_HIP`, `GGML_SYCL`, `GGML_BLAS`, `GGML_NATIVE`, `GGML_STATIC`, `GGML_CUDA_PEER_MAX_BATCH_SIZE`.

Output binaries: `build/bin/llama-*`, `build/bin/ggml-*`.

## Tests

```bash
cmake --build build --target tests
ctest --test-dir build --output-on-failure
ctest --test-dir build --label-exclude gpu --output-on-failure   # skip GPU tests
ctest --test-dir build --label-exclude "main|gpu" --output-on-failure  # quick smoke
```

Individual test: `./build/bin/test-<name>`. Full CI: `bash ci/run.sh ./tmp/results ./tmp/mnt` (see `ci/README.md`). Python tokenizer tests: `tests/test-tokenizer-0.py`, `tests/test-tokenizer-random.py`. Shell tests: `tests/test-tokenizer-0.sh`, `tests/test-lora-conversion-inference.sh`.

Benchmark scripts in `scripts-local/`: `bench-llama.py`, `bench-multi.py`, `save-baseline.sh`.

## Code Style

- **clang-format** (v15+): 4-space, 120 chars, `BraceWrapping.AfterCaseLabel: true`, `BreakBeforeBraces: Attach`
- **clang-tidy**: bugprone, readability, performance, portability, misc
- **Naming:** `snake_case` for functions/variables, `<class>_<method>` for API, UPPER_CASE enums
- **Language:** C++17, no fancy STL, basic for-loops. Python: flake8 (125 chars, excluding `examples`/`tools`), mypy strict, ty (`ty.toml`)
- **Pre-commit:** `pre-commit run --all-files` (trailing-whitespace, end-of-file, yaml, flake8)

## PR Guidelines (from CONTRIBUTING.md)

- Check existing issues/PRs before submitting; avoid duplicates.
- **One feature per PR.** For new models or features, add **CPU support only** in the initial PR; backends follow in separate PRs.
- New quantization types carry disproportionate maintenance burden: must provide conversion to HF, perplexity comparisons vs FP16/BF16, KL divergence data, and performance data on pure CPU.
- If modifying ggml, run `test-backend-ops` with at least two backends; add test cases for new operators.
- Verify perplexity and performance are not negatively affected (`llama-perplexity`, `llama-bench`).
- Check `ci/run.sh` locally before publishing.

## GGUF Models

- Models must be in GGUF format (`.gguf`). Convert from PyTorch/HF: `python convert_hf_to_gguf.py <model_path>`. Quantize: see `tools/quantize/README.md`. Add models: `docs/development/HOWTO-add-model.md`.
- Conversion scripts require Python >= 3.10 with torch, transformers, sentencepiece, numpy. `pip install -e ".[dev]"` via pyproject.toml.

## Local Workflow Scripts (`scripts-local/`)

- `rebuild-llama.sh` — Build/deploy/restart cycle for llama services. Flags: `--build`, `--bench`, `--no-deploy`, `--dry-run`, `--force`. Loads config from `.conf` files.
- `sync-fork.sh` — **Always use for upstream sync** (never manual merge/rebase). Flags: `--branch`, `--force`, `--dry-run`. Uses ff-only merge with autostash. Saves current branch, syncs master with upstream, rebases current branch on master, force-pushes.
- `.conf` files — Model service configs (Qwen3.6-35B-A3B, Gemma-4). Define SERVICE_NAME, MODEL_PATH, CUDA settings, context size, reasoning budgets.
- `vram-linter.py` — VRAM usage validation.

## Gotchas

- **ggml is a dependency**, not a submodule. Changes to ggml ops must be synced upstream.
- **Dynamic backends**: multiple compiled in, selected at runtime via `--device`. Use `--device none` for CPU-only.
- **`ggml_mul_mat` transposes B**: `C = B A^T`.
- **GGML backend `.so` loading**: enable with `GGML_BACKEND_DL`.
- No git submodules.

## Useful References

- [AGENTS.md](AGENTS.md) — contributor and AI-agent guidelines
- [CONTRIBUTING.md](CONTRIBUTING.md) — project contribution rules
- [HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — adding new model support
- [Server README-dev](tools/server/README-dev.md) — server development scope
- [Parsing docs](docs/development/parsing.md) — PEG parser for model output
- [Jinja README](common/jinja/README.md) — template engine
- [Build docs](docs/build.md)
- [PR template](.github/pull_request_template.md)
- API changelogs: [libllama](https://github.com/ggml-org/llama.cpp/issues/9289), [llama-server REST](https://github.com/ggml-org/llama.cpp/issues/9291)
