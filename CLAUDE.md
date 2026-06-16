# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

- **Do NOT commit or push without explicit human approval.** If user asks you to commit, use `Assisted-by: <name>`, NOT `Co-authored-by:`.
- **Do NOT run `git push` or `gh pr create` on the user's behalf.** Automated PR submissions can result in a contributor ban.
- **Do NOT write PR descriptions, commit messages, or reviewer responses.**
- **Do NOT implement features the user does not fully understand.**
- **Avoid unicode characters:** use ASCII equivalents (`-` not `—`, `->` not `→`, `x` not `×`, `...` not `…`).
- **Before writing code, read all relevant files and understand existing patterns.** If the change is large or introduces a new pattern, PAUSE and ask for confirmation.
- **Majority of code must be human-authored.** AI may assist with corrections, formatting, repetitive patterns, or documentation drafts for understood components.

## Architecture

### Core Library (llama.cpp)

The library is a plain C/C++ implementation with no external dependencies (beyond ggml). Public API is in `include/llama.h`.

**Key types:**

- **`llama_model`** — loaded from a GGUF file; holds weights and architecture metadata. Models live in `src/models/` (133 model implementations).
- **`llama_context`** — runtime state: KV cache, batches, compute graph. Created via `llama_init_from_model()`.
- **`llama_batch`** — token batch for parallel decoding across sequences.
- **`llama_sampler`** — composable token-sampling chain (temp, top-p, grammar, penalties, etc.).
- **`llama_vocab`** — tokenizer (SPM, BPE, WPM, UGM).
- **`llama_cparams`** / **`llama_hparams`** — context and hyperparameter configs.
- **`llama-kv-cache`** — unified KV cache (DSA, ISWA variants); supports hybrid and recurrent memory.

**Inference pipeline:**

1. `llama_model_load()` — read GGUF, allocate weights, dispatch to model-specific loader in `src/models/`.
2. `llama_init_from_model()` — create context, allocate KV cache, initialize backend buffers.
3. Build `llama_batch` with prompt tokens.
4. `llama_decode(ctx, batch)` — build ggml compute graph, execute on backend.
5. `llama_sampler_sample(sampler, ctx, -1)` — sample next token from logits.
6. Repeat steps 3-5 for generation.

### ggml — the tensor library

ggml lives in `ggml/` and is a **dependency, not a submodule** (upstream: `github.com/ggml-org/ggml`). It provides:

- Tensor allocation and ops (mul_mat, norm, rope, etc.)
- Backend abstraction: CPU (with SIMD dispatch), CUDA, Metal, Vulkan, SYCL, HIP
- Compute graph builder: ops are graphed, then scheduled and executed

**Matrix multiplication is unconventional:** `C = ggml_mul_mat(ctx, A, B)` means $C = B A^T$. See `CONTRIBUTING.md` for the diagram.

Backends are dynamic — multiple can be compiled in and selected at runtime via `--device`. Use `--device none` for CPU-only. GGML backend dynamic loading builds `.so` files loaded at runtime (`GGML_BACKEND_DL`).

### Server Architecture

`llama-server` is an OpenAI-compatible HTTP server (`tools/server/`). Read `tools/server/README-dev.md` before implementing server features.

**Core components:**

- **`server_context`** — main inference state; single-threaded; holds `llama_context` and all active slots.
- **`server_slot`** — one per parallel request; manages prompt, generation, and state for a single sequence.
- **`server_queue`** — thread-safe task queue (HTTP workers -> `server_context`).
- **`server_response`** — thread-safe result queue (`server_context` -> HTTP workers).
- **`server_task`** / **`server_task_result`** — units of work and results.
- **`server_tokens`** — unified token representation (text + multimodal).

**Request flow:** HTTP handler parses JSON, creates `server_task`, pushes to queue -> `server_context` dispatches to a `server_slot` -> slot calls `update_slots()` which batches across slots and calls `llama_decode()` -> results flow back via `server_task_result`.

**Thread model:** `server_context` runs on one thread. Heavy post-processing blocks multi-sequence throughput. HTTP workers handle JSON parsing, chat templates, tokenization — keep these concerns separate.

### Key Directories

| Path | Contents |
|------|----------|
| `src/` | Core llama library (model, context, graph, KV cache, sampler, vocab) |
| `src/models/` | Model-specific architecture implementations (133 models) |
| `ggml/` | Tensor library (ggml), upstream dependency at `github.com/ggml-org/ggml` |
| `include/llama.h` | Public C API header |
| `common/` | Shared utilities (chat, sampling, jinja, peg-parser, arg parsing, HTTP helpers, logging) |
| `tools/` | CLI tools (server, cli, quantize, bench, imatrix, gguf-split, perplexity, tokenize, tts, rpc, ui) |
| `tools/server/` | OpenAI-compatible HTTP server |
| `tools/ui/` | SvelteKit web UI |
| `examples/` | Example programs |
| `tests/` | Test sources (C++ unit tests, Python tokenizer tests, shell integration tests) |
| `grammars/` | GBNF grammar files for structured output |
| `conversion/` | Python model conversion scripts |
| `scripts-local/` | Local workflow scripts (rebuild-llama.sh, sync-fork.sh, model .conf files) |

## Build

llama.cpp uses **CMake**. The Makefile is a stub that errors out.

```bash
# Basic build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# With CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)

# Use presets (Ninja, pre-configured toolchains)
cmake --preset x64-linux-gcc-release
cmake --build build-x64-linux-gcc-release
```

CMake presets in `CMakePresets.json` cover Linux GCC (debug/release/reldbg, static), Windows MSVC/LLVM, SYCL, and Vulkan (Windows only). **CUDA is not preset-based** — enable manually with `-DGGML_CUDA=ON`.

Key CMake options: `GGML_CUDA`, `GGML_METAL`, `GGML_VULKAN`, `GGML_HIP`, `GGML_SYCL`, `GGML_BLAS`, `GGML_NATIVE`, `GGML_STATIC`, `GGML_CUDA_PEER_MAX_BATCH_SIZE`.

Output binaries: `build/bin/llama-*`, `build/bin/ggml-*`.

## Tests

Tests are CMake targets + CTest. Build tests first, then:

```bash
cmake --build build --target tests
ctest --test-dir build --output-on-failure
ctest --test-dir build --label-exclude gpu --output-on-failure   # skip GPU tests
ctest --test-dir build --label-exclude "main|gpu" --output-on-failure  # quick smoke
```

Individual test: `./build/bin/test-<name>` (e.g., `./build/bin/test-chat`).

Full CI locally: `bash ci/run.sh ./tmp/results ./tmp/mnt` (see `ci/README.md`). Set `GG_BUILD_CUDA=1` etc. for GPU backends.

Python tokenizer tests: `tests/test-tokenizer-0.py`, `tests/test-tokenizer-random.py`.
Shell integration tests: `tests/test-tokenizer-0.sh`, `tests/test-lora-conversion-inference.sh`.

Local benchmark scripts in `scripts-local/`: `bench-llama.py`, `bench-multi.py`, `save-baseline.sh`.

## Code Style

- **clang-format** (clang-tools v15+): 4-space indent, column limit 120, `BraceWrapping.AfterCaseLabel: true`, `BreakBeforeBraces: Attach`
- **clang-tidy**: bugprone, readability, performance, portability, misc checks
- Naming: `snake_case` for functions/variables, `<class>_<method>` pattern for API (e.g., `llama_sampler_chain_remove`), UPPER_CASE prefixed enum values (e.g., `LLAMA_VOCAB_TYPE_BPE`)
- C++17, no fancy STL constructs, basic for-loops preferred
- Python: flake8 (max 125 chars), mypy (strict mode), ty (`ty.toml`)
- Pre-commit: `pre-commit run --all-files` (trailing-whitespace, end-of-file, yaml, flake8)

## GGUF Model Format

- Models must be in GGUF format (`.gguf`).
- Convert from PyTorch/HF: `python convert_hf_to_gguf.py <model_path>`
- Quantize: see `tools/quantize/README.md`
- Add new model support: `docs/development/HOWTO-add-model.md`

## Gotchas

- **ggml is a dependency**, not a submodule. Changes to ggml ops must be synced upstream.
- **Backends are dynamic**: multiple backends compiled in, selected at runtime via `--device`. Use `--device none` for CPU-only.
- **Matrix multiplication is unconventional**: `C = ggml_mul_mat(ctx, A, B)` means $C = B A^T$.
- **GGML backend dynamic loading**: built as `.so` files loaded at runtime. Enable with `GGML_BACKEND_DL`.
- **Metal is enabled by default on macOS**. Disable with `-DGGML_METAL=OFF`.
- **No git submodules** currently in the repo.

## Server Development

The `llama-server` tool is a separate sub-project. Before implementing server features, read `tools/server/README-dev.md` for scope and conventions.

## Python Scripts

Model conversion scripts (`conversion/*.py`) require Python >= 3.10 with torch, transformers, sentencepiece, numpy.

```bash
pip install -e ".[dev]"  # via pyproject.toml
```

## Local Workflow Scripts

`scripts-local/` contains custom scripts for this environment:

- `rebuild-llama.sh` — Build/deploy/restart cycle for llama services. Supports `--build`, `--bench`, `--no-deploy`, `--dry-run`, `--force` flags. Loads config from `.conf` files.
- `sync-fork.sh` — **Always use this script for upstream sync** (never manual git merge/rebase). Supports `--branch`, `--force`, `--dry-run`. Uses ff-only merge with autostash. Saves current branch, syncs master with upstream, rebases current branch on master, force-pushes.
- `.conf` files — Model service configurations (Qwen3.6-35B-A3B, Gemma-4 variants). Define SERVICE_NAME, MODEL_PATH, CUDA settings, context size, reasoning budgets, etc.
- `bench-llama.py`, `bench-multi.py` — Benchmarking utilities.
- `vram-linter.py` — VRAM usage validation.

## Useful References

- [AGENTS.md](AGENTS.md) — full contributor and AI-agent guidelines
- [CONTRIBUTING.md](CONTRIBUTING.md) — project contribution rules, coding/naming guidelines
- [HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — adding new model support
- [Server README-dev](tools/server/README-dev.md) — server development scope and architecture
- [Parsing docs](docs/development/parsing.md) — PEG parser for model output
- [Jinja README](common/jinja/README.md) — template engine
- [Build docs](docs/build.md) — build documentation
