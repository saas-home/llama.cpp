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
```

Individual test: `./build/bin/test-<name>` (e.g., `./build/bin/test-chat`).

Python tokenizer tests: `tests/test-tokenizer-0.py`, `tests/test-tokenizer-random.py`.
Shell integration tests: `tests/test-tokenizer-0.sh`, `tests/test-lora-conversion-inference.sh`.

Local benchmark scripts in `scripts-local/`: `bench-llama.py`, `bench-multi.py`, `save-baseline.sh`.

## Code Style

- **clang-format** (clang-tools v15+): 4-space indent, column limit 120, `BraceWrapping.AfterCaseLabel: true`, `BreakBeforeBrices: Attach`
- **clang-tidy**: bugprone, readability, performance, portability, misc checks
- Naming: `snake_case` for functions/variables, UPPER_CASE for enum values
- C++17, no fancy STL constructs, basic for-loops preferred
- Python: flake8 (max 125 chars), mypy (strict mode)

## Key Directories

| Path | Contents |
|------|----------|
| `src/` | Core llama library implementation |
| `src/models/` | Model-specific architecture implementations (132 models) |
| `ggml/` | Tensor library (ggml), upstream dependency at `github.com/ggml-org/ggml` |
| `include/llama.h` | Public C API header |
| `common/` | Shared utilities (chat, sampling, jinja, peg-parser) |
| `tools/` | CLI tools (server, cli, quantize, bench, imatrix, gguf-split, tts, parser, tokenize, completion, rpc, ui, etc.) |
| `tools/server/` | OpenAI-compatible HTTP server |
| `tests/` | Test sources (C++ unit tests, Python tokenizer tests, shell integration tests) |
| `conversion/` | Python model conversion scripts |
| `scripts-local/` | Local workflow scripts (rebuild-llama.sh, sync-fork.sh, model .conf files) |

## Gotchas

- **ggml is a dependency**, not a submodule. Changes to ggml ops must be synced upstream.
- **Backends are dynamic**: multiple backends compiled in, selected at runtime via `--device`. Use `--device none` for CPU-only.
- **Matrix multiplication is unconventional**: `C = ggml_mul_mat(ctx, A, B)` means $C = B A^T$.
- **GGML backend dynamic loading**: built as `.so` files loaded at runtime. Enable with `GGML_BACKEND_DL`.
- **Metal is enabled by default on macOS**. Disable with `-DGGML_METAL=OFF`.

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
- [CONTRIBUTING.md](CONTRIBUTING.md) — project contribution rules
- [HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — adding new model support
- [Server README-dev](tools/server/README-dev.md) — server development scope
- [Parsing docs](docs/development/parsing.md) — PEG parser for model output
- [Jinja README](common/jinja/README.md) — template engine
- [Build docs](docs/build.md) — build documentation
