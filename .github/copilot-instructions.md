# GitHub Copilot Instructions for llama.cpp

> [!IMPORTANT]
> This project does **not** accept pull requests that are fully or predominantly AI-generated. AI tools may be utilized solely in an assistive capacity.
> Read more: [CONTRIBUTING.md](CONTRIBUTING.md)

## Critical Rules

- **Do NOT commit or push without explicit human approval.** If user asks you to commit, use `Assisted-by: <name>`, NOT `Co-authored-by:`.
- **Do NOT run `git push` or `gh pr create` on the user's behalf.** Automated PR submissions can result in a contributor ban.
- **Do NOT write PR descriptions, commit messages, or reviewer responses.**
- **Do NOT implement features the user does not fully understand.**
- **Avoid unicode characters:** use ASCII equivalents (`-` not `—`, `->` not `→`, `x` not `×`, `...` not `…`).
- **Before writing code, read all relevant files and understand existing patterns.** If the change is large or introduces a new pattern, PAUSE and ask for confirmation.
- **Majority of code must be human-authored.** AI may assist with corrections, formatting, repetitive patterns, or documentation drafts for understood components.
- Code comments must be concise — only state a constraint the code itself can't show. Never restate what the code already says.

## Architecture Overview

**Core library:** Plain C/C++ with no dependencies beyond ggml. Public API is `include/llama.h`.

**Key types:**
- **`llama_model`** — loaded from GGUF; holds weights + architecture metadata. Model implementations live in `src/models/` (137 models).
- **`llama_context`** — runtime state: KV cache, batches, compute graph. Created via `llama_init_from_model()`.
- **`llama_batch`** — token batch for parallel decoding across sequences.
- **`llama_sampler`** — composable token-sampling chain (temp, top-p, grammar, penalties).
- **`llama_vocab`** — tokenizer (SPM, BPE, WPM, UGM).
- **`llama-kv-cache`** — unified KV cache (DSA, ISWA variants); hybrid and recurrent memory.

**ggml** lives in `ggml/` and is a **dependency, not a submodule** (upstream: `github.com/ggml-org/ggml`). Provides tensor ops and backend abstraction.

**Matrix multiplication is unconventional:** `C = ggml_mul_mat(ctx, A, B)` means $C = B A^T$.

**Backends are dynamic** — multiple compiled in, selected at runtime via `--device`. Use `--device none` for CPU-only.

## Build

llama.cpp uses **CMake**. The Makefile is a stub that errors out.

```bash
# Basic build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# With CUDA
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

Output binaries: `build/bin/llama-*`, `build/bin/ggml-*`.

## Code Style

- **clang-format** (v15+): 4-space, 120 chars, `BraceWrapping.AfterCaseLabel: true`, `BreakBeforeBraces: Attach`
- **clang-tidy**: bugprone, readability, performance, portability, misc
- **Naming:** `snake_case` for functions/variables, `<class>_<method>` for API, UPPER_CASE enums
- **Language:** C++17, no fancy STL, basic for-loops
- **Python:** flake8 (125 chars, excluding `examples`/`tools`), mypy strict

## Key Directories

| Path | Contents |
|------|----------|
| `src/` | Core llama library (model, context, graph, KV cache, sampler, vocab) |
| `src/models/` | Model-specific architecture implementations (137 models) |
| `ggml/` | Tensor library (ggml), upstream dependency |
| `include/llama.h` | Public C API header |
| `common/` | Shared utilities (chat, sampling, jinja, peg-parser) |
| `tools/server/` | OpenAI-compatible HTTP server |
| `tools/ui/` | SvelteKit web UI |
| `tests/` | C++ unit tests, Python tokenizer tests |
| `grammars/` | GBNF grammar files |
| `scripts-local/` | Local workflow scripts (rebuild, sync, benchmarks) |

## Server

The `llama-server` tool is a separate sub-project. Before implementing server features, read `tools/server/README-dev.md` for scope and conventions.

## Local Developer Reference (saashome-dev)

- **CachyOS rebuild:** Use `scripts-local/llama-rebuild-cachyos.sh` (NOT `rebuild-llama.sh` — the Ubuntu script uses `dpkg`/`apt` and will fail on CachyOS).
- **Upstream sync:** Always use `scripts-local/sync-fork.sh` (NOT `git pull`).
- **VRAM validation:** `python3 scripts-local/vram-linter.py <config.conf>` — **Known bug:** only recognizes "qwen3", "qwopus", and "ornith" architecture names; Gemma-4 models are misdetected.
- **Benchmarking:** `scripts-local/bench-llama.py`, `scripts-local/bench-multi.py`, `scripts-local/save-baseline.sh`.
- **Gemini skills:** `.gemini/skills/llm-stack-optimizer/SKILL.md` (LLM config generation), `tools/ui/src/lib/components/app/SKILL.md` (Svelte 5 conventions).

## Gotchas

- **ggml is a dependency**, not a submodule. Changes to ggml ops must be synced upstream.
- **GGML backend dynamic loading:** built as `.so` files loaded at runtime. Enable with `GGML_BACKEND_DL`.
- **Metal is enabled by default on macOS**. Disable with `-DGGML_METAL=OFF`.
- **No git submodules** currently in the repo.

## Useful References

- [AGENTS.md](AGENTS.md) — full contributor and AI-agent guidelines
- [CLAUDE.md](CLAUDE.md) — architecture deep-dive, VRAM analysis, hardware notes
- [CONTRIBUTING.md](CONTRIBUTING.md) — project contribution rules
- [HOWTO-add-model.md](docs/development/HOWTO-add-model.md) — adding new model support
- [Server README-dev](tools/server/README-dev.md) — server development scope
- [Parsing docs](docs/development/parsing.md) — PEG parser for model output
- [Jinja README](common/jinja/README.md) — template engine
- [Build docs](docs/build.md)
- [PR template](.github/pull_request_template.md)
