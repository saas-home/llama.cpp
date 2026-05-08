# Draft: Upstream issue for SOFT_MAX regression on Qwen3.6 (PRs #22468 / #22478)

> **Note before submitting**: This is a fact summary as raw material — please rephrase in your own voice, verify each fact still reproduces on your current setup, and adapt to whatever issue template `ggml-org/llama.cpp` requires. Per the project's AI policy ([AGENTS.md](../AGENTS.md)), GitHub issue text should be human-written, not pasted verbatim from AI output.
>
> **Where to file**: https://github.com/ggml-org/llama.cpp/issues/new (look for an "Eval bug" or similar template).
>
> **Pre-submission check**: re-run `./build/bin/llama-server --version`, `nvidia-smi`, and `nvcc --version` so the build IDs and driver/toolkit strings in the report are current.

---

## Suggested title

`Eval bug: SOFT_MAX failed / invalid argument at warmup on Qwen3.6-35B-A3B (gated_delta_net path), introduced by #22468 / #22478`

## Environment

- **Hardware**: NVIDIA RTX 4070 Ti SUPER (16 GB, sm_89 / Ada Lovelace)
- **CPU**: AMD Ryzen 9 7950X3D
- **OS**: Ubuntu 26.04 LTS, glibc 2.43
- **NVIDIA driver**: 595.58.03
- **CUDA toolkit tested**: 12.4 (Ubuntu apt), 12.8.2 (NVIDIA), 13.2 (NVIDIA) — **all three reproduce the bug**, so toolkit version is not the cause
- **llama.cpp**: master tip with PRs #22468 and #22478

## Model

`Qwen3.6-35B-A3B`, specifically the APEX-I-Balanced GGUF from [`mudler/Qwen3.6-35B-A3B-APEX-GGUF`](https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF). The bug is triggered on the `gated_delta_net` (GDN) layer path.

Architecture: 40 layers, hybrid layout `10 × (3× GDN→MoE → 1× attention→MoE)`, 256 experts (8 routed + 1 shared).

## Symptom

At warmup decode (and on the first real decode if `--no-warmup` is used), the server aborts with:

```
ggml_cuda_compute_forward: SOFT_MAX failed
CUDA error: invalid argument
  current device: 0, in function ggml_cuda_compute_forward at ggml-cuda.cu:2962
```

Stack trace path: `ggml_backend_sched_graph_compute_async` → `llama_context::graph_compute` → `llama_context::process_ubatch` → `llama_decode`. SIGABRT, exit 134.

## Reproduction

Build llama.cpp at master HEAD with `-DGGML_CUDA=ON`, deploy `llama-server` with the Qwen3.6-35B-A3B model. Any inference request (or even the warmup empty run) triggers the abort.

Reproduces independently of:

- KV cache type (`q8_0`, `q4_0`, `f16` — all fail)
- Flash attention (`on` or `off` — all fail)
- Context size (8K, 32K, 256K — all fail)
- CUDA graphs (`GGML_CUDA_DISABLE_GRAPHS=1` — still fails)
- Multimodal projector (with/without `--mmproj`)
- Reasoning/thinking mode
- Grammar/JSON schema constraints

## Bisect result

The two commits that introduced the regression:

- [`098705a29`](https://github.com/ggml-org/llama.cpp/commit/098705a29) — *CUDA: fuse SSM_CONV + ADD(bias) + SILU* (#22478, 2026-04-29)
- [`3142f1dbb`](https://github.com/ggml-org/llama.cpp/commit/3142f1dbb) — *ggml-cuda: refactor fusion code* (#22468, 2026-04-29)

## Workaround (verified)

Reverting both commits restores normal operation:

```bash
git revert --no-edit 098705a29 3142f1dbb
```

After the reverts, the same model runs cleanly on all three CUDA toolkit versions tested (12.4, 12.8.2, 13.2). No other changes needed.

## Diagnostic notes

- The error is `cudaErrorInvalidValue` raised by `cudaGetLastError()` at `ggml-cuda.cu:2960`, immediately after the SOFT_MAX dispatch (line 2962 is the `CUDA_CHECK` abort site).
- `dmesg | grep -i nvidia` is **clean** — no XID errors. This points to userspace-side validation rejecting the kernel launch parameters, not a runtime GPU fault.
- Likely cause: the fusion refactor in #22468 changed kernel grid/block dimensions or shared-memory request for some softmax shape produced by qwen3.6's GDN path. The shape isn't reached on dense-attention-only models, which may explain why CI didn't catch it.

## Suggested next steps for maintainers

1. Check that the fusion refactor preserves the previous launch parameters for SOFT_MAX dispatched from GDN-containing graphs.
2. Consider adding a CI test path that exercises a GDN-containing model architecture (Qwen3.6, Qwen3 Next, or any model using `GATED_DELTA_NET` op).
3. If the fix is non-trivial, consider reverting #22478 and #22468 until resolved (the workaround above is what's currently working).

---

## Appendix — useful diagnostic commands

```bash
# Build/version info
./build/bin/llama-server --version

# Driver and toolkit
nvidia-smi --query-gpu=driver_version --format=csv,noheader
/usr/local/cuda/bin/nvcc --version | tail -2

# Repro launch (single qwen3.6, minimal config — strips all variables to isolate the kernel bug)
./build/bin/llama-server \
  --model /path/to/Qwen3.6-35B-A3B-*.gguf \
  --n-gpu-layers 999 --n-cpu-moe 36 \
  --ctx-size 8192 --parallel 1 \
  --port 8080 --flash-attn on --kv-unified

# After the abort, capture dmesg for any GPU-side faults (typically empty in this bug)
sudo dmesg --since "5 minutes ago" | grep -iE "nvidia|XID|NVRM"

# Verify reverts work
git revert --no-edit 098705a29 3142f1dbb
rm -rf build && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)
# Then re-run the server above — warmup completes cleanly.
```
