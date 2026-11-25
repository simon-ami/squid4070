# squid4070

**A single-binary, zero-Python Zig + Ada Lovelace CUDA (`sm_89`) inference engine and OpenAI-compatible HTTP/SSE server for [`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) on an NVIDIA RTX 4070 Ti SUPER (16 GiB) — running on Linux and native Windows x64.**

> **Attribution:** Forked and adapted from [`tensorfish/squid5090`](https://github.com/tensorfish/squid5090) (Apache-2.0), which builds on kernel designs from [`satellitedown/inkling-2-mini-abliterated-ingine`](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine) and [`AtomicBot-ai/atomic-engine`](https://github.com/AtomicBot-ai/atomic-engine). Full attributions live in [`NOTICE`](NOTICE).

## What Changed vs. `squid5090`

- **Ada Lovelace (`sm_89`) target:** Ported from Blackwell (`sm_120a`) / CUDA 13 to Ada (`sm_89`) on CUDA 12.4–12.5 (`cudaGraphCondTypeIf` 3-branch conditional capture fallback for CUDA `< 13.0`).
- **GGUF MoE & Dense Loader (`src/gguf.zig`, `cuda/gguf_quants.cuh`, `cuda/kernels_gguf_moe.cu`):** Direct loading of Unsloth Dynamic GGUF weights (`UD-IQ2_XXS`, `UD-IQ3_XXS`, including `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ4_NL`, `Q4_K`–`Q8_0`) plus `dspark-Inkling-2-Mini-Q8_0.gguf` speculative drafting and optional host-pinned embeddings (`--embed-host`).
- **Native Windows x64 port (`src/tiger.zig`, `src/server.zig`, `src/main.zig`, `build.zig`, `cuda/squid4070_cuda.def`):** Compiles CUDA kernels into `squid4070_cuda.dll` (MSVC + `nvcc`) and cross-compiles `squid4070.exe` via Zig (`x86_64-windows-gnu`) using Win32 file mapping, `QueryPerformanceCounter`, and Winsock2 — no WSL2 required.

## Measured Performance (RTX 4070 Ti SUPER, 16 GiB, Native Windows x64)

Measured with `--mode fp8 --kv q4 --mtp 7 --embed-host --no-prompt-lookup` and `dspark-Inkling-2-Mini-Q8_0.gguf`:

| Model | Max Context | Cold Load | Standalone Decode (`ctx=16K`) | Peak VRAM | Free VRAM |
|---|---:|---:|---:|---:|---:|
| `Inkling-2-Mini-UD-IQ2_XXS.gguf` | `458,752` | `55.6 s` | `139.7 tok/s` (`dl=1.74`) | `13,974 MiB` | `2,090 MiB` |
| `Inkling-2-Mini-UD-IQ3_XXS.gguf` | `196,608` | `72.0 s` | `118.2 tok/s` (`dl=1.68`) | `14,432 MiB` *(at 190K fill)* | `1,632 MiB` |

## Build

### Linux (`sm_89`)

Requires CUDA `12.4+` (`nvcc`, `cuBLAS`), `g++`, and **Zig `0.17.1+`**:

```bash
zig build                        # ReleaseFast binary at zig-out/bin/squid4070
zig test src/gguf.zig -lc        # CPU GGUF header & parser unit tests
```

### Native Windows x64 (`squid4070.exe` + `squid4070_cuda.dll`)

1. Compile the 6 CUDA translation units with VS2022 `x64` (`vcvars64.bat`) + CUDA `12.4+` `nvcc.exe`, and link `squid4070_cuda.dll` against `cuda/squid4070_cuda.def`, `cudart.lib`, and `cublas.lib`:
   ```bat
   nvcc.exe -O3 --use_fast_math -std=c++20 -gencode arch=compute_89,code=sm_89 --expt-relaxed-constexpr ^
     -I cuda -Xcompiler "/MD /O2 /Zc:__cplusplus /utf-8 /EHsc" -c cuda\<unit>.cu -o obj\<unit>.obj
   link.exe /DLL /DEF:cuda\squid4070_cuda.def /OUT:squid4070_cuda.dll cudart.lib cublas.lib obj\*.obj
   ```
2. Generate the MinGW import library and cross-compile `squid4070.exe` with Zig:
   ```bash
   zig dlltool -d cuda/squid4070_cuda.def -D squid4070_cuda.dll -l build-win/libsquid4070_cuda.dll.a -m i386:x86-64
   zig build -Dtarget=x86_64-windows-gnu -Doptimize=ReleaseFast -Dcuda-import-lib=build-win/libsquid4070_cuda.dll.a
   ```
3. Place `squid4070.exe`, `squid4070_cuda.dll`, `cudart64_12.dll`, `cublas64_12.dll`, and `cublasLt64_12.dll` together in one `bin/` directory alongside a `meta/` directory containing the model tokenizer & chat template files (`tokenizer.json`, `config.json`, `chat_template.jinja`, `tokenizer_config.json`).

## Usage

```bash
# OpenAI-compatible HTTP/SSE server (Inkling-2-Mini UD-IQ3_XXS, 192K context, DSpark)
squid4070 serve D:\models\unsloth\Inkling-2-Mini-GGUF\Inkling-2-Mini-UD-IQ3_XXS.gguf \
  --meta-dir C:\Users\sbene\squid4070\meta \
  --draft-gguf D:\models\unsloth\Inkling-2-Mini-GGUF\dspark-Inkling-2-Mini-Q8_0.gguf \
  --host 127.0.0.1 --port 9932 --mode fp8 --kv q4 --ctx 196608 --mtp 7 --embed-host --no-prompt-lookup

# Quick CLI generation or tokenizer check
squid4070 tokenize C:\Users\sbene\squid4070\meta --prompt "Hello world"
squid4070 generate <model.gguf> --meta-dir <meta_dir> --draft-gguf <draft.gguf> --mode fp8 --kv q4 --ctx 16384 --mtp 7 --embed-host --prompt "Hello"
```

### Key CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--meta-dir` | model dir | Directory with `tokenizer.json`, `config.json`, and `chat_template.jinja` |
| `--draft-gguf` | none | Optional DSpark `Q8_0` draft GGUF for speculative decoding |
| `--mode` | `fp8` (`serve`), `exact` (CLI) | Dense weight precision (`exact` \| `fp8` \| `nvfp4`) |
| `--kv` | `auto` | Global KV cache format (`bf16` \| `fp8` \| `q4`) |
| `--ctx` | `65536` (`serve`), `8192` (CLI) | Maximum context window |
| `--mtp` | `7` (`serve`), `0` (CLI) | Speculative draft width (`0` = off, `7` = adaptive `{2,4,8}` graph switch) |
| `--embed-host` | off | Keep embedding table in pinned host RAM to save VRAM |
| `--no-prompt-lookup` | off | Disable hybrid O(1) prompt-lookup drafting |
| `--host` / `--port` | `0.0.0.0` / `8000` | HTTP server bind address and port |
| `--api-key` | `$SQUID_API_KEY` | Optional Bearer token auth for `serve` |

## Scope & References

- **Scope:** Text-only, single-user (`c=1`) OpenAI `/v1/chat/completions` & `/v1/completions` server with static `32 MiB` HTTP buffers (no vision encoder or constrained grammar decoding).
- **Docs:** [`GGUF_MAP.md`](GGUF_MAP.md) (tensor mapping), [`LEARNINGS.md`](LEARNINGS.md), [`LITE.md`](LITE.md), [`LICENSE`](LICENSE), [`NOTICE`](NOTICE).
