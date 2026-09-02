# EXL3 on Turing

ExLlamaV3 running on a Quadro RTX 6000 (Turing, sm_75), which upstream does not
support. Qwen3.8-27B at 4.00 bpw, 262,144-token context, 42-45 tok/s with MTP
speculative decoding.

Upstream ExLlamaV3 requires compute capability 8.0 (Ampere). Its prebuilt wheels
carry no sm_75 kernels and the source does not compile for it. These patches fix
that. Everything is gated so Ampere and newer keep their original code paths.

## Measured

Quadro RTX 6000, 24 GB, Turing sm_75. Qwen3.8-27B EXL3 4.00bpw, Q4 KV cache,
TabbyAPI, single stream unless noted.

| | |
|---|---|
| decode, prose | 42-44 tok/s |
| decode, code | 42-45 tok/s |
| decode, no MTP | 31-33 tok/s |
| prefill | 812 tok/s at 24k, 457 at 32k, 296 at 64k |
| context | 262,144 loaded, 19.7 GiB |
| needle retrieval, Q4 cache | 12/12 to 64k depth |
| two concurrent streams | 49.9 tok/s aggregate |

Card ceilings I measured for reference: 531 GB/s streaming, 500 GB/s for an fp16
GEMV at batch 1. At 4 bpw the model reads ~14-16 GB per token, so ~38 tok/s is
the hard roofline for single-token decode. MTP beats it by emitting more than one
token per weight-read pass.

## What the patches do

| patch | what |
|---|---|
| `0001-exllamav3-sm75-cuda-kernels` | the port: cp.async, mma, atomics, tile sizes |
| `0002-exllamav3-sm75-triton-attention-tiles` | fit Triton attention into 64KB shared memory |
| `0003-exllamav3-qwen35-vl-context-length` | Qwen3.5 VL context bug, not Turing specific |
| `0004-exllamav3-chat-generator-chunk-size` | adds `-gcs` prefill chunk flag |
| `0005-tabbyapi-allow-sm75` | TabbyAPI rejects anything below cc 8.0 |

### The three real problems

**`cp.async` does not exist before sm_80.** ExLlamaV3 uses it for the software
pipeline in both the GEMM and the int8 GEMV. It hides behind four wrappers in
`ptx.cuh`, so the fix is four synchronous `uint4` copy bodies plus a
`__syncthreads()` for the wait. Neither `cp_async_wait` call site sits in
divergent control flow, so that substitution cannot deadlock. Pipeline depth
drops to 2 because with synchronous copies there is nothing left to overlap.

**`mma.m16n8k16` needs sm_80.** Turing has `m16n8k8`. A k16 fragment is exactly
two k8 fragments concatenated along K: `a[0],a[1]` cover k=0..7 and `a[2],a[3]`
cover k=8..15, and B splits the same way. Two k8 instructions accumulating into
the same registers are arithmetically identical, with no fragment reshuffling.
There are **three** sites, not two: both wrappers in `ptx.cuh` and a hand-written
one in `exl3_gemv_kernel.cuh` that bypasses the wrappers entirely.

**Shared memory is 64KB, not 100KB+.** This was the barrier that actually cost
time, not the instruction set. It bit in three places: the MoE `SMEM_MAX`, one
GEMM shape's pipeline depth, and the Triton attention tiles. The Triton formula,
calibrated against an observed launch rather than assumed, is
`block_m * head_dim * 2` for the Q tile plus `block_n * head_dim * 8` for K and V,
which are double-buffered regardless of `num_stages`. Clamping `num_stages` does
nothing; `block_m` is the term that has to move.

`cuda::atomic_ref` also needs C++20 while the build is C++17, so it becomes
`atomicAdd` / `atomicExch`. Keep `atomicExch` for the resets: other blocks may
still be grabbing tickets, and a plain store there is a race.

## Quickstart against upstream

Verified: all four exllamav3 patches apply cleanly to a pristine checkout at the
base commit below, and the tabbyAPI patch applies at its own.

```bash
# 1. upstream at the commit these were cut against
git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3
git checkout 0c49587a7c235e6303a6bbedc8b665272ad3a2ea

# 2. apply. --check first so a bad rebase fails loudly
for p in ../exl3-turing/patches/000[1-4]*.patch; do
  git apply --check "$p" && git apply "$p" || echo "FAILED $p"
done

# 3. build for Turing
python -m venv ~/exl3venv
~/exl3venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
TORCH_CUDA_ARCH_LIST=7.5 CUDA_HOME=/usr/local/cuda-12.8   ~/exl3venv/bin/pip install -e . --no-build-isolation

# 4. confirm
~/exl3venv/bin/python -c "import exllamav3, torch; print(torch.cuda.get_device_capability(0))"
```

`TORCH_CUDA_ARCH_LIST=7.5` matters twice: it selects the kernels, and `setup.py`
reads it to define `EXL3_SM75`, which gates the shared-memory accommodations.
Build for any other architecture and you get upstream tuning unchanged.

Ninja must be on `PATH`, or the build falls back to serial distutils and takes
hours.

### TabbyAPI

```bash
git clone https://github.com/theroyallab/tabbyAPI && cd tabbyAPI
git checkout fcc1a1078e1f766dad305045c7c4d30aaefa6458
git apply ../exl3-turing/patches/0005-tabbyapi-allow-sm75.patch

pip install -e .            # NO extras: the pinned exllamav3 wheels have no sm_75 kernels
pip uninstall -y exllamav3  # remove any wheel that would shadow the patched source
PYTHONPATH=/path/to/exllamav3 python main.py
```

`PYTHONPATH` is load-bearing. Without it TabbyAPI imports a stock exllamav3 and
dies on sm_75.

See [AGENTS.md](AGENTS.md) for verification steps and the failure modes worth
knowing before you debug something.


## Config that works

```yaml
model:
  model_dir: /path/to/models
  model_name: 4.00bpw
  max_seq_len: 262144
  cache_mode: Q4
  tool_format: qwen3_5      # else tool_calls is never populated
  reasoning: true           # else <think> text lands in `content`
draft_model:
  draft_mode: mtp
  draft_num_tokens: 1       # measured optimum, see below
  draft_cache_mode: Q4
```

**MTP draft depth peaks at 1 and declines after.** At 98.75% acceptance you are
already emitting ~2 tokens per weight-read pass, and each extra draft costs an
MTP forward pass faster than it earns accepted tokens.

| ndt | prose tok/s | acceptance |
|---|---|---|
| off | 33.1 / 33.5 | |
| **1** | **42.0 / 45.0** | 98.75% |
| 2 | 41.6 / 38.8 | 94.09% |
| 4 | 37.4 / 34.6 | 84.38% |
| 6 | 32.6 / 33.8 | 79.94% |

`draft_mode: ngram` gets 11.9% acceptance on this model. Not worth it.
`dynamic_draft` gave no gain over fixed ndt=1.

## Things that cost me time

**Measure decode on the machine with the GPU.** My baseline read 28.7 tok/s
across a Tailscale link and 33.3 locally. Every remote number was inflated.

**Near-full VRAM does not OOM, it crawls.** Above ~22 GiB of 23 GiB this model
drops to about 2 tok/s while `/v1/models` still answers in milliseconds, so it
looks hung rather than full. Check VRAM before diagnosing anything else.

**Do not put a restart loop in an autostart script.** Mine respawned a second
server on every restart; the instances fought over the card and produced exactly
the symptom above. Single attempt with a guard.

**Greedy decoding is not deterministic here.** Three runs at temperature 0 gave
453, 508 and 463 characters. Do not try to prove MTP is lossless by diffing
output. Compare category scores instead.

## Testing the Ampere path

I only have a Turing card, so the one claim I cannot verify myself is that these
patches leave Ampere and newer untouched. Everything is gated behind
`__CUDA_ARCH__ >= 800` or `EXL3_SM75`, but that is reasoning, not evidence.

If you have a 30-series or newer card, `tools/ampere-verify.sh` checks it:

```bash
git clone https://github.com/vcruz305/exl3-turing && cd exl3-turing
./tools/ampere-verify.sh --quick          # static check, about a minute, no build

export EXL3_TEST_MODEL=/path/to/any/exl3/model
./tools/ampere-verify.sh                  # full A/B, builds clean and patched
```

`--quick` confirms the Turing accommodations stay switched off without
`EXL3_SM75`: `SMEM_MAX` should resolve to `90 * 1024`, `GEMM_SHAPE_4` should keep
4 pipeline stages, and the MoE atomics should be unguarded only in the sense of
using `atomicExch` rather than a plain store.

The full run builds pristine upstream and the patched tree at the same commit,
for your own architecture, and benchmarks both. **They should match within
run-to-run noise.** It also fails loudly if `EXL3_SM75` gets defined on a non-7.5
build, which would mean the gate is leaking.

If patched comes out slower, please open an issue here with both numbers. That is
a real bug and I want to know before this goes upstream.

## Limitations

- **Ampere and newer are untested by me.** Every change is behind
  `__CUDA_ARCH__ >= 800` or `EXL3_SM75`, so upstream paths should be untouched, but
  I have no Ampere card. See [Testing the Ampere path](#testing-the-ampere-path) if
  you can help close this.
- Correctness on Turing is behavioural, not numerical: coherent output plus
  benchmark scores matching the pre-port baseline. No bit-exact comparison against
  a reference implementation.
- Retrieval is verified to 64k depth, not to the full 262k.
- Volta (sm_70) is not supported and is deliberately still rejected. The port
  relies on `ldmatrix` and `mma.m16n8k8`, both sm_75 and newer.

## Upstream

These are meant to go upstream. Patches 3 and 4 are independent of Turing and
should land on their own.

- exllamav3 base: `0c49587a7c235e6303a6bbedc8b665272ad3a2ea`
- tabbyAPI base: `fcc1a1078e1f766dad305045c7c4d30aaefa6458`
