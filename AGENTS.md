# AGENTS.md

Notes for anyone, human or agent, working in this repo or carrying these patches
upstream.

## What this repo is

Patches that make ExLlamaV3 build and run on Turing (sm_75). Upstream requires
compute capability 8.0. There is no vendored source here, only patches against
recorded upstream commits, plus the config and measurements that make them useful.

Nothing here is a fork to develop in. If you change a patch, regenerate it from a
real tree and re-verify on hardware. Do not hand-edit the `.patch` files.

## Applying

```bash
git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3
git checkout 0c49587a7c235e6303a6bbedc8b665272ad3a2ea
for p in ../exl3-turing/patches/000[1-4]*.patch; do
  git apply --check "$p" && git apply "$p" || echo "FAILED $p"
done
# optional: multi-GPU Tensor-Parallel fix (inactivity watchdog)
git apply ../exl3-turing/patches/0006-exllamav3-tp-cpu-reduce-inactivity-watchdog.patch
```

Always `--check` first. All four are verified to apply cleanly at that commit.
On a newer upstream expect `ptx.cuh` and `triton_paged.py` to need rebasing.
`0006` is independent and optional: it fixes the native-TP watchdog so the
CPU-reduce deadline measures inactivity rather than total pass time, which lets
long forward passes run over multiple GPUs without a spurious timeout.

## Building

```bash
python -m venv ~/exl3venv
~/exl3venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
TORCH_CUDA_ARCH_LIST=7.5 CUDA_HOME=/usr/local/cuda-12.8 \
  ~/exl3venv/bin/pip install -e . --no-build-isolation
```

`TORCH_CUDA_ARCH_LIST=7.5` does two jobs: it selects the kernels, and `setup.py`
reads it to define `EXL3_SM75`, which gates the shared-memory accommodations.
Build for any other architecture and you get upstream tuning, which is the point.

Ninja must be on `PATH` or the build silently falls back to serial distutils and
takes hours instead of minutes.

## Verifying a change

Compiling is not evidence. A mis-split MMA or a bad tile does not crash, it
produces fluent nonsense. Minimum bar:

1. `python -c "import exllamav3, torch; print(torch.cuda.get_device_capability(0))"`
2. Generate ~100 tokens and **read them**. Quote them in any report.
3. Measure decode tok/s **on the machine with the GPU**. Remote measurement over
   a network inflates wall-clock and produced a 14% error here.
4. If you touched anything shared-memory or MMA related, run a needle test at
   several depths. Short prompts will not catch tile or cache-quantisation bugs.

## Hazards

These each cost real time.

**Near-full VRAM does not OOM, it crawls.** Above roughly 22 of 23 GiB this model
drops to about 2 tok/s while `/v1/models` still answers in milliseconds, so it
looks hung rather than full. Check `nvidia-smi` before diagnosing anything else.
Any measurement taken in that state is worthless.

**One slow request blocks the whole server.** TabbyAPI stops answering everything,
including trivial endpoints, while a long generation runs. That is saturation, not
a crash.

**Never put a restart loop in an autostart script.** A loop respawns a second
server on every restart; the instances fight over the card and produce the
symptom above.

**Greedy decoding is not deterministic here.** Three runs at temperature 0 gave
453, 508 and 463 characters. Do not try to prove a change is lossless by diffing
output. Compare benchmark category scores instead.

**`pgrep -f <pattern>` matches the invoking command itself.** It will kill your
own shell and return empty output. Use `ps -eo args | grep -c '[p]attern'`.

## Upstream

Four separate PRs, not one. A mixed 300-line PR stalls; small independent ones
merge.

| patch | upstream | independent of Turing |
|---|---|---|
| `0003` VL context length | exllamav3 | yes, send first |
| `0004` `-gcs` chunk flag | exllamav3 | yes |
| `0001` + `0002` sm_75 port | exllamav3 | no |
| `0005` cc gate | tabbyAPI | no, needs the port first |

State these limitations in the PR rather than letting a reviewer find them:

- Ampere and newer are **untested**. Every change is behind `__CUDA_ARCH__ >= 800`
  or `EXL3_SM75`, so upstream paths should be unaffected, but that is reasoning,
  not evidence.
- Turing correctness is **behavioural**: coherent output and benchmark scores
  matching the pre-port baseline. No bit-exact comparison against a reference.
- Volta (sm_70) is deliberately still rejected. The port needs `ldmatrix` and
  `mma.m16n8k8`, both sm_75 and newer.

## Base commits

- exllamav3 `0c49587a7c235e6303a6bbedc8b665272ad3a2ea`
- tabbyAPI `fcc1a1078e1f766dad305045c7c4d30aaefa6458`
