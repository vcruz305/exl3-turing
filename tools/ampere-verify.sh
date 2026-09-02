#!/usr/bin/env bash
# Verify these patches do not change anything on Ampere or newer.
#
# The patches are all gated behind __CUDA_ARCH__ >= 800 or EXL3_SM75, so on a
# 30-series card they should be inert. This checks that claim rather than
# assuming it, which is the one thing I cannot test myself.
#
#   ./ampere-verify.sh --quick    static check only, ~1 minute, no build
#   ./ampere-verify.sh            full A/B: builds clean and patched, benchmarks both
#
# Requires: git, python3, a CUDA toolkit, and a torch install matching it.
set -uo pipefail

BASE=0c49587a7c235e6303a6bbedc8b665272ad3a2ea
PATCHES="$(cd "$(dirname "$0")/../patches" && pwd)"
WORK="${WORK:-$HOME/ampere-verify}"
QUICK=0; [ "${1:-}" = "--quick" ] && QUICK=1

CC=$(python3 -c "import torch;print('%d%d'%torch.cuda.get_device_capability(0))" 2>/dev/null || echo "??")
NAME=$(python3 -c "import torch;print(torch.cuda.get_device_name(0))" 2>/dev/null || echo unknown)
echo "GPU: $NAME  (compute capability $CC)"
if [ "$CC" = "75" ]; then
  echo "This is a Turing card. This script is for Ampere or newer; nothing to check."
  exit 1
fi
ARCH="${CC:0:1}.${CC:1:1}"
echo "Will build for TORCH_CUDA_ARCH_LIST=$ARCH"
echo

fetch () {  # fetch <dir> <apply?>
  rm -rf "$1"; git clone -q https://github.com/turboderp-org/exllamav3 "$1"
  ( cd "$1" && git checkout -q "$BASE" )
  if [ "$2" = "patched" ]; then
    for p in "$PATCHES"/000[1-4]*.patch; do
      ( cd "$1" && git apply "$p" ) || { echo "  FAILED to apply $(basename "$p")"; exit 1; }
    done
  fi
}

echo "=== 1. static check: do the Turing accommodations stay switched off? ==="
mkdir -p "$WORK"; fetch "$WORK/patched" patched
Q="$WORK/patched/exllamav3/exllamav3_ext/quant"
probe () {  # probe <file> <symbol>
  out=$(echo "#include \"$2\"" > /tmp/_p.cu 2>/dev/null; \
        nvcc -E -arch=sm_$CC -I"$Q" -I"$WORK/patched/exllamav3/exllamav3_ext" /tmp/_p.cu 2>/dev/null \
        | grep -oE "$3" | tail -1)
  echo "$out"
}
SM=$(grep -A6 'ifdef EXL3_SM75' "$Q/exl3_moe_common.cuh" | grep -oE '\(90 \* 1024\)|\(60 \* 1024\)' | tail -1)
SH=$(grep -A6 'ifdef EXL3_SM75' "$Q/exl3_kernel_map.cuh" | grep -oE '16,     16,    512,     [0-9]' | tail -1)
echo "  without EXL3_SM75, SMEM_MAX resolves to : ${SM:-?}   (expect 90 * 1024)"
echo "  without EXL3_SM75, GEMM_SHAPE_4 is      : ${SH:-?}   (expect ... 512,     4)"
echo "  guards present in ptx.cuh               : $(grep -c '__CUDA_ARCH__' "$WORK/patched/exllamav3/exllamav3_ext/ptx.cuh")"
echo "  unguarded atomics (should be 0)         : $(grep -c '\*next_ticket_ptr = 0' "$Q/exl3_moe_kernel.cuh")"
echo
if [ "$QUICK" = "1" ]; then
  echo "Static check done. Re-run without --quick for the build and benchmark A/B."
  exit 0
fi

build () {  # build <dir>
  echo "  building $(basename "$1") for $ARCH ..."
  ( cd "$1" && TORCH_CUDA_ARCH_LIST="$ARCH" pip install -e . --no-build-isolation -q ) \
    > "$1/build.log" 2>&1
  if grep -qi "defining EXL3_SM75" "$1/build.log"; then
    echo "  !! EXL3_SM75 WAS DEFINED. It must only be set for a 7.5-only build."
  fi
  grep -ciE "error" "$1/build.log" | sed 's/^/  build errors: /'
}

bench () {  # bench <dir> <label>
  ( cd "$1" && python3 - "$2" <<'PY'
import sys, time, torch
sys.path.insert(0, ".")
import exllamav3
from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
MODEL = __import__("os").environ.get("EXL3_TEST_MODEL", "")
if not MODEL:
    print(f"  {sys.argv[1]}: set EXL3_TEST_MODEL to an EXL3 model dir to benchmark"); raise SystemExit
cfg = Config.from_directory(MODEL); model = Model.from_config(cfg)
cache = Cache(model, max_num_tokens=8192); model.load()
tok = Tokenizer.from_config(cfg); gen = Generator(model=model, cache=cache, tokenizer=tok)
ids = tok.encode("Explain what a transformer attention head does, step by step.")
gen.enqueue(Job(input_ids=ids, max_new_tokens=8))
while gen.num_remaining_jobs():
    for _ in gen.iterate(): pass
rates, text = [], ""
for _ in range(2):
    gen.enqueue(Job(input_ids=ids, max_new_tokens=200))
    torch.cuda.synchronize(); t = time.time(); n = 0
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            text += r.get("text", "") or ""
            tk = r.get("token_ids"); n += len(tk) if tk is not None else 0
    torch.cuda.synchronize(); rates.append(n / (time.time() - t))
print(f"  {sys.argv[1]}: {rates[0]:.2f} / {rates[1]:.2f} tok/s")
print(f"    sample: {text.strip()[:100]!r}")
PY
  )
}

echo "=== 2. build and benchmark CLEAN upstream ==="
fetch "$WORK/clean" clean
build "$WORK/clean"; bench "$WORK/clean" "clean   "
echo "=== 3. build and benchmark PATCHED ==="
build "$WORK/patched"; bench "$WORK/patched" "patched "
echo
echo "Compare the two tok/s lines. They should match within run-to-run noise,"
echo "and both samples should be coherent. If patched is slower, the guards are"
echo "leaking Turing settings onto Ampere and I want to know."
