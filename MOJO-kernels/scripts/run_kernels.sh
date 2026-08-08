#!/usr/bin/env bash
# run_kernels.sh -- compile and run the Asian and MLP kernels in all three languages
# (CUDA C / PyTorch / Mojo) on one GPU. Writes results/<slug>/run.txt.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$PKG_ROOT/src"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
SLUG=$(echo "$GPU_NAME" | tr 'A-Z' 'a-z' | sed -e 's/nvidia//g' -e 's/generation//g' | tr -cd 'a-z0-9')
[ -z "$SLUG" ] && SLUG="gpu"
ARCH=$(python3 -c "import subprocess;o=subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'],text=True);print('sm_'+o.strip().split('\n')[0].replace('.',''))")
OUT="$PKG_ROOT/results/$SLUG"; mkdir -p "$OUT"
PY="${PYTORCH_PYTHON:-python3}"

{
echo "# MOJO-kernels run -- $GPU_NAME ($ARCH) -- $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for K in asian american mlp; do
  echo ""; echo "############### $K : CUDA C ###############"
  nvcc -O3 -arch="$ARCH" -use_fast_math "$SRC/${K}_cuda.cu" -o "$OUT/${K}_cuda"
  "$OUT/${K}_cuda"
  echo ""; echo "############### $K : PyTorch ###############"
  "$PY" "$SRC/${K}_pytorch.py"
  echo ""; echo "############### $K : Mojo ###############"
  mojo "$SRC/${K}_mojo.mojo"
done
} 2>&1 | grep -vE "UserWarning|cpu = _conversion" | tee "$OUT/run.txt"
echo ""; echo "Wrote $OUT/run.txt   (figures: FIGDIR=../figures python3 scripts/gen_plots_kernels.py)"
