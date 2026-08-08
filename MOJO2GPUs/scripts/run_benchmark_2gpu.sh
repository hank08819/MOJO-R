#!/usr/bin/env bash
# run_benchmark_2gpu.sh -- compile and run all three TWO-GPU Monte Carlo benchmarks.
# Uses GPU 0 and GPU 1 (data-parallel path split). Results -> results/<slug>_x2/.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$PKG_ROOT/src"

NG=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NG" -lt 2 ]; then echo "need 2 GPUs, found $NG"; exit 1; fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
SLUG=$(echo "$GPU_NAME" | tr 'A-Z' 'a-z' | sed -e 's/nvidia//g' -e 's/generation//g' | tr -cd 'a-z0-9')
[ -z "$SLUG" ] && SLUG="gpu"
SLUG="${SLUG}_x2"
ARCH=$(python3 -c "
import subprocess
out = subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader'], text=True)
print('sm_' + out.strip().split('\n')[0].replace('.', ''))")
OUT="$PKG_ROOT/results/$SLUG"
mkdir -p "$OUT"
RUNLOG="$OUT/run.txt"
echo "Detected $NG x $GPU_NAME (arch $ARCH); 2-GPU results -> $OUT"

{
echo "Detected GPU arch: $ARCH"
{
  echo "# MOJO2GPUs (two-GPU) manifest"
  echo "date_utc:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:        $(hostname)"
  echo "gpu:         $GPU_NAME"
  echo "gpu_count:   $NG"
  echo "ngpu_used:   2"
  echo "driver:      $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "cuda_driver: $(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | head -1 | awk '{print $3}')"
  echo "nvcc:        $(nvcc --version | grep -i release | sed 's/^[[:space:]]*//')"
  echo "gpu_arch:    $ARCH"
  echo "mojo:        $(mojo --version)"
  echo "python:      $("${PYTORCH_PYTHON:-python3}" --version 2>&1)"
  echo "torch:       $("${PYTORCH_PYTHON:-python3}" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
} | tee "$OUT/manifest.txt"
echo ""

echo "=========================================="
echo " Compiling + running CUDA C benchmark (2 GPU) ..."
echo "=========================================="
nvcc -O3 -arch="$ARCH" -use_fast_math "$SRC/mc_cuda_2gpu.cu" -o "$OUT/mc_cuda_2gpu"
"$OUT/mc_cuda_2gpu"

echo ""
echo "=========================================="
echo " Running PyTorch GPU baseline (2 GPU) ..."
echo "=========================================="
"${PYTORCH_PYTHON:-python3}" "$SRC/mc_pytorch_2gpu.py"

echo ""
echo "=========================================="
echo " Running Mojo GPU kernel (2 GPU) ..."
echo "=========================================="
mojo "$SRC/mc_mojo_2gpu.mojo"

echo ""
echo "Done (2 GPU)."
} 2>&1 | tee "$RUNLOG"

python3 "$SCRIPT_DIR/collect_results.py" --run "$RUNLOG" --out "$OUT/timings.json" \
        --gpu "$GPU_NAME x2" --slug "$SLUG" --arch "$ARCH" || \
  echo "WARN: could not write timings.json"

echo ""
echo "Wrote $OUT/{manifest.txt,run.txt,timings.json}"
