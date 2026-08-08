#!/usr/bin/env bash
# run_benchmark.sh -- compile and run all three Monte Carlo benchmarks.
# Uses GPU 0 only; the second GPU is never touched.
#
# Path-aware and multi-GPU: results are written to results/<gpu-slug>/, so runs on
# different GPU types never overwrite each other. The GPU architecture is detected
# automatically for nvcc. After the run, results/<slug>/timings.json is produced
# (parsed from the printed tables) for gen_plots.py and collect_results.py.
set -e

# ── locate package root from this script's location (works from any cwd) ─────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$PKG_ROOT/src"

# ── detect GPU name, slug, and nvcc arch ─────────────────────────────────────
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
SLUG=$(echo "$GPU_NAME" | tr 'A-Z' 'a-z' | sed -e 's/nvidia//g' -e 's/generation//g' \
        | tr -cd 'a-z0-9')
[ -z "$SLUG" ] && SLUG="gpu0"
ARCH=$(python3 -c "
import subprocess
out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap',
                               '--format=csv,noheader'], text=True)
print('sm_' + out.strip().split('\n')[0].replace('.', ''))   # e.g. sm_90 for H100
")
OUT="$PKG_ROOT/results/$SLUG"
mkdir -p "$OUT"
RUNLOG="$OUT/run.txt"
echo "Detected GPU: $GPU_NAME  (arch $ARCH, slug '$SLUG')"
echo "Results -> $OUT"

# everything below is teed into results/<slug>/run.txt
{
echo "Detected GPU arch: $ARCH"

# ── provenance manifest (reproducibility) ────────────────────────────────────
{
  echo "# MOJO-GPU-reprocubility manifest"
  echo "date_utc:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host:        $(hostname)"
  echo "gpu:         $GPU_NAME"
  echo "gpu_count:   $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
  echo "driver:      $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "cuda_driver: $(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | head -1 | awk '{print $3}')"
  echo "nvcc:        $(nvcc --version | grep -i release | sed 's/^[[:space:]]*//')"
  echo "gpu_arch:    $ARCH"
  echo "mojo:        $(mojo --version)"
  echo "python:      $("${PYTORCH_PYTHON:-python3}" --version 2>&1)"
  echo "torch:       $("${PYTORCH_PYTHON:-python3}" -c 'import torch; print(torch.__version__)' 2>/dev/null)"
} | tee "$OUT/manifest.txt"
echo ""

# ── 1. CUDA C ─────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Compiling CUDA C reference kernel ..."
echo "=========================================="
nvcc -O3 -arch="$ARCH" -use_fast_math "$SRC/mc_cuda.cu" -o "$OUT/mc_cuda"
echo " Running CUDA C benchmark ..."
"$OUT/mc_cuda"

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Running PyTorch GPU baseline ..."
echo "=========================================="
"${PYTORCH_PYTHON:-python3}" "$SRC/mc_pytorch.py"

# ── 3. Mojo ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo " Running Mojo GPU kernel ..."
echo "=========================================="
mojo "$SRC/mc_mojo.mojo"

echo ""
echo "Done."
} 2>&1 | tee "$RUNLOG"

# ── parse the printed tables into a structured timings.json ──────────────────
python3 "$SCRIPT_DIR/collect_results.py" --run "$RUNLOG" --out "$OUT/timings.json" \
        --gpu "$GPU_NAME" --slug "$SLUG" --arch "$ARCH" || \
  echo "WARN: could not write timings.json (see collect_results.py)"

echo ""
echo "Wrote $OUT/{manifest.txt,run.txt,timings.json}"
echo "Refresh figures:   FIGDIR=../figures GPU_SLUG=$SLUG python3 scripts/gen_plots.py"
echo "Cross-GPU table:   python3 scripts/collect_results.py --table"
