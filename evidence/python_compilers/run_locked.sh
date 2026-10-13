#!/usr/bin/env bash
# European kernel, five implementations, one clock-locked session, GPU 0 only.
# Lock the clock first:   sudo nvidia-smi -i 0 -lgc 2505      (restore: sudo nvidia-smi -i 0 -rgc)
# Set PY to a Python with torch 2.5.1, triton 3.1.0, numba 0.67.0; run mojo through the pixi env of this repo.
set -e
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0
PY="${PY:-python3}"; NVCC="${NVCC:-nvcc}"; MOJO="${MOJO:-pixi run mojo}"
R=raw; mkdir -p $R ${R}_repeat
{
echo "date_utc: $(date -u +%FT%TZ)"
nvidia-smi -i 0 --query-gpu=name,driver_version,clocks.gr,clocks.max.gr --format=csv
$NVCC --version | tail -1
echo "### CUDA C";  $NVCC -O3 -arch=sm_89 -use_fast_math mc_cuda.cu -o mc_cuda && ./mc_cuda
echo "### Mojo";    $MOJO mc_mojo.mojo
echo "### PyTorch"; $PY mc_pytorch.py
for b in 64 128 256; do echo "### Numba b$b";  BLOCK=$b $PY mc_numba.py $R;  done
for b in 64 128 256; do echo "### Triton b$b"; BLOCK=$b $PY mc_triton.py $R; done
echo "### run-to-run repeat"
BLOCK=128 $PY mc_numba.py ${R}_repeat; BLOCK=128 $PY mc_triton.py ${R}_repeat
echo DONE
} > results_locked.log 2>&1
