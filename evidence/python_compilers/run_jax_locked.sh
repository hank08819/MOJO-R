#!/usr/bin/env bash
# JAX added to the R3-1 comparison (2026-10-13). GPU 0, clock locked at 2505 MHz.
# CUDA C and Mojo are re-run in the same session as anchors against Table 4.
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
R=raw_jax; mkdir -p $R ${R}_repeat
{
echo "date_utc: $(date -u +%FT%TZ)"
nvidia-smi -i 0 --query-gpu=name,driver_version,clocks.gr --format=csv
${JAXPY:-python3} -c "import jax,jaxlib;print(\"jax\",jax.__version__,\"jaxlib\",jaxlib.__version__)"
echo "### CUDA C (anchor)"; ./mc_cuda | grep -v "^RAW\|^EVT"
echo "### Mojo (anchor)";   (pixi run mojo mc_mojo.mojo | grep -v "^RAW")
for v in vec loop; do echo "### JAX $v"; VARIANT=$v ${JAXPY:-python3} mc_jax.py $R 2>&1 | grep "^jax"; done
echo "### run-to-run repeat"; VARIANT=vec ${JAXPY:-python3} mc_jax.py ${R}_repeat 2>&1 | grep "^jax"
nvidia-smi -i 0 --query-gpu=clocks.gr --format=csv
echo DONE
} > $R.log 2>&1
