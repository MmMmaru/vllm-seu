#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/zsx/vllm-seu_test3
TUNE_ROOT="$ROOT/.bench/ppu_tuning"
CACHE_ROOT="$TUNE_ROOT/cache/no_autotune_cold"

mkdir -p "$TUNE_ROOT/logs" "$CACHE_ROOT/vllm" "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/torchinductor"

export ZSX_USE_INSTALLED_VLLM=1
export ZSX_PPU_EAGER=0
export ZSX_GPU_MEMORY_UTILIZATION=0.82
export ZSX_MAX_MODEL_LEN=2048
export ZSX_MAX_NUM_SEQS=16
export ZSX_FLASHINFER_AUTOTUNE=0
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export PPU_SDK=/usr/local/PPU_SDK
export CUDA_PATH="$PPU_SDK/CUDA_SDK"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/site-packages/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /tmp
exec "$ROOT/.venv/bin/python" "$TUNE_ROOT/benchmark_public.py" \
  --dataset-path "$ROOT/datasets/mmbench/mmbench_dev_en.tsv" \
  --model-path "$ROOT/Qwen3.5-2B" \
  --backend vllm \
  --num-samples 500 \
  --warmup-samples 2 \
  --output "$ROOT/result_no_autotune_cold_500.json"
