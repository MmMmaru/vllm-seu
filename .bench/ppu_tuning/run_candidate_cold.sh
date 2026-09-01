#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/zsx/vllm-seu_test3
CANDIDATE="$ROOT/.bench/ppu_tuning/candidate"
BENCHMARK="$CANDIDATE/benchmark_public.py"
WRAPPER="$CANDIDATE/evaluation_wrapper.py"
CACHE_ROOT="$(mktemp -d /tmp/dndx-cudagraph-cache.XXXXXX)"
trap 'rm -rf -- "$CACHE_ROOT"' EXIT

test -f "$BENCHMARK"
test -f "$WRAPPER"
echo "[run] variant=candidate"
echo "[run] benchmark=$BENCHMARK"
echo "[run] wrapper=$WRAPPER"
echo "[run] cudagraph=prefill max=576; mm encoder=on; vision max=1024"
echo "[run] cold_cache=$CACHE_ROOT"

mkdir -p "$CACHE_ROOT/vllm" "$CACHE_ROOT/triton" \
  "$CACHE_ROOT/torchinductor" "$CACHE_ROOT/flashinfer"

export ZSX_USE_INSTALLED_VLLM=0
export ZSX_PPU_EAGER=0
export ZSX_GPU_MEMORY_UTILIZATION=0.82
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$CACHE_ROOT/flashinfer"
export PPU_SDK=/usr/local/PPU_SDK
export CUDA_PATH="$PPU_SDK/CUDA_SDK"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/site-packages/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /tmp
"$ROOT/.venv/bin/python" "$BENCHMARK" \
  --dataset-path "$ROOT/datasets/mmbench/mmbench_dev_en.tsv" \
  --model-path "$ROOT/Qwen3.5-2B" \
  --backend vllm \
  --num-samples "${2:-500}" \
  --warmup-samples 2 \
  --output "$ROOT/${1:-result_cudagraph_candidate_500.json}"
