#!/usr/bin/env bash
# PPU 直接评测入口：mmbench_dev_en 200 条，Qwen3.5-2B，vLLM backend（默认 ngram）。
# 硬编码，无变量配置；cwd 必须为 vllm-seu（eval 包由 sys.path[0]=cwd 导入）。
set -euo pipefail

source /etc/profile.d/ppu-sdk.sh
cd /root/xrs/vllm-workspace/vllm-seu
export PYTHONPATH=/root/xrs/vllm-workspace/vllm-seu
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_PPU_FUSED_QK_NORM_GATE=true
export VLLM_PPU_FUSED_GDN_PREFILL=true
export VLLM_PPU_FUSED_GDN_DECODE=true

mkdir -p eval/results
python eval/benchmark_public.py \
    --dataset-path /mnt/data/datasets/mmbench/mmbench_dev_en.tsv \
    --model-path /mnt/data/models/Qwen3.5-2B \
    --backend vllm \
    --num-samples 10000 \
    --warmup-samples 5 \
    --output eval/results/mmbench_vllm_qwen3.5-2B.json