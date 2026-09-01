# export VLLM_SPEC_METHOD=ngram
export PYTHONPATH=vllm-seu
# export VLLM_PPU_FUSED_GATE_SILU=false
# export VLLM_PPU_FUSED_QK_NORM_GATE=true
# export VLLM_PPU_FUSED_GDN_PREFILL=false
# export VLLM_ENFORCE_EAGER=1
# export PROFILE_REQUEST_INDEX=3      # 采集第3个请求
# export PROFILE_BACKEND=torch
# export PROFILE_TRACE_DIR=".temp/profile-$(date +%d-%H-%M)"

cd ~/xrs/vllm-workspace
source /etc/profile.d/ppu-sdk.sh
mkdir -p eval/results
python eval/benchmark_public.py \
    --dataset-path "/mnt/data/datasets/mmbench/mmbench_dev_en.tsv" \
    --model-path "/mnt/data/models/Qwen3.5-2B" \
    --backend "vllm" \
    --num-samples "200" \
    --warmup-samples "3" \
    --output "eval/results/mmbench_vllm_qwen3.5-2B.json" 