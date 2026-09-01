#!/usr/bin/env bash
# 使用 Nsight Systems 采集一次 deferred MTP 请求及其 CPU runtime 活动。

set -euo pipefail
export PROFILE_REQUEST_INDEX="${PROFILE_REQUEST_INDEX:-3}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" && -x "${PROJECT_ROOT}/../vllm/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/../vllm/.venv/bin/python"
fi
VLLM_ROOT="${VLLM_ROOT:-${PROJECT_ROOT}/vllm-seu}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/../datasets/mmbench}"
MODEL_PATH="${MODEL_PATH:-/home/maru/huggingface/Qwen3.5-2B}"
PROFILE_DIR="${PROFILE_DIR:-${PROJECT_ROOT}/.temp/profile-$(date +%d-%H-%M)}"
LOG_DIR="${PROJECT_ROOT}/.log/profile"
RUN_NAME="${RUN_NAME:-deferred-request-${PROFILE_REQUEST_INDEX}}"

mkdir -p "${PROFILE_DIR}" "${LOG_DIR}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "nsys is required but was not found in PATH." >&2
    exit 1
fi

echo "profile request index: ${PROFILE_REQUEST_INDEX}"
echo "profile output: ${PROFILE_DIR}/${RUN_NAME}"

# PROFILE_REQUEST_INDEX is the only request-selection control variable.
PYTHONPATH="${VLLM_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
VLLM_USE_FLASHINFER_SAMPLER=0 \
nsys profile \
    --force-overwrite=true \
    --output="${PROFILE_DIR}/${RUN_NAME}" \
    --trace=cuda,nvtx \
    --sample=process-tree \
    --cudabacktrace=kernel \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-graph-trace=node \
    "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/eval/benchmark_public.py" \
    --dataset-path "${DATASET_ROOT}/mmbench_dev_cn.tsv" \
    --model-path "${MODEL_PATH}" \
    --output "${PROFILE_DIR}/${RUN_NAME}_result.json" \
    --num-samples 50 \
    --backend vllm \
    --device cuda \
    --warmup-samples 3 \
    2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"

# export PROFILE_BACKEND="torch"
# "${PYTHON_BIN}" \
#     "${PROJECT_ROOT}/eval/benchmark_public.py" \
#     --dataset-path "${DATASET_ROOT}/mmbench_dev_cn.tsv" \
#     --model-path "${MODEL_PATH}" \
#     --output "${PROFILE_DIR}/${RUN_NAME}_result.json" \
#     --num-samples 20 \
#     --backend vllm \
#     --device cuda \
#     --warmup-samples 3 \
#     2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
