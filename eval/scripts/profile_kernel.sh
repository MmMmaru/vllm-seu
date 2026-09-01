#!/usr/bin/env bash
# Collect a focused Nsight Compute report for a selected linear/GEMM kernel.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" && -x "${PROJECT_ROOT}/../vllm/.venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/../vllm/.venv/bin/python"
fi
VLLM_ROOT="${VLLM_ROOT:-${PROJECT_ROOT}/vllm-seu}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/../datasets/mmbench}"
DATASET_FILE="${DATASET_FILE:-mmbench_dev_cn.tsv}"
MODEL_PATH="${MODEL_PATH:-/home/maru/huggingface/Qwen3.5-2B}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
WARMUP_SAMPLES="${WARMUP_SAMPLES:-3}"
PROFILE_REQUEST_INDEX="${PROFILE_REQUEST_INDEX:-3}"
RUN_NAME="${RUN_NAME:-gemm-request-${PROFILE_REQUEST_INDEX}}"
PROFILE_DIR="${PROFILE_DIR:-${PROJECT_ROOT}/.temp/profile-$(date +%d-%H-%M)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/.log/profile_kernel}"
NCU_OUTPUT="${NCU_OUTPUT:-${PROFILE_DIR}/${RUN_NAME}}"
NCU_BIN="${NCU_BIN:-ncu}"
NCU_KERNEL_REGEX="${NCU_KERNEL_REGEX:-regex:.*Marlin.*}"
NCU_LAUNCH_SKIP="${NCU_LAUNCH_SKIP:-0}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"

mkdir -p "${PROFILE_DIR}" "${LOG_DIR}"

if ! command -v "${NCU_BIN}" >/dev/null 2>&1; then
    echo "${NCU_BIN} is required but was not found in PATH." >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter was not found: ${PYTHON_BIN}" >&2
    exit 1
fi

echo "profile request index: ${PROFILE_REQUEST_INDEX}"
echo "profile kernel regex: ${NCU_KERNEL_REGEX}"
echo "profile output: ${NCU_OUTPUT}.ncu-rep"

PYTHONPATH="${VLLM_ROOT}:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
VLLM_USE_FLASHINFER_SAMPLER=0 \
PROFILE_REQUEST_INDEX="${PROFILE_REQUEST_INDEX}" \
"${NCU_BIN}" \
    --target-processes all \
    --profile-from-start off \
    --kernel-name-base function \
    --kernel-name "${NCU_KERNEL_REGEX}" \
    --launch-skip "${NCU_LAUNCH_SKIP}" \
    --launch-count "${NCU_LAUNCH_COUNT}" \
    --section SpeedOfLight \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section LaunchStats \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    --force-overwrite \
    --export "${NCU_OUTPUT}" \
    "${PYTHON_BIN}" \
    "${PROJECT_ROOT}/eval/benchmark_public.py" \
    --dataset-path "${DATASET_ROOT}/${DATASET_FILE}" \
    --model-path "${MODEL_PATH}" \
    --output "${PROFILE_DIR}/${RUN_NAME}_result.json" \
    --num-samples "${NUM_SAMPLES}" \
    --backend vllm \
    --device cuda \
    --warmup-samples "${WARMUP_SAMPLES}" \
    2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
