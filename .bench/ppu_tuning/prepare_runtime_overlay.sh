#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALLED_VLLM="${VLLM_INSTALLED_ROOT:-/usr/local/lib/python3.12/site-packages/vllm}"
RUNTIME="$ROOT/.bench/runtime"

test -f "$INSTALLED_VLLM/__init__.py"
case "$RUNTIME" in
  "$ROOT/.bench/runtime") ;;
  *) echo "Refusing to replace unexpected runtime path: $RUNTIME" >&2; exit 1 ;;
esac

rm -rf -- "$RUNTIME"
mkdir -p "$RUNTIME"
cp -a "$INSTALLED_VLLM" "$RUNTIME/vllm"

for rel in \
  vllm/model_executor/layers/fused_qk_norm_rope.py \
  vllm/model_executor/models/qwen3_next.py \
  vllm/v1/worker/block_table.py \
  vllm/v1/worker/gpu_worker.py
do
  install -D -m 0644 "$ROOT/$rel" "$RUNTIME/$rel"
done

cd /tmp
PYTHONPATH="$RUNTIME" "$ROOT/.venv/bin/python" - <<'PY'
import vllm
from vllm.model_executor.layers import fused_qk_norm_rope
from vllm.model_executor.models import qwen3_next

print(f"runtime vllm: {vllm.__file__}")
print(f"fusion kernel: {fused_qk_norm_rope.__file__}")
print(f"model dispatch: {qwen3_next.__file__}")
PY
