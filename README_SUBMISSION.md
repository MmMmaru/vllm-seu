# DNDX 提交说明(README_SUBMISSION)

## 依赖清单

本提交依赖 vLLM for SAIL PPU 定制版(上游 `flytiger-eco/vllm-for-sail`,分支 `ppu-0.23.0`)及其 PPU 依赖。完整依赖清单已写入随提交附带的 `eval/requirements_extra.txt`,与 vllm 仓库的 `requirements/build/ppu.txt`、`requirements/ppu.txt` 保持一致


## 配置方法

```bash
# 2. Build vLLM (the machine needs network access to GitHub)

# 2.1 Set environment variables
export HGGC_ENABLE_COMPRESS=1
export NVCC_APPEND_FLAGS="-Xfatbin -compress-all"
export VLLM_REQUIRE_RUST_FRONTEND=0

# 2.2 Install build dependencies
cd vllm-for-sail
pip install -r requirements/build/ppu.txt
pip install -r requirements/ppu.txt
pip install numpy==1.26.0

# 2.3 Build
python setup.py bdist_wheel

# 3. Install vLLM
pip install dist/vllm*.whl

pip install -e .
```

## 导入方式
设置PYTHONPATH为vllm-seu根目录或使用 `pip install -e .` 安装后,即可在任意工作目录导入
安装后,评测脚本(可位于任意工作目录)使用:

```python
from vllm.eval.evaluation_wrapper import VLMModel, GenerationConfig, GenerationResult
```
backend选择vllm或者auto

## 技术细节

详见 docs/report/report.md