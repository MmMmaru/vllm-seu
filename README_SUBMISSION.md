# DNDX 提交说明(README_SUBMISSION)

## 依赖清单

本提交依赖 vLLM for SAIL PPU 定制版(上游 `flytiger-eco/vllm-for-sail`,分支 `ppu-0.23.0`)及其 PPU 依赖。安装入口与依赖内容如下,与 `requirements/build/ppu.txt`、`requirements/ppu.txt` 保持一致。

### 构建期依赖(requirements/build/ppu.txt)

```text
# Should be mirrored in pyproject.toml
cmake>=3.26.1
ninja
packaging>=24.2
setuptools>=77.0.3,<81.0.0
setuptools-scm>=8
setuptools-rust>=1.9.0
torch
wheel
jinja2>=3.1.6
regex
build
protobuf >= 5.29.6, !=6.30.*, !=6.31.*, !=6.32.*, !=6.33.0.*, !=6.33.1.*, !=6.33.2.*, !=6.33.3.*, !=6.33.4.*
```

### 运行期依赖(requirements/ppu.txt)

```text
# Common dependencies
-r common.txt

numba

# Dependencies for PPU
# torch
# torchaudio
# flashmla==2.0.0
# These must be updated alongside torch
# torchvision
# FlashInfer should be updated together with the Dockerfile
# flashinfer-python
apache-tvm-ffi==0.1.9
# tilelang

# Required for faster safetensors model loading
fastsafetensors >= 0.2.2
```

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
```

## 导入方式

安装后,评测脚本(可位于任意工作目录)使用:

```python
from vllm.eval.evaluation_wrapper import VLMModel, GenerationConfig, GenerationResult
```

本地自测入口:`benchmark_public.py --backend vllm --dataset-path <dev.tsv> --model-path <Qwen3.5-2B>`。

## 技术细节

详见 docs/report/report.pdf