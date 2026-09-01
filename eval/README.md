# DNDX 多模态推理调优赛选手说明

## 1. 比赛目的

本次比赛面向多模态单选题推理场景，目标是在相同硬件、相同模型、相同公开自测数据和相同评测入口下，优化模型推理过程。

选手需要在不更换模型权重、不修改评测规则、不伪造结果的前提下，提升模型推理表现。重点考察：

- 准确率：模型最终答案是否正确。
- TTFT：从开始推理到产生第一个有效输出 token 的时间。
- 吞吐量：解码阶段 tokens/s。
- 工程实现质量：优化是否稳定、可复现、可在主办方环境运行。

本次比赛不是换模型比赛，也不是刷数据集比赛。所有最终成绩都由主办方在统一环境和私有测试集上复测得出。

## 2. 统一评测条件

主办方统一提供或指定：

- 硬件环境：同一台评测机器或同规格 GPU。
- 模型权重：固定版本的 `Qwen3.5-2B`。
- 评测入口：固定 benchmark 入口和 wrapper 接口。
- 最终测试集：主办方私有测试集，不提前发放。
- Python/CUDA/PyTorch/Transformers 等基础环境。

选手本地可以使用公开自测集调试：

```text
/home/maru/project/datasets/mmbench/
  mmbench_dev_en.tsv
  mmbench_dev_cn.tsv
```

公开自测结果仅用于调试，不作为最终排名依据。

## 3. 选手可修改范围

选手主要修改：

```text
evaluation_wrapper.py
```

必须保留以下接口契约：

```python
class VLMModel:
    def generate_with_metrics(
        self,
        *,
        image,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        ...
```

返回结构必须保持：

```python
GenerationResult(
    text=...,
    token_count=...,
    ttft_seconds=...,
    elapsed_seconds=...,
    meta=...,
)
```

## 4. 允许的优化方向

在不破坏模型正确性和评测公平性的前提下，允许选手进行以下优化：

~~- 量化及其算子优化。~~

- attention、matmul、norm、sampling 等 kernel 优化。
- KV cache 布局、分配和复用优化。
- 图像预处理和 tokenizer 调用优化。

- 投机解码
- 模型加载方式优化，例如 dtype、device map、初始化流程优化。
- CUDA Graph 或静态图优化。
- 自定义算子替换。
- prefill/decode 路径优化。
- 推理后端替换，但必须使用主办方指定的同一模型权重。
- monkey patch 模型内部模块，但不得改变任务语义和评测接口。

## 5. 禁止条例

以下行为一经发现，主办方可判定成绩无效：

- 更换模型权重、蒸馏模型、量化后使用非主办方认可的权重包。
- 修改正式评测数据或依赖外部私有数据。
- 修改主办方内部评测脚本、评分逻辑或反作弊逻辑。
- 跳过样本、筛选样本、提前终止部分样本。
- 根据 `sample_id`、题号、图片 hash、题干 hash 查表返回答案。
- 硬编码公开集或疑似测试集答案。
- 联网调用外部模型、API、搜索服务或远程推理服务。
- 伪造 `token_count`、`ttft_seconds`、`elapsed_seconds`。
- 将准确率评测和性能评测拆成两套不同策略。
- 输出无法明确映射到唯一 `A/B/C/D` 的答案。
- 利用解析漏洞、异常输出或超长输出影响评分。
- 在评测机器上写入持久化缓存以跨次复测获利，除非主办方明确允许。

## 6. 评分办法

主办方会在私有测试集上统一复测提交结果。核心指标：

- Accuracy：正确题数 / 总题数。
- Avg TTFT：平均首 token 延迟，越低越好。
- Avg Throughput：平均解码吞吐，越高越好。

建议采用两阶段排名：

1. 准确率门槛
   - 以主办方 baseline 为基准。
   - 准确率低于 baseline 允许范围的提交不进入性能排名。
   - 推荐门槛：`Accuracy >= baseline_accuracy - 2%`。

2. 性能综合排名
   - 在准确率合格的提交中比较 TTFT 和吞吐量。
   - TTFT 越低越好，吞吐量越高越好。

可选综合分示例：

```text
FinalScore = AccuracyScore * 0.6 + TTFTScore * 0.2 + ThroughputScore * 0.2
```

实际最终公式以主办方发布版本为准。

## 7. 本地自测方法

安装依赖：

```bash
uv pip install -r requirements.txt
```

准备模型目录：

```text
./Qwen3.5-2B
```

运行少量样本自测：

```bash
./.venv/bin/python benchmark_public.py \
  --dataset-path /home/maru/project/datasets/mmbench/mmbench_dev_en.tsv \
  --model-path /home/maru/huggingface/Qwen3.5-2B \
  --backend transformers \
  --num-samples 20 \
  --output result_dev_en.json
```

运行中文自测：

```bash
./.venv/bin/python benchmark_public.py \
  --dataset-path /home/maru/project/datasets/mmbench/mmbench_dev_cn.tsv \
  --model-path /home/maru/huggingface/Qwen3.5-2B \
  --backend transformers \
  --num-samples 20 \
  --output result_dev_cn.json
```

输出中会包含自测准确率、TTFT、吞吐量等信息。自测集成绩只用于调试，不代表最终排名。

## 8. 提交要求

选手提交：

```text
evaluation_wrapper.py
```

如实现依赖额外源码，可提交压缩包，但必须包含：

```text
evaluation_wrapper.py
requirements_extra.txt  # 如有额外依赖
README_SUBMISSION.md    # 简要说明优化点和运行要求
```

不得提交模型权重、测试集、预计算答案表或外部服务调用凭据。

## 9. 复测说明

主办方会将选手提交的 `evaluation_wrapper.py` 放入内部评分包，在统一环境中运行私有测试集。若本地自测成绩和主办方复测成绩存在差异，以主办方复测结果为准。
