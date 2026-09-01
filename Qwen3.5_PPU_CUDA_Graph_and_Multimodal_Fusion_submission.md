# Qwen3.5-2B PPU CUDA Graph 与多模态算子融合设计及性能验证

> 单文件提交版，文档日期：2026-09-01  
> 工程根目录统一记为 `vllm-workspace/`。本文中的工程文件路径均从该目录开始，不依赖服务器用户名、主目录或本机绝对路径。  
> 代码分析基于 XRS 中已整合的实现；本文已内嵌必要的配置、测试数据和 profiling 结果，不依赖图片、JSON、日志或代码快照附件。

## 1. 结论摘要

test3 相对原版的两类优化相互独立，可以叠加：

1. **CUDA Graph 扩展**：扩大语言模型 decoder/prefill 的捕获尺寸，并为多模态视觉 encoder 增加按视觉 token budget 捕获的 CUDA Graph。
2. **多模态 QK/MRoPE 算子融合**：把 `split -> Q/K RMSNorm -> MRoPE -> gate copy` 合并为一个 Triton kernel，同时支持纯文本 1D RoPE 和 Qwen3.5 多模态 `(T,H,W)` 三轴 MRoPE。

20 个正式样本、3 个 warmup 的严格四格 A/B 结果为：

- eager 下仅开启 QK/MRoPE 融合：平均 TTFT **下降 5.89%**，平均生成吞吐 **提高 7.03%**。
- CUDA Graph 下再开启融合：平均 TTFT **下降 2.59%**，平均生成吞吐 **提高 4.25%**，benchmark wall time **下降 5.88%**。
- CUDA Graph + 融合相对两项全关：平均 TTFT **下降 25.68%**，平均生成吞吐 **提高 427.07%**。
- 四组的 `response_text`、`parsed_answer`、`correct` 和 `token_count` 逐样本差异数全部为 **0**。

CUDA Graph 两组使用独立进程冷启动，首次 `torch.compile` 和图捕获使 wall time 明显增加；这不等同于常驻服务稳态变慢。20 个样本下，`graph_no_fusion -> graph_qk_fusion` 的 P95 TTFT 上升 6.23%，说明小样本尾延迟仍有波动，不能只报告平均值。

## 2. 优化边界与关系

```text
Qwen3.5 多模态请求
├─ 视觉 Encoder
│  └─ Encoder CUDA Graph：捕获并重放视觉 encoder
│
└─ 语言模型 Prefill / Decode
   ├─ Full / Piecewise CUDA Graph：捕获并重放语言模型执行段
   └─ Full-attention 层
      └─ QK RMSNorm + MRoPE + gate
         └─ Fused Triton kernel：替代多个独立 kernel
```

“fused kernel”是算子融合的具体实现，环境变量控制模型是否 dispatch 到它；CUDA Graph 记录并重放一串 kernel launch。两者不是同一种优化：

- 算子融合减少 kernel 数量、中间显存读写和 launch 次数。
- CUDA Graph 降低 CPU 调度与 kernel launch 提交开销。
- 融合决定“图里包含哪些 kernel”；CUDA Graph 决定“逐个 launch 还是整图 replay”。
- 图模式下 profiler 可能只显示 `cudaGraphLaunch`；如需展开节点，应使用 `nsys --cuda-graph-trace=node`。为了单独观察 fused kernel，本文的 Torch profiler 使用 eager 模式。

test3 最初新增的多模态融合特指 **Q/K RMSNorm + partial MRoPE + gate copy**。`Gate+SiLU`、`GDN prefill` 和 `GDN decode` 是 XRS 中的其他融合开关，不属于该 kernel。

## 3. CUDA Graph 设计

### 3.1 原版、test3 与当前 XRS 的配置差异

| 项目 | 原版/默认配置 | test3 性能分支 | 当前 XRS |
|---|---|---|---|
| Decoder/prefill capture sizes | `[1,2,4,8,16,24,32]` | `[1,2,4,8,16,24,32,128,192,256,320,384,448,512,576]` | `[1,2,4,8,16,24,32,40,48,56,64,128,256,384,448,512]` |
| 最大语言模型捕获尺寸 | 默认小尺寸 | 576 | 512 |
| 多模态 Encoder graph | 关闭 | 开启 | 开启 |
| Encoder token budgets | 无 | `[256,384,512,640,768,896,1024]` | `[64,128,256,384,512,640,768,896,1024]` |
| 图模式 | 默认 | 由 vLLM 解析 | `FULL_AND_PIECEWISE` |
| 单请求图片限制 | 无显式限制 | 1 | 1 |

当前配置入口：

```text
vllm-workspace/eval/evaluation_wrapper.py:229-280
```

核心配置为：

```python
enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1"
compilation_config = {
    "cudagraph_mm_encoder": True,
    "encoder_cudagraph_token_budgets": [
        64, 128, 256, 384, 512, 640, 768, 896, 1024
    ],
    "cudagraph_capture_sizes": [
        1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64,
        128, 256, 384, 448, 512
    ],
    "max_cudagraph_capture_size": 512,
    "cudagraph_mode": "FULL_AND_PIECEWISE",
}
```

### 3.2 语言模型 CUDA Graph 的调用顺序

初始化和捕获阶段：

```text
vllm-workspace/eval/evaluation_wrapper.py
└─ AsyncEngineArgs(compilation_config=...)
   └─ vllm-workspace/vllm-seu/vllm/v1/worker/gpu_model_runner.py
      ├─ 使用 CUDAGraphWrapper 包装模型
      └─ capture_model()
         ├─ CudagraphDispatcher.get_capture_descs()
         ├─ 捕获 PIECEWISE graphs
         ├─ 捕获 FULL decode graphs
         └─ 捕获 Encoder graphs
```

正式请求阶段：

```text
GPUModelRunner.execute_model()
├─ CudagraphDispatcher.dispatch(num_tokens, ...)
│  ├─ 规则匹配 decode：FULL
│  ├─ 规则匹配 prefill/mixed：PIECEWISE
│  └─ 超过最大尺寸或条件不满足：NONE/eager
├─ set_forward_context(cudagraph_runtime_mode, batch_descriptor)
└─ CUDAGraphWrapper.__call__()
   ├─ 已捕获且 key 匹配：entry.cudagraph.replay()
   └─ mode=NONE 或 key 不匹配：执行原始 runnable
```

主要实现位置：

```text
vllm-workspace/vllm-seu/vllm/v1/cudagraph_dispatcher.py:239-328
vllm-workspace/vllm-seu/vllm/compilation/cuda_graph.py:233-361
vllm-workspace/vllm-seu/vllm/v1/worker/gpu_model_runner.py:3900-3940
vllm-workspace/vllm-seu/vllm/v1/worker/gpu_model_runner.py:4397-4421
vllm-workspace/vllm-seu/vllm/v1/worker/gpu_model_runner.py:6731-6797
```

`FULL_AND_PIECEWISE` 表示：满足条件的统一 decode batch 优先走 FULL graph；prefill 或混合形状可以走 PIECEWISE graph；无法匹配的动态情况安全回退 eager。

### 3.3 `encoder_cudagraph_token_budgets` 的含义

该配置是视觉 Encoder 预先捕获的**输出视觉 token 尺寸列表**，不是生成 token 上限，也不是文本上下文长度。

Qwen3.5 对每张图片或视频的计算方式为：

```python
input_patches = t * h * w
encoder_output_tokens = t * (h // spatial_merge_size) * (w // spatial_merge_size)
```

例如 224×224 图片在 `patch_size=14`、`spatial_merge_size=2` 时：

```text
16 × 16 = 256 个输入 patch
8 × 8 = 64 个 Encoder 输出视觉 token
```

运行时选择最小的可容纳 budget：

| 实际视觉 token | 选择的 graph |
|---:|---:|
| 64 | 64 |
| 65 | 128 |
| 230 | 256 |
| 300 | 384 |
| 700 | 768 |
| 900 | 1024 |
| 1025 | 无匹配，回退 eager |

执行顺序为：

```text
图片输入
→ 计算 image_grid_thw 和 Encoder 输出 token 数
→ 选择最小可容纳的 token budget
→ 将输入复制、padding 到固定 graph buffer
→ encoder graph.replay()
→ 去除 padding并恢复原始图片顺序
→ 将视觉 embedding 合入语言模型输入
```

主要实现位置：

```text
vllm-workspace/vllm-seu/vllm/model_executor/models/qwen3_vl.py:1819-1870
vllm-workspace/vllm-seu/vllm/v1/worker/encoder_cudagraph.py:192-318
vllm-workspace/vllm-seu/vllm/v1/worker/encoder_cudagraph.py:320-442
```

budget 越密，padding 浪费越少，但启动捕获时间和 graph 显存占用越大。当前 `[64,128,...,1024]` 是对小图命中率、显存和冷启动开销的折中。

### 3.4 冷启动 wall time

图模式的独立进程要经历模型加载、`torch.compile`、各尺寸 warmup 和 CUDA Graph capture。本次测试中单个 compile range 约耗时 25 秒。请求级 TTFT/吞吐不包含全部进程启动成本，而 `wall s` 包含，因此会出现“请求更快、冷启动 wall 更长”。

生产环境应使用常驻 EngineCore，在启动阶段以真实图片尺寸预热，再测稳态；不应使用每次重启进程的 wall time 代替在线稳态指标。

## 4. 多模态融合 kernel 设计

### 4.1 公共模型调用链

```text
vllm-workspace/eval/benchmark_public.py::run_benchmark()
└─ vllm-workspace/eval/evaluation_wrapper.py::VLMModel.generate_with_metrics()
   └─ _generate_with_vllm_async()
      └─ AsyncLLM.generate()
         └─ GPUModelRunner.execute_model()
            └─ Qwen3_5ForConditionalGeneration.forward()
               └─ Qwen3_5Model.forward()
                  └─ Qwen3_5DecoderLayer.forward()
                     ├─ linear_attention：GDN 路径
                     └─ full_attention：Qwen3NextAttention.forward()
```

QK/MRoPE 融合仅作用于 `full_attention` 层，不作用于 `linear_attention/GDN` 层。

### 4.2 Dispatch 条件

判断位置：

```text
vllm-workspace/vllm-seu/vllm/model_executor/models/qwen3_next.py:307-375
```

启用条件包括：

- `VLLM_PPU_FUSED_QK_NORM_GATE=true`；
- 模型启用 `attn_output_gate`；
- 使用 NeoX-style RoPE；
- 运行在 CUDA/PPU CUDA 兼容后端；
- RoPE dtype 为 FP16 或 BF16；
- 当前为纯文本，或多模态 MRoPE 满足：`MRotaryEmbedding`、3 段 section、section 总和等于 `rotary_dim/2`、interleaved 布局。

本次多模态运行的启动日志为：

```text
PPU fused QK-norm+RoPE+gate is enabled
(VLLM_PPU_FUSED_QK_NORM_GATE=True, attn_output_gate=True,
is_neox_style=True, cuda=True, text_only=False,
mrope=(11, 11, 10), dtype=torch.float16).
```

旧实现只接受纯文本，因此 `text_only=False` 会禁用融合；当前条件为 `(text_only or supports_mrope)`。当 `(11,11,10)` 等结构校验通过时，多模态请求会真正进入 fused kernel。

### 4.3 不走融合时的执行顺序

```text
Qwen3NextAttention.forward()
├─ qkv_proj(hidden_states)
├─ _project_qkv_gate(qkv, positions)
│  ├─ qkv.split() 得到 q_gate、k、v
│  ├─ view + torch.chunk() 得到 q 和 gate
│  ├─ q_norm(q)：GemmaRMSNorm
│  ├─ k_norm(k)：GemmaRMSNorm
│  └─ rotary_emb(positions, q, k)
│     └─ MRotaryEmbedding.forward_cuda()
│        └─ triton_mrope()
├─ attention(q, k, v)
├─ attn_output *= sigmoid(gate)
└─ o_proj(attn_output)
```

对应位置：

```text
vllm-workspace/vllm-seu/vllm/model_executor/models/qwen3_next.py:377-414
vllm-workspace/vllm-seu/vllm/model_executor/layers/layernorm.py:137-172
vllm-workspace/vllm-seu/vllm/model_executor/layers/rotary_embedding/mrope.py:263-354
```

### 4.4 走融合时的执行顺序

```text
Qwen3NextAttention.forward()
├─ qkv_proj(hidden_states)
├─ _project_qkv_gate(qkv, positions)
│  ├─ qkv.split() 得到 q_gate、k、v
│  └─ fused_qk_rmsnorm_rope_gate()
│     ├─ 参数、shape、dtype 和 MRoPE section 校验
│     ├─ 分配 q_out、k_out、gate_out
│     └─ _fused_qk_rmsnorm_rope_gate_kernel
│        ├─ Q/K RMSNorm
│        ├─ T/H/W partial MRoPE
│        ├─ 非 rotary 尾部透传
│        └─ Q gate copy
├─ attention(q, k, v)
├─ attn_output *= sigmoid(gate)
└─ o_proj(attn_output)
```

kernel 实现位置：

```text
vllm-workspace/vllm-seu/vllm/model_executor/layers/fused_qk_norm_rope.py:16-272
```

kernel grid 为 `(n_tokens, num_q_heads + num_kv_heads)`，每个 Triton program 处理一个 token 的一个 Q/K head：

- RMSNorm 的平方和、`rsqrt` 和权重乘法在 FP32 中计算。
- 使用 `weight + 1` 保持 GemmaRMSNorm 语义。
- 归一化结果先回到输入 dtype，再转 FP32 做 RoPE，以匹配未融合路径的存储舍入行为。
- `[0, rotary_dim)` 执行 partial RoPE，`[rotary_dim, head_dim)` 只执行 RMSNorm。
- 多模态 positions 形状为 `(3, n_tokens)`，分别表示 T/H/W。
- Q head 在同一 program 内复制 gate；K head 不执行 gate copy。
- `v` 不参与该融合，直接传给 attention。

多模态位置选择的核心逻辑为：

```python
is_h = (rot_offs % 3 == 1) & (rot_offs < 3 * MROPE_SECTION_H)
is_w = (rot_offs % 3 == 2) & (rot_offs < 3 * MROPE_SECTION_W)
pos = where(is_h, pos_h, where(is_w, pos_w, pos_t))
```

未融合路径需要两个 RMSNorm、MRoPE 及多个拆分/复制操作；融合路径使用一次 Triton launch 完成整条链，并减少中间 tensor 往返显存。

### 4.5 融合与 CUDA Graph 的四种组合

| 配置 | 模型执行方式 | QK/MRoPE 部分 |
|---|---|---|
| `VLLM_ENFORCE_EAGER=1`，fusion=false | 每次正常 forward | 多个独立 RMSNorm/MRoPE 操作 |
| `VLLM_ENFORCE_EAGER=1`，fusion=true | 每次正常 forward | 单个 fused Triton kernel |
| `VLLM_ENFORCE_EAGER=0`，fusion=false | capture 后 graph replay | 图中包含多个未融合节点 |
| `VLLM_ENFORCE_EAGER=0`，fusion=true | capture 后 graph replay | 图中包含 fused kernel 节点 |

环境变量必须在 Engine 初始化和图捕获之前设置。图捕获完成后再修改环境变量，不会改变已经捕获的 graph 内容。

## 5. 性能测试方法与真实终端输出

### 5.1 固定条件

- 设备：PPU-ZW810E。
- 模型：Qwen3.5-2B，FP16。
- 数据集：MMBench dev EN。
- 正式样本：20。
- warmup：3。
- `max_num_seqs=1`。
- 关闭 speculative decoding、async scheduling 和 FlashInfer sampler。
- 关闭 Gate+SiLU、GDN prefill、GDN decode，避免干扰 QK/MRoPE 的归因。
- 四组只修改 `VLLM_ENFORCE_EAGER` 和 `VLLM_PPU_FUSED_QK_NORM_GATE`。

测试通过 VSCode Remote SSH 集成终端真实执行。终端最终汇总如下，内容直接内嵌在本文中：

```text
=== VS CODE TERMINAL PERFORMANCE SUMMARY ===
| case                  | avg TTFT | P50    | P95    | avg tok/s | P50     | P95     | wall s  | acc   |
| eager_no_fusion       | 62.666   | 61.726 | 86.521 | 44.271    | 44.442  | 44.739  | 77.751  | 16/20 |
| eager_qk_fusion       | 58.974   | 61.539 | 65.531 | 47.385    | 47.793  | 48.957  | 77.482  | 16/20 |
| graph_no_fusion       | 47.810   | 32.027 | 52.041 | 223.836   | 231.714 | 234.733 | 258.283 | 16/20 |
| graph_qk_fusion       | 46.574   | 31.964 | 55.282 | 233.338   | 235.776 | 237.126 | 243.086 | 16/20 |

Answer equivalence against eager_no_fusion:
- eager_qk_fusion: response_text=0, parsed_answer=0, correct=0, token_count=0
- graph_no_fusion: response_text=0, parsed_answer=0, correct=0, token_count=0
- graph_qk_fusion: response_text=0, parsed_answer=0, correct=0, token_count=0
```

### 5.2 四格 A/B 指标

| 配置 | 平均 TTFT ms | P50 | P95 | 平均 tok/s | P50 | P95 | 冷启动 wall s | 正确率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eager，无融合 | 62.666 | 61.726 | 86.521 | 44.271 | 44.442 | 44.739 | 77.751 | 16/20 |
| eager，QK/MRoPE 融合 | 58.974 | 61.539 | 65.531 | 47.385 | 47.793 | 48.957 | 77.482 | 16/20 |
| CUDA Graph，无融合 | 47.810 | 32.027 | 52.041 | 223.836 | 231.714 | 234.733 | 258.283 | 16/20 |
| CUDA Graph + QK/MRoPE 融合 | 46.574 | 31.964 | 55.282 | 233.338 | 235.776 | 237.126 | 243.086 | 16/20 |

| 对比 | 平均 TTFT | P95 TTFT | 平均吞吐 | 冷启动 wall |
|---|---:|---:|---:|---:|
| 仅 CUDA Graph | -23.71% | -39.85% | +405.60% | +232.19% |
| eager 下仅融合 | -5.89% | -24.26% | +7.03% | -0.35% |
| Graph 下再开融合 | -2.59% | +6.23% | +4.25% | -5.88% |
| Graph + 融合相对全关 | -25.68% | -36.11% | +427.07% | +212.65% |

逐样本等价性中的 `0` 表示相对 `eager_no_fusion` 没有任何差异。四组正确率相同，且生成文本、解析答案和 token 数完全一致。

### 5.3 历史 500 样本辅助结果

| 历史配置 | 平均 TTFT ms | 平均 tok/s | wall s | 正确率 |
|---|---:|---:|---:|---:|
| 公共基线 | 67.870 | 412.541 | 297.364 | 413/500 |
| CUDA Graph tuned cold | 48.526 | 406.016 | 302.851 | 414/500 |
| 同图配置、QK fusion off | 48.722 | 405.380 | 302.225 | 414/500 |
| 同图配置、QK fusion on | 48.815 | 412.474 | 288.903 | 413/500 |

历史 graph tuned 相对公共基线：平均 TTFT **下降 28.50%**，平均吞吐 **下降 1.58%**，wall **增加 1.85%**。历史纯融合对比：平均吞吐 **提高 1.75%**、wall **下降 4.41%**，但平均 TTFT **增加 0.19%**，且正确数相差 1，因此只作为辅助证据，不作为严格语义等价 A/B。

## 6. Profiler 证据

为了避免 `cudaGraphLaunch` 隐藏独立 kernel，profiling 使用 `VLLM_ENFORCE_EAGER=1`，只切换 QK/MRoPE 融合。请求 3 的 Torch profiler 结果为：

| 指标 | fusion off | fusion on | 变化 |
|---|---:|---:|---:|
| Self CUDA time total | 184.682 ms | 170.403 ms | **-7.73%** |
| 5 样本平均 TTFT | 78.845 ms | 73.895 ms | **-6.28%** |
| 5 样本平均 tok/s | 37.791 | 39.553 | **+4.66%** |

Profiler 关键输出：

```text
fusion off: Self CUDA time total: 184.682ms
fusion on : Self CUDA time total: 170.403ms
_fused_qk_rmsnorm_rope_gate_kernel: 204 calls, 618.323us total, 3.031us/call
_triton_mrope_forward (off):       204 calls, 350.719us total, 1.719us/call
```

不能直接用 618.323 µs 与 350.719 µs 判断 fused kernel 更慢：前者完成 RMSNorm、MRoPE 和 gate copy 的整条链，后者只是未融合链中的 MRoPE 单项。融合收益应结合 Self CUDA time 和端到端 TTFT/吞吐判断。

## 7. 单文件可复现指令

运行前先激活已安装 vLLM、Torch 和 PPU SDK 的 Python 环境。下面只假设当前目录的父目录中存在 `vllm-workspace/`，不依赖其绝对位置。

### 7.1 公共设置

```bash
cd vllm-workspace
source /etc/profile.d/ppu-sdk.sh

export PYTHONPATH="$PWD/vllm-seu"
export DATASET_PATH="${DATASET_PATH:-/mnt/data/datasets/mmbench/mmbench_dev_en.tsv}"
export MODEL_PATH="${MODEL_PATH:-/mnt/data/models/Qwen3.5-2B}"

export VLLM_BASIC=0
export VLLM_SPEC_METHOD=none
export VLLM_DTYPE=float16
export VLLM_ASYNC_SCHEDULING=0
export VLLM_USE_FLASHINFER_SAMPLER=0

# 隔离 QK/MRoPE 时关闭其他融合。
export VLLM_PPU_FUSED_GATE_SILU=false
export VLLM_PPU_FUSED_GDN_PREFILL=false
export VLLM_PPU_FUSED_GDN_DECODE=false

mkdir -p eval/results
```

### 7.2 eager、无 QK/MRoPE 融合

```bash
export VLLM_ENFORCE_EAGER=1
export VLLM_PPU_FUSED_QK_NORM_GATE=false

python eval/benchmark_public.py \
  --dataset-path "$DATASET_PATH" \
  --model-path "$MODEL_PATH" \
  --backend vllm \
  --num-samples 20 \
  --warmup-samples 3 \
  --output eval/results/eager_no_fusion.json
```

### 7.3 eager、只开 QK/MRoPE 融合

```bash
export VLLM_ENFORCE_EAGER=1
export VLLM_PPU_FUSED_QK_NORM_GATE=true

python eval/benchmark_public.py \
  --dataset-path "$DATASET_PATH" \
  --model-path "$MODEL_PATH" \
  --backend vllm \
  --num-samples 20 \
  --warmup-samples 3 \
  --output eval/results/eager_qk_fusion.json
```

### 7.4 CUDA Graph、无 QK/MRoPE 融合

```bash
export VLLM_ENFORCE_EAGER=0
export VLLM_PPU_FUSED_QK_NORM_GATE=false

python eval/benchmark_public.py \
  --dataset-path "$DATASET_PATH" \
  --model-path "$MODEL_PATH" \
  --backend vllm \
  --num-samples 20 \
  --warmup-samples 3 \
  --output eval/results/graph_no_fusion.json
```

### 7.5 CUDA Graph + QK/MRoPE 融合

```bash
export VLLM_ENFORCE_EAGER=0
export VLLM_PPU_FUSED_QK_NORM_GATE=true

python eval/benchmark_public.py \
  --dataset-path "$DATASET_PATH" \
  --model-path "$MODEL_PATH" \
  --backend vllm \
  --num-samples 20 \
  --warmup-samples 3 \
  --output eval/results/graph_qk_fusion.json
```

### 7.6 XRS 全优化运行

```bash
export VLLM_ENFORCE_EAGER=0
export VLLM_PPU_FUSED_QK_NORM_GATE=1
export VLLM_PPU_FUSED_GATE_SILU=1
export VLLM_PPU_FUSED_GDN_PREFILL=1
export VLLM_PPU_FUSED_GDN_DECODE=1

python eval/benchmark_public.py \
  --dataset-path "$DATASET_PATH" \
  --model-path "$MODEL_PATH" \
  --backend vllm \
  --num-samples 20 \
  --warmup-samples 3 \
  --output eval/results/all_optimizations.json
```

`true` 与 `1` 对这些布尔环境变量等价。`VLLM_PPU_FUSED_GDN_DECODE=1` 控制 GDN decode 融合；QK/MRoPE 融合没有单独的 decode 开关，符合条件的 full-attention prefill/decode 都使用同一 dispatch。CUDA Graph 的 decode size 1 已包含在 `cudagraph_capture_sizes` 中。

全优化配置适合最终吞吐测试，不适合归因单个 kernel。环境变量为 true 不代表一定命中，仍需检查启动日志中的 enabled/disabled 条件。

## 8. 最终判断

多模态算子融合实际有效：它在 `text_only=False`、`mrope=(11,11,10)` 的真实多模态请求中启用，profiler 明确出现 204 次 fused kernel 调用；严格 A/B 同时显示 Self CUDA time、TTFT 和吞吐改善，且四组输出逐项一致。

CUDA Graph 的主要价值是改善常驻服务稳态的请求延迟和 kernel 提交效率，代价是冷启动编译/捕获时间与 graph 显存。融合与 CUDA Graph 可以叠加，但性能报告应分别给出冷启动 wall、请求级稳态指标、尾延迟和语义等价性，不能只摘取一个最有利数字。
