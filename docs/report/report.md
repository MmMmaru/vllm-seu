# 真武810e Qwen3.5-2B 推理优化报告
## 项目背景

本报告面向平头哥自研 PPU 芯片真武810e（Zhenwu 810E）上的大模型推理优化，目标是让 Qwen3.5-2B 多模态模型在 MMBench 公开评测集上获得更低的 TTFT 与更高的吞吐，同时保持 Accuracy 与输出语义严格无损。推理框架采用阿里云最新支持的 vLLM v0.23.0 PPU 定制版（vLLM-for-SAIL，ppu-0.23.0 分支）：PPU 平台层复用 CUDA 生态（`dispatch_key="CUDA"`），算子栈以 Triton 与 DeepGEMM 为主，kernel launch 路径与 CUDA 完全一致，可直接用现有工具链做请求级 profiling。

权重默认使用 FP16。PPU 的 CUDA 兼容层对 NVIDIA 优化的 Marlin kernel 无对应加速，W8A16 量化在 PPU 上整体劣于 FP16，暂不作为 PPU 提交配置（本地 NVIDIA 环境下 W8A16 有 15-20% 吞吐提升、准确率下降 <2%，但不可直接迁移）。

请求特征：图像问答类多模态请求，prefill 长度约 300-800 tokens，decode 在非思考模式下仅约 20-40 tokens；思考模式在评测中精度下降，已关闭。

profile 显示 PPU 上推理的主要瓶颈在 CPU 侧 kernel 调度而非设备计算：单次 `cudaLaunchKernel` 的 CPU 开销约 26us，eager 模式下 18 层 GDN 串行 launch 累积成 prefill 的主要 CPU 瓶颈（设备空转 86%，wall 105.8ms / busy 15.2ms）。针对这一特征，优化分三条线推进：(1) CUDA graph 捕获 encoder/decoder，减少 kernel launch overhead；(2) 将 eager 段零散 Triton kernel 融合为专用 kernel（fused gated silu、fused qk norm gate、fused chunked GDN kernel）；(3) 投机解码（n-gram / MTP）摊薄 decode 成本。评测统一以首有效 token 到达时间计 TTFT，并同时核对吞吐与 Accuracy，结论均做交错 A/B 复测。其中 fused qk norm gate 已扩展支持 Qwen3.5 多模态 `(T,H,W)` 三轴 MRoPE（真实多模态请求 `mrope=(11,11,10)` 可命中 fused kernel），CUDA graph 侧为视觉 encoder 增加按输出视觉 token budget 捕获的 Encoder Graph；两类优化相互独立、可叠加。

### 推理框架设置
采用阿里云最新支持的vllm0.23.0镜像开发
权重考虑到PPU上marlin kernel优化问题，暂时未使用。本地W8A16量化下吞吐提升15-20%，准确率维持<2%
prefill长度约为300-800 tokens之间
decode长度在非思考模式下约20-40 tokens
思考模式开启后精度降低，选择关闭思考模式
## 推理框架优化
### CUDA graph优化
通过调整encoder、 decoder的cudagraph capture size，启用cuda graph capture，减少kernel launch overhead
```python
compilation_config={
    "cudagraph_mm_encoder": True,
    "encoder_cudagraph_token_budgets": [64, 128, 256, 384, 512, 640, 768, 896, 1024],
    "cudagraph_capture_sizes": [1,2,4,8,16,24,32,40,48,56,64,128,256,384,448,512],
    "max_cudagraph_capture_size": 512,
    "cudagraph_mode": "FULL_AND_PIECEWISE"
},
```
结果对比

**budgets 语义**：`encoder_cudagraph_token_budgets` 是视觉 encoder 预捕获的**输出视觉 token 尺寸列表**（不是生成 token 上限）：Qwen3.5 每张图输出视觉 token 数 = `t·(h//spatial_merge_size)·(w//spatial_merge_size)`，224×224（patch=14、merge=2）为 16×16 → 8×8 = 64 个视觉 token；运行时按"最小可容纳"选择 graph（64→64、65→128、…、1025→无匹配回退 eager），输入复制/padding 到固定 graph buffer 后 replay，再去 padding 恢复原图顺序合入语言模型输入。

`cudagraph_mode=FULL_AND_PIECEWISE`：满足条件的统一 decode batch 优先走 FULL graph，prefill/混合形状走 PIECEWISE，无法匹配安全回退 eager。budget 越密 padding 浪费越少，但捕获时间与 graph 显存越大。图模式为独立进程冷启动，首次 `torch.compile`/capture 计入 wall 但不计入请求级 TTFT/吞吐（单个 compile range 约 25s），稳态评估应使用常驻 EngineCore 预热。

| setup | sample | en TTFT(ms) | en 吞吐(tps) | en Acc |
|---|---|---|---|---|
| vllm 默认参数(none) 最基础 | 200 | 63.895 | 45.658 | 0.770 | eager 无优化（VLLM_BASIC=1） |
| vllm 无 kernel + cuda graph | 200 | 35.973 | 354.749 | 0.765 | 四 kernel 全关 |

### 投机解码优化
Qwen3.5-2B支持MTP，在英文数据集下对比n-gram及MTP效果，通过消融实验确定投机解码方案。  
结果对比
| setup | sample | en TTFT(ms) | en 吞吐(tps) | en Acc |
|---|---|---|---|---|
| vllm ngram | 200 | 29.494 | 687.279 | 0.765 |
| vllm MTP=3 | 200 | 36.601 | 402.895 | 0.765 |

## 融合kernel设计
### fused chunked GDN kernel
#### 前后对比profiling结果
![baseline eager](png/image-2.png)
*图1：baseline eager 的 profiling 结果*
![fused](png/image-3.png)
*图2：fused 的 profiling 结果*

#### 问题定位

GDN（Gated DeltaNet）层 prefill 默认路径走 flash-linear-attention（FLA）的 eager 六段 Triton 链 `chunk_gated_delta_rule_fwd`：`chunk_local_cumsum` → `chunk_scaled_dot_kkt_fwd` → `solve_tril` → `recompute_w_u_fwd` → `chunk_gated_delta_rule_fwd_h` → `chunk_fwd_kernel_o`。每层约 10 次 kernel launch，外加上层 `ChunkGatedDeltaRuleFunction` 的 autograd.Function / input_guard / custom_fwd 三层包装与每段中间 tensor 的 `.new_empty` 分配。PPU 上单次 `cudaLaunchKernel` 的 CPU 开销约 26us，18 层串行累积后构成 prefill 的主要 CPU 瓶颈（prefill 段设备空转 86%）。

#### kernel 设计方案

新增 `vllm/model_executor/layers/fla/ops/chunk_fused.py`，把前 4 段（chunk 局部、chunk 间完全并行）融合为单个 Triton kernel `fused_chunk_intra`，grid `(NT, B*H)`，一个 program 处理一个 chunk 的头，中间矩阵 A/Ai 完全驻留寄存器：

1. **g 块内 cumsum**：gate 在 chunk 内做 fp32 前缀和（对应 `chunk_local_cumsum`）；
2. **A 矩阵**：`A = strict_lower(β·K·Kᵀ·exp(g_i − g_j))`，fp32 累加，仅在寄存器中构建（对应 `chunk_scaled_dot_kkt_fwd`）；
3. **三角求逆**：`Ai = (I + A)⁻¹`，64×64 下三角求逆按 16×16 分块在寄存器内完成——对角线分块逐行前代消元、非对角分块按块回代，算法逐行移植自 `solve_tril.py` 的 `merge_16x16_to_64x64_inverse_kernel`；求逆后按原链路精度 fp32→k.dtype（rtne 舍入）落盘，对齐 eager 链 HBM 往返的舍入行为；
4. **WY 参数**：`u = Ai·(β·V)`、`w = Ai·(β·K·exp(g_cu))`，与 `wy_fast.py` 一致。

中间矩阵 `A`/`Ai`（各 `[T,H,64]` fp32）不再落 HBM；⑤ `chunk_delta_h`（跨 chunk 串行状态扫描）与 ⑥ `chunk_o`（输出）无融合余地，原样复用。数学上即 Gated DeltaNet 的 WY 表示分块并行形式。

接入方式：`ChunkGatedDeltaRule.forward_fused` 推理专用路径，直连三个前向函数、绕开 autograd 包装；`VLLM_PPU_FUSED_GDN_PREFILL=1` 开启（默认开），K/V≠128 时报警告并自动 fallback 到原生路径。

精度与收益：对拍测试 `tests/kernels/test_fused_chunk_intra.py` 本地 RTX 4060 与 PPU 均 16/16 通过，与 eager 链位级一致（PPU 最大差 4.4e-16）；微基准（T=340、Hg=16、H=32，PPU）CPU enqueue 0.194 → 0.059 ms/层（-70%）、device 0.188 → 0.150 ms（-20%）；端到端 mmbench en100 × 2 轮 TTFT 45.3 → 36.5 ms（**-19.4%**），Accuracy 0.77 无损。

#### 前后 kernel 调用链对比（每 GDN 层 prefill）

融合前（eager）：
```text
conv1d ─→ fused_post_conv_prep（拆 q/k/v/g/beta，k l2norm）
  └→ chunk_gated_delta_rule（autograd 三层包装）
       ├─ ① chunk_local_cumsum
       ├─ ② chunk_scaled_dot_kkt_fwd
       ├─ ③ solve_tril（merge_16x16_to_64x64_inverse_kernel）   ← A/Ai 落 HBM
       ├─ ④ recompute_w_u_fwd                                  ← 再读回
       ├─ ⑤ chunk_gated_delta_rule_fwd_h
       └─ ⑥ chunk_fwd_kernel_o
```

融合后（fused）：
```text
conv1d ─→ fused_post_conv_prep
  └→ chunk_gated_delta_rule（forward_fused 直连，绕开 autograd）
       ├─ ①+②+③+④ fused_chunk_intra（单 kernel，A/Ai 寄存器驻留，零 HBM 往返）
       ├─ ⑤ chunk_gated_delta_rule_fwd_h（复用）
       └─ ⑥ chunk_fwd_kernel_o（复用）
```

| 项目 | 融合前 | 融合后 |
|---|---|---|
| GDN prefill kernel 数/层 | 6（加 q/k l2norm 共约 10 次 launch） | 3（共约 5 次 launch） |
| 中间张量 A/Ai | `[T,H,64]` fp32 写 + 读各一次 | 寄存器驻留，零 HBM 往返 |
| autograd 包装 | Function/input_guard/custom_fwd | 直连 forward_fused，绕开 |
| CPU enqueue/层 | 0.194 ms | 0.059 ms（-70%） |
| device 时间/层 | 0.188 ms | 0.150 ms（-20%） |
| 端到端 TTFT | 35.7 ms | 30.3 ms（-19.4%，acc 无损） |
### fused gated silu kernel
![alt text](png/image-5.png)
*fused 前的profiling*
![alt text](png/image-2.png)
*fused 后的profiling*

#### 优化目标与范围
单 token decode 中 MLP 先算 gate/up 投影再执行 SiLU×up（SwiGLU），两段直接数据依赖，可由一个 kernel 完成，避免中间结果写回显存再读回。本融合指 **gate_up 投影 + SwiGLU 融合**（vLLM 原版已把 gate/up 两套权重合并为一次投影）。适用条件：BF16、TP=1、无量化/LoRA/bias、输入与权重连续且 16 字节对齐。原生融合形状：`x[1,2048]`、`weight[12288,2048]` → `h[1,6144]`；M=1 指算子实际输入只有一行（Graph padding 后 M>1 不命中）。不包含 `down_proj`、GDN、attention、LM head、视觉编码；其他形状由上层回退，直接调用原生算子会报错。

#### 计算内容与接入
令 D=6144，合并权重前 D 行为 gate、后 D 行为 up：`gate = x@W_gate.T`、`up = x@W_up.T`、`h = SiLU(gate)*up`、`y = h@W_down.T`（down_proj 保留在融合外）。

```text
Qwen3_5MLP.forward
  ├─ eager 满足条件 → torch.ops._C.ppu_gate_silu
  └─ 编译路径 → torch.ops.vllm.ppu_gate_silu
                   ├─ 满足 M=1 等条件 → torch.ops._C.ppu_gate_silu
                   └─ 不满足 → Linear + 激活回退
      原生融合入口 → C++ ppu_gate_silu → vllm_ppu_gate_silu_launch
                    → gate_silu_fused_pair_w4_kernel
  → 原 vLLM down_proj
```

C++ 桥接检查设备/BF16/形状/连续性与对齐、分配输出、取当前 stream 传给 HGGC launcher；异步提交、检查启动错误，不做设备全局同步。

#### M=1 局限与 prefill 路由（避免 TTFT 回退）
当前实现只支持严格 M=1，主要覆盖首 token 之后的 decode；prefill 通常 M>1 不命中。旧接入在 prefill 虽不启动融合 kernel，却进入 custom op 后 fallback，破坏 Inductor 对 SiLU×up 的原融合：24 层 MLP 净增 24 次 launch，稳态 TTFT 由 50.945ms 增至 55.669ms（**+9.27%**），language/prefill 由 24.535ms 增至 27.628ms（**+12.61%**）。必须采用路由：

```text
M = 1 decode  → Gate-Up 融合 kernel
M > 1 prefill → custom op 外直接使用原 vLLM/Inductor 路径
```

该路由可避免 TTFT 回退，但现有 M=1 kernel 本身不能优化 TTFT；真正优化 TTFT 需重新实现支持多 token GEMM 的 prefill 融合 kernel。

#### PPU kernel 并行设计
`csrc/ppu/gate_silu_fused.hg`，设备函数 `gate_silu_fused_pair_w4_kernel`（SDK HGGC 编译模式 + CUDA 风格线程、向量加载、shuffle）：

| 参数 | 当前实现 |
| --- | --- |
| 每 block 线程数 | 128（4 个 32-lane warp） |
| 每 block 输出数 | 2，总 block 数 6144/2 = 3072 |
| 每 lane 每轮读取 | 8 个 BF16（`uint4` 载体，16 字节） |
| K 维循环 | 步长 32×8=256，K=2048 共 8 轮 |
| 点积累加 | FP32；`__shfl_down_sync` 归约 → lane 0 写共享内存 |

warp 配对（`output_index = blockIdx.x*2 + (warp&1)`，`row = output_index + (warp>=2 ? 6144 : 0)`）：warp 0/1 计算两个 gate 行，warp 2/3 计算对应 up 行；block 内一次同步后 warp 0/1 的 lane 0 将 gate 与对应 up 配对，执行 SiLU 激活、逐元素乘法并写出。本卡 L2 为 64 MiB 而单份目标权重仅 48 MiB，重复访问权重受缓存影响；历史 device 基准使用轮换真实层权重降低该偏差。

#### 实测结果（compile + CUDA Graph）
两版均为 Inductor mode=3、`FULL_DECODE_ONLY`、`capture_sizes=[1]`、`enforce_eager=False`；纯文本、并发 1、固定输出 64 tokens、关 prefix cache；独立引擎 A→B，各预热 5 次、测 30 次，无 profiler，排除加载/编译/捕获时间；每版正式 Graph 回放 1890 次。

| 指标（请求中位数） | 原路径 | 融合路径 | 变化 |
| --- | ---: | ---: | ---: |
| 引擎 TTFT | 26.828 ms | 29.511 ms | 慢 10.00% |
| 引擎 TPOT | 2.97224 ms/token | 2.88664 ms/token | 快 2.88% |
| Decode 吞吐 | 336.446 token/s | 346.423 token/s | 提高 2.97% |
| 完整请求主机耗时 | 214.132 ms | 211.780 ms | 降低 1.10% |

TPOT = (末 token 时间 − 首 token 时间)/63，decode 吞吐为其倒数，不是多并发服务总吞吐。融合后 30 次输出稳定且与基线 token 一致。TTFT 变慢与其 M=1 定位一致：decode 融合不能优化 TTFT，收益集中在 TPOT/decode 吞吐。

#### 使用与核验
在创建引擎和 Graph 捕获之前设置 `export VLLM_PPU_FUSED_GATE_SILU=1`（默认关闭）。关闭时用 0 并重新创建引擎；已捕获的 Graph 不会因之后修改 Python 开关而自动换路径。确认 Python 实际加载的是编译了 `ppu_gate_silu` `_C` 扩展的 PPU 版 vLLM（仅下载源码不代表安装环境已更新）。当前验证仅覆盖 M=1/Graph 配置，不代表所有 batch、图模式或多卡均已验证。

### fused qk norm gate kernel
![alt text](png/image-1.png)
*baseline eager 的 profiling 结果*
![alt text](png/image-4.png)
*fused 后的 profiling 结果*

#### 多模态 QK/MRoPE 融合设计
把 `split → Q/K RMSNorm → MRoPE → gate copy` 合并为单个 Triton kernel（`vllm/model_executor/layers/fused_qk_norm_rope.py`），同时支持纯文本 1D RoPE 与 Qwen3.5 多模态 `(T,H,W)` 三轴 MRoPE；仅作用于 full-attention 层，不作用于 linear_attention/GDN 层。

- **启用条件**（启动日志逐项校验）：`VLLM_PPU_FUSED_QK_NORM_GATE=true` + `attn_output_gate=True` + NeoX-style RoPE + CUDA/PPU 兼容后端 + RoPE dtype FP16/BF16 +（纯文本，或多模态 MRoPE：`MRotaryEmbedding`、3 段 section、section 和 = `rotary_dim/2`、interleaved 布局，如 `mrope=(11,11,10)`）。旧实现仅接受纯文本、多模态会禁融合；条件升级为 `(text_only or supports_mrope)` 后真实多模态请求可命中。
- **kernel 设计**：grid `(n_tokens, num_q_heads + num_kv_heads)`，一个 Triton program 处理一个 token 的一个 Q/K head。RMSNorm 平方和、rsqrt 与权重乘法在 FP32 完成，`weight + 1` 保持 GemmaRMSNorm 语义；归一化结果先回输入 dtype 再转 FP32 做 RoPE，对齐未融合路径的存储舍入；`[0, rotary_dim)` 做 partial RoPE，`[rotary_dim, head_dim)` 只做 RMSNorm；Q head 在同一 program 内复制 gate，K head 不复制；`v` 不参与融合。
- **多模态位置选择**：`is_h = (rot_offs%3==1) & (rot_offs < 3·section_h)`、`is_w = (%3==2) & (< 3·section_w)`，命中用 H/W 轴，否则用 T 轴。
- 未融合需要 2×RMSNorm + MRoPE 及多次拆分/复制；融合路径单次 Triton launch 完成整条链，减少中间 tensor 往返显存。
- 环境变量必须在 Engine 初始化和图捕获**之前**设置；图捕获完成后修改不改变已捕获 graph 内容。



## 评测结果
| setup | sample | en TTFT(ms) | en 吞吐(tps) | en Acc |
|---|---|---|---|---|
| **ngram+cudagraph+融合kernel（完整配置）en 全量** | 4029 | 29.240 | 780.153 | 0.796 |
| **ngram+cudagraph+融合kernel（完整配置）cn 全量** | 4029 | 29.232 | 302.436 | 0.834 |
