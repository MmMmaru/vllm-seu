# exps：PPU benchmark 对比（2026-08-31）

## 环境

- 远程 PPU：`remote run ppu`，workspace `/root/xrs/vllm-workspace`，单卡
- 代码：当前工作区 `vllm-seu` 源码树（`PYTHONPATH=vllm-seu`），含 PPU 融合 kernel
  （fused Gate+SiLU 默认关、fused QK-norm+RoPE+gate、fused GDN prefill），CUDA graph
  开（`cudagraph_mode=FULL_AND_PIECEWISE`），参照 `scripts/run_eval.sh`
- 模型：`/mnt/data/models/Qwen3.5-2B`（FP16）；数据集 `mmbench_dev_{en,cn}.tsv`
  （en/cn 实际各 4029 条有效样本；文件含约 3700 行非数据行）
- 评测入口：`eval/benchmark_public.py`（`--warmup-samples 3`、greedy、max_new_tokens=64）
- 口径：TTFT=首个流式 delta token 到达时间；吞吐=decode 窗口 tokens/s；
  Accuracy=答案解析正确率；`public_validation_passed`（全量集上少量样本因
  输出不含 A/B/C/D 判定 `missing_choice_answer`，为固定样本、轮间可复现）

## 规则

- 非完整配置跑 `mmbench_dev_en` 200 条 × 3 轮取均值（r1/r3 正序、r2 反序交错）
- 完整配置（ngram + CUDA graph + 融合 kernel）跑全量 en/cn（各 4029 条）× 3 轮取均值
- 默认参数(none) = 最基础 vLLM 配置：`VLLM_BASIC=1`（eager、无 cuda graph、
  无 compilation/cudagraph_mm_encoder、同步调度、无投机）+ 四 kernel 开关全关
- 三个 custom kernel 消融实验：`VLLM_PPU_FUSED_{GATE_SILU,QK_NORM_GATE,GDN_PREFILL}`
  逐个开启（graph 模式，`VLLM_PPU_FUSED_GDN_DECODE=0` 控制，spec=none），
  各 en200 × 3 轮；基线 = 四开关全关（en200_nokernel）

## setups

```text
transformers backend
vllm backend，默认参数 = 最基础配置（VLLM_BASIC=1：eager + 无任何优化，spec=none）
vllm backend，不使用自定义kernel，优化cuda graph（四 kernel 开关全关 + graph）
vllm backend，使用ngram（VLLM_SPEC_METHOD=ngram）
vllm backend，使用MTP=3（VLLM_SPEC_METHOD=mtp VLLM_SPEC_NUM_TOKENS=3）
vllm backend，使用MTP=3 + streaming（+ VLLM_MTP_STREAM_OUTPUT=1）
vllm + 三个kernel消融实验（GATE_SILU / QK_NORM_GATE / GDN_PREFILL 逐个开）
```

## 结果（三轮均值）

| setup | sample | en TTFT(ms) | en 吞吐(tps) | en Acc | 备注 |
|---|---|---|---|---|---|
| transformers | 200 | 177.272 | 61.496 | 0.740 | HF bf16 顺序生成 |
| vllm 默认参数(none) 最基础 | 200 | 63.895 | 45.658 | 0.770 | eager 无优化（VLLM_BASIC=1） |
| vllm 无 kernel + cuda graph | 200 | 35.973 | 354.749 | 0.765 | 四 kernel 全关 |
| vllm ngram | 200 | 29.494 | 687.279 | 0.765 | 吞吐 +103% vs 无 kernel |
| vllm MTP=3 | 200 | 36.601 | 402.895 | 0.765 | TTFT 最高 |
| vllm MTP=3 + streaming | 200 | 29.223 | 385.141 | 0.765 | TTFT -7.4ms vs 非 stream |
| **ngram+cudagraph+融合kernel（完整配置）en 全量** | 4029 | 29.240 | 780.153 | 0.796 | 12 条 missing_choice_answer |
| **ngram+cudagraph+融合kernel（完整配置）cn 全量** | 4029 | 29.232 | 302.436 | 0.834 | 45 条 missing_choice_answer |

所有 vllm 200 条轮次 `public_validation_passed=true`，Accuracy 与历史基线一致（0.765）；
最基础配置 acc 0.770（200 条子集内 1-2 题波动）。

## 历史参考（2026-08-27，同代码口径，en200×2 轮）

```text
                          TTFT(ms)   吞吐(tps)  Acc
none  r1/r2               36.356/35.196  353.2/356.5  0.765
ngram r1/r2               33.975/33.256  669.8/669.2  0.765
```

## 三个 kernel 消融实验（en200 × 3 轮，graph 模式，GDN_DECODE=0）

基线 = 四 kernel 全关（= 上表 en200_nokernel）；逐核开启对比：

| 配置 | TTFT(ms) | 吞吐(tps) | Acc | vs 基线 TTFT |
|---|---|---|---|---|
| 基线（全关） | 35.973 | 354.749 | 0.765 | - |
| + fused Gate+SiLU（GATE_SILU=1） | 37.050 | 358.388 | 0.765 | +1.1 ms |
| + fused QK-norm+RoPE+gate（QK_NORM_GATE=1） | 37.787 | 357.452 | 0.765 | +1.8 ms |
| + fused GDN prefill（GDN_PREFILL=1） | 30.267 | 340.142 | 0.765 | **-5.7 ms（-16%）** |
| + 三个全部开启 | 30.380 | 341.937 | 0.765 | -5.6 ms |
| 默认组合（QK_NORM_GATE+GDN_PREFILL+GDN_DECODE） | 29.877 | 338.124 | 0.765 | -6.1 ms |

结论：
- **fused GDN prefill kernel 是 TTFT 收益的唯一主要来源**（-16%），全量默认组合的
  -6.1ms 收益基本全部来自它（GDN_DECODE 额外约 -0.4ms）。
- fused Gate+SiLU 与 fused QK-norm+RoPE+gate 单开对 TTFT 呈轻微回退（+1~2ms，
  三轮稳定，非噪声），吞吐 +1% 左右；在 GDN prefill 之上再开启无叠加收益
  （all3 30.38 ≈ 仅 GDN prefill 30.27）。当前代码下二者已可正常入 CUDA graph
  （`ppu_gate_silu` fake impl + 移除 enforce_eager 限制后验证通过），其价值主要
  体现在 kernel 级 launch 数/耗时（见下表 profiling），端到端以 GDN prefill 为主导。
- 消融各轮 acc 均 0.765，无准确率回归。

## 结论要点

- 最基础配置（eager 无优化）TTFT 63.9ms / 45.7tps；仅 cuda graph（无 kernel）
  TTFT 36.0ms / 354.7tps；+ 融合 kernel 组合 TTFT 29.9ms —— 优化栈累计
  TTFT -53%、吞吐 ×7.4。
- ngram 相对无 kernel 基线：吞吐 355 → 687 tps（+94%），TTFT 持平（29.5ms vs
  36.0ms 是在完整 kernel 组合上的对比），准确率无损。
- MTP=3 非 stream TTFT 回退（36.6 ms）；+streaming 后 TTFT 29.2 ms（-7.4 ms），
  吞吐 385 tps（略低于非 stream 403，符合 draft 线程竞争的既有结论）。
- transformers 基线显著慢（TTFT 177 ms、61.5 tps），Acc 0.740（200 条子集波动）。
- 完整配置全量：en 0.796 / cn 0.834 Acc，en 吞吐 780 tps 与 200 条子集口径
  （687 tps）量级一致；cn 吞吐 302 tps（中文输出更长/解码窗口更大）。

## profiling 对比（en10，request-3，torch profiler）

```text
模式                                request-3 trace 大小   备注
eager mode（无 kernel 优化）          10.5 MB (gz)          逐 kernel 记录
cuda graph（只开两个可入图 kernel）     1.6 MB (gz)          graph replay 紧凑
```

- P1：`VLLM_ENFORCE_EAGER=1` + `VLLM_PPU_FUSED_*` 四开关全关 + `VLLM_SPEC_METHOD=none`
  （profile 开销下该 10 条 TTFT 89.7ms / 44.2 tps）
- P2：CUDA graph 开 + `VLLM_PPU_FUSED_QK_NORM_GATE=1` + `VLLM_PPU_FUSED_GDN_PREFILL=1`
  + `VLLM_PPU_FUSED_GATE_SILU=0`（fused gate silu 无法入图，未启用；TTFT 32.9ms / 329.4 tps）
- trace 产物：`.temp/exps/profile_eager_nokernel/`、`.temp/exps/profile_graph_2kernel/`
  （含 `request-3_rank0.*.pt.trace.json.gz`）

## 详细结果（三轮均值 JSON）

```json
{
  "backend": "transformers",
  "rounds": 3,
  "sample_count": 200,
  "avg_ttft_ms": 177.272,
  "avg_throughput_tokens_per_sec": 61.496,
  "accuracy": 0.74,
  "public_validation_passed": true
}
transformers backend

{
  "backend": "vllm",
  "rounds": 3,
  "sample_count": 200,
  "avg_ttft_ms": 29.877,
  "avg_throughput_tokens_per_sec": 338.124,
  "accuracy": 0.765,
  "public_validation_passed": true
}
vllm 默认参数（none）

{
  "backend": "vllm",
  "rounds": 3,
  "sample_count": 200,
  "avg_ttft_ms": 35.973,
  "avg_throughput_tokens_per_sec": 354.749,
  "accuracy": 0.765,
  "public_validation_passed": true
}
vllm 不使用自定义kernel，优化cuda graph

{
  "backend": "vllm",
  "rounds": 3,
  "sample_count": 200,
  "avg_ttft_ms": 29.494,
  "avg_throughput_tokens_per_sec": 687.279,
  "accuracy": 0.765,
  "public_validation_passed": true
}
vllm 使用ngram

{
  "backend": "vllm",
  "rounds": 3,
  "sample_count": 200,
  "avg_ttft_ms": 36.601,
  "avg_throughput_tokens_per_sec": 402.895,
  "accuracy": 0.765,
  "public_validation_passed": true
}
vllm 使用MTP=3

{
  "backend": "vllm",
  "rounds": 3,
  "sample_count": 200,
  "avg_ttft_ms": 29.223,
  "avg_throughput_tokens_per_sec": 385.141,
  "accuracy": 0.765,
  "public_validation_passed": true
}
vllm 使用MTP=3 + streaming

{
  "backend": "vllm",
  "rounds": 3,
  "sample_count": 4029,
  "avg_ttft_ms": 29.24,
  "avg_throughput_tokens_per_sec": 780.153,
  "accuracy": 0.795979,
  "public_validation_passed": false
}
完整配置（ngram+cudagraph+融合kernel）全量 en（12 条 missing_choice_answer）

{
  "backend": "vllm",
  "rounds": 3,
  "sample_count": 4029,
  "avg_ttft_ms": 29.232,
  "avg_throughput_tokens_per_sec": 302.436,
  "accuracy": 0.833706,
  "public_validation_passed": false
}
完整配置（ngram+cudagraph+融合kernel）全量 cn（45 条 missing_choice_answer）

## 各轮明细
| 文件 | TTFT(ms) | tps | acc | passed |
|---|---|---|---|---|
| abl_all3_r1.json | 30.667 | 344.117 | 0.765 | True |
| abl_all3_r2.json | 30.252 | 341.312 | 0.765 | True |
| abl_all3_r3.json | 30.222 | 340.381 | 0.765 | True |
| abl_gatesilu_r1.json | 36.54 | 359.629 | 0.765 | True |
| abl_gatesilu_r2.json | 37.229 | 357.068 | 0.765 | True |
| abl_gatesilu_r3.json | 37.381 | 358.467 | 0.765 | True |
| abl_gdnprefill_r1.json | 31.134 | 340.069 | 0.765 | True |
| abl_gdnprefill_r2.json | 29.847 | 338.535 | 0.765 | True |
| abl_gdnprefill_r3.json | 29.819 | 341.821 | 0.765 | True |
| abl_qknorm_r1.json | 39.107 | 358.696 | 0.765 | True |
| abl_qknorm_r2.json | 37.35 | 356.292 | 0.765 | True |
| abl_qknorm_r3.json | 36.904 | 357.369 | 0.765 | True |
| basic_r1.json | 62.758 | 46.597 | 0.77 | True |
| basic_r2.json | 64.179 | 45.716 | 0.77 | True |
| basic_r3.json | 64.747 | 44.661 | 0.77 | True |
| en200_mtp3_r1.json | 36.54 | 405.026 | 0.765 | True |
| en200_mtp3_r2.json | 36.098 | 401.264 | 0.765 | True |
| en200_mtp3_r3.json | 37.165 | 402.396 | 0.765 | True |
| en200_mtp3s_r1.json | 29.402 | 383.004 | 0.765 | True |
| en200_mtp3s_r2.json | 29.214 | 386.862 | 0.765 | True |
| en200_mtp3s_r3.json | 29.053 | 385.558 | 0.765 | True |
| en200_ngram_r1.json | 29.444 | 686.331 | 0.765 | True |
| en200_ngram_r2.json | 29.045 | 688.195 | 0.765 | True |
| en200_ngram_r3.json | 29.994 | 687.312 | 0.765 | True |
| en200_nokernel_r1.json | 35.561 | 355.808 | 0.765 | True |
| en200_nokernel_r2.json | 35.818 | 352.556 | 0.765 | True |
| en200_nokernel_r3.json | 36.541 | 355.883 | 0.765 | True |
| en200_none_r1.json | 30.463 | 339.416 | 0.765 | True |
| en200_none_r2.json | 29.638 | 335.306 | 0.765 | True |
| en200_none_r3.json | 29.53 | 339.65 | 0.765 | True |
| en200_tf_r1.json | 177.867 | 61.144 | 0.74 | True |
| en200_tf_r2.json | 175.239 | 62.133 | 0.74 | True |
| en200_tf_r3.json | 178.709 | 61.211 | 0.74 | True |
| full_cn_ngram_r1.json | 29.192 | 300.292 | 0.833706 | False (45) |
| full_cn_ngram_r2.json | 29.268 | 304.037 | 0.833706 | False (45) |
| full_cn_ngram_r3.json | 29.237 | 302.979 | 0.833706 | False (45) |
| full_en_ngram_r1.json | 29.291 | 777.704 | 0.795979 | False (12) |
| full_en_ngram_r2.json | 29.268 | 779.03 | 0.795979 | False (12) |
| full_en_ngram_r3.json | 29.161 | 783.726 | 0.795979 | False (12) |
