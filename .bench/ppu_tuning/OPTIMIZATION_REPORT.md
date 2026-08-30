# Qwen3.5-2B 图文评测合法优化检查

约束来源：`eval/README.md`。所有有效对比均使用相同原始
`Qwen3.5-2B` 权重、同一个 500 样本图文公开集、相同采样和评分脚本，并使用
独立冷缓存。没有修改题目、答案、计时或 token 统计。

## 推荐候选：视觉 encoder + multimodal prefill CUDA Graph

候选文件：`.bench/ppu_tuning/evaluation_wrapper_candidate.py`

相对基线 `result_public_pr44176_verified_500.json`：

| 指标 | 基线 | 候选 | 变化 |
|---|---:|---:|---:|
| 平均 TTFT | 67.870 ms | 48.526 ms | -28.50% |
| TTFT P50 | 62.057 ms | 44.486 ms | -28.31% |
| TTFT P95 | 79.612 ms | 50.659 ms | -36.37% |
| TTFT P99 | 346.134 ms | 211.357 ms | -38.94% |
| 平均解码吞吐 | 412.541 tok/s | 406.016 tok/s | -1.58% |
| 解码吞吐 P50 | 377.523 tok/s | 377.967 tok/s | +0.12% |
| 准确率 | 413/500 | 414/500 | +1 题 |
| 公开校验失败 | 0 | 0 | 相同 |
| 冷启动总耗时 | 297.364 s | 302.851 s | +1.85% |

候选捕获 256--1024 视觉 token 的 encoder 图，并把 PIECEWISE prefill 图扩展到
576 token。公开集实际视觉 token 为 256--1024、prompt 为 116--553。
私有集超过这些范围时 vLLM 会安全回退 eager，不会跳过或截断样本。

原始候选结果：`result_cudagraph_tuned_cold_500.json`。
日志：`.bench/ppu_tuning/logs/cudagraph_tuned_cold.log`。
最小补丁：`.bench/ppu_tuning/evaluation_wrapper_cudagraph.patch`。

## 已排除

- `max_model_len=640, max_num_seqs=1`：TTFT 70.767 ms、吞吐
  409.849 tok/s、总耗时 298.911 s，没有收益。
- 禁用 FlashInfer autotune：TTFT 70.062 ms、吞吐 407.956 tok/s、总耗时
  310.267 s，没有收益。
- W8A16 权重：`eval/README.md` 禁止使用非主办方认可的量化权重，已终止，不是
  有效候选。
- 跨次持久化编译缓存：README 明确禁止依赖它获取复测收益，因此不作为优化。

## 隔离复现

以下脚本为每次运行创建临时冷缓存，并在退出时删除，避免跨次缓存获利：

```bash
cd /root/zsx/vllm-seu_test3
bash .bench/ppu_tuning/run_candidate_cold.sh
```

主项目的 `eval/evaluation_wrapper.py` 尚未被候选覆盖。

## WSL 原始 CUDA Graph 基线核对

WSL `/home/zsx/vllm-seu/eval/evaluation_wrapper.py` 与
`/home/zsx/vllm-seu_test3/eval/evaluation_wrapper.py` 均未传入
`compilation_config`。以原项目参数创建 engine config 后，两者实际解析为：

- `cudagraph_mode=FULL_AND_PIECEWISE`
- `cudagraph_capture_sizes=[1, 2, 4, 8, 16, 24, 32]`
- `max_cudagraph_capture_size=32`
- `compile_mm_encoder=False`
- `cudagraph_mm_encoder=False`

因此远程优化前基线采用上述 vLLM 默认配置，并固定在
`.bench/ppu_tuning/baseline/` 隔离副本中。WSL test2 的 README 第 133--138 行
记载了曾把 prefill 捕获尺寸扩大到 384 的实验，但该配置并不在 test2 当前
wrapper 或运行脚本中，不能视为 test2 当前默认配置。
