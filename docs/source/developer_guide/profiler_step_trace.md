# Step 区间 Trace 采集指南

本文介绍如何在 Ascend `model_runner` 中使用 `torch_npu.profiler` 按固定 step 区间采集 `CPU + NPU` trace。当前插桩位于 [vllm_ascend/worker/model_runner_v1.py](../../../vllm_ascend/worker/model_runner_v1.py)，对应多模态模型如 Qwen-VL 的图模式执行路径。

## 适用场景

当前插桩主要面向以下问题定位场景：

- 多模态 Qwen-VL 推理
- 开启 ACL Graph
- 只采集第 `20` 到第 `23` 个 step
- 导出 profiler trace，并打印每个 step 的总耗时

这里的“一步推理”定义为从 `NPUModelRunner.execute_model()` 进入开始，到 `NPUModelRunner.sample_tokens()` 返回结束。因此日志中打印的耗时是完整单步耗时，包含输入准备、图回放或前向计算、后处理和采样，不只是 forward 耗时。

## 开启方式

在启动服务或离线推理脚本之前，设置以下环境变量：

```bash
export VLLM_ASCEND_TRACE_STEPS_ENABLE=1
export VLLM_ASCEND_TRACE_STEPS_START=20
export VLLM_ASCEND_TRACE_STEPS_END=23
export VLLM_ASCEND_TRACE_DIR=/path/to/profiler_trace
```

其中：

- `VLLM_ASCEND_TRACE_STEPS_ENABLE`：是否开启 step 区间 trace 采集
- `VLLM_ASCEND_TRACE_STEPS_START`：起始 step，默认值为 `20`
- `VLLM_ASCEND_TRACE_STEPS_END`：结束 step，默认值为 `23`
- `VLLM_ASCEND_TRACE_DIR`：trace 输出目录

如果直接使用默认区间 `20-23`，最小配置如下：

```bash
export VLLM_ASCEND_TRACE_STEPS_ENABLE=1
export VLLM_ASCEND_TRACE_DIR=/path/to/profiler_trace
```

## 运行方式

按原有方式启动目标 workload 即可，例如启动多模态 Qwen-VL 图模式请求流，复现需要分析的问题。

profiler 采用延迟启用方式：

- step `20` 开始时启动采集
- step `23` 完成后停止采集

这样可以跳过前面的 warmup step，避免无关 trace 干扰分析。

## 输出结果

trace 文件输出到：

```text
${VLLM_ASCEND_TRACE_DIR}/rank_<rank>/
```

单卡或单 rank 场景下一般为 `rank_0`。该目录下会保存 `torch_npu.profiler` 导出的结果，包括用于时间线分析的 `trace_view.json`。

运行日志中还会打印每个目标 step 的总耗时，例如：

```text
Profile target step 20 total latency: 38.42 ms
Profile target step 21 total latency: 37.95 ms
Profile target step 22 total latency: 38.10 ms
Profile target step 23 total latency: 37.88 ms
torch_npu profiler finished for steps [20, 23]. Per-step latency(ms): 20=38.42, 21=37.95, 22=38.10, 23=37.88
```

## 注意事项

- 当前实现按 step 区间触发，不依赖 request ID。
- 即使模型执行过程中走到提前返回分支，step 也会正常收口并记录耗时。
- 多 rank 场景下，每个 rank 都会写入各自独立的 trace 目录。
- 如果需要调整采集区间，只需修改 `VLLM_ASCEND_TRACE_STEPS_START` 和 `VLLM_ASCEND_TRACE_STEPS_END`。
