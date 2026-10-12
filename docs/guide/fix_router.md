# HyperParallel fix-router 特性原理

fix-router 用确定的循环分配替换 MoE 的专家选择，减少数据相关的专家负载波动，便于比较通信、专家计算和内存占用。
它仍执行真实 gate 和 Top-K，并从新选专家读取真实分数，因此 gate 保留梯度和计算开销。开关默认关闭；开启后用于调试和性能分析，
不用于收敛、模型精度或真实路由吞吐验收。

## MindFormers 参考实现

参考版本为 MindFormers 提交 `44a47972f9ebd5ba746433588b99ed2068d419d7`。关键位置：

- `mindformers/pynative/transformers/moe/router.py` 的 `TopKRouter` 和 `_debug_force_load_balance_routing`。
- `mindformers/parallel_core/transformer_config.py` 的 `moe_router_force_expert_balance`，默认 `False`。

设本地 token 数为 `T`，每个 token 选择 `K` 个专家，全局专家数为 `E`。将 `[T, K]` 展开为路由槽位，
槽位编号 `i` 对应专家 `i % E`。例如 `T=3, K=2, E=4` 时得到 `[[0,1], [2,3], [0,1]]`。
专家接收量只可能是 `floor(T*K/E)` 或 `ceil(T*K/E)`；只有 `T*K` 能被 `E` 整除时才完全相等。

替换后的权重为 `scores.gather(1, fixed_indices)`，其中 `scores` 是原始 logits 经模型评分函数变换后的无偏分数。
随后继续执行原模型的归一化和缩放。专家偏置和 group mask 只参与原 Top-K 选择，不进入固定专家的 combine 权重。
实现通过原选择张量乘零后加固定编号，保留选择张量携带的布局线索，再读取分数。

## HyperParallel 的实现

公共函数 `fixed_expert_indices` 生成编号，`fixed_topk_routing` 替换编号并读取匹配权重。
二者位于 `hyper_parallel.components.functional.fix_router`，并由 functional 包导出。
热路径使用设备上的 `arange`、取余和 `gather`，不读取 tensor 到 CPU，不增加集合通信，也不缓存跨 microbatch 的编号。

默认使用 MindFormers 的专家顺序。已有 DeepSeek-V3 和 Kimi EP 基准入口保留目标 rank 优先顺序：
每个 EP rank 连续拥有 `L` 个专家，目标数 `Q=E/L`，槽位 `i` 对应的专家为：

```text
destination = i % Q
local_expert = (i // Q) % L
expert = destination * L + local_expert
```

例如 `E=8, L=2` 时，一个完整周期是 `0,2,4,6,1,3,5,7`。它保留原 EP 分配顺序，修正原实现只改编号却保留旧 Top-K 权重的问题。
新权重按固定专家重新读取，之后执行原有 `norm_topk_prob` 和 `routed_scaling_factor`。
这使梯度落在实际被调度的专家分数上；开启固定路由后的历史 loss 和 grad norm 数值会因此改变。

## 负载边界

两种顺序的单个发送 rank 专家负载差均不超过一，`K<=E` 时每个 token 内的专家不重复。
目标 rank 优先顺序在 `T*K` 能被 `Q` 整除时，每个发送 rank 向各目标发送相同槽位数；有余数时差最多一。
多个发送 rank 都从槽位零重新开始，余数可能累积到同一目标，因此不能宣称任意 batch、任意分片下全局 rank 负载完全相等。

固定路由以每次调用的本地 token 顺序为准。改变 TP、CP、EP 的 token 切分会改变全局 token 对应的专家编号，
不保证不同并行拓扑与未分片模型的输出一致。张量专家维需要完整无偏分数；自动分片下应在 router 本地计算边界调用公共函数。
empty batch 返回 `[0,K]`；小 batch 可以有零负载专家。容量裁剪等下游策略仍可能进一步改变最终负载。

## 支持入口

| 入口 | 开关 | 固定编号顺序 | 权重规则 |
| --- | --- | --- | --- |
| `TokenChoiceTopKRouter` | `fix_router=True` | 专家顺序 | 保留原 sigmoid 或 softmax 分数与缩放，不引入新归一化 |
| `MoE` | `router_kwargs={"fix_router": True}` | 专家顺序 | 沿用 router 的权重规则，计数与辅助损失使用新编号 |
| `deepseekv3_ep_compute_fn` | `fix_router=True` | 目标 rank 优先 | 无偏 sigmoid 分数，原归一化与缩放 |
| `kimi_k26_ep_compute_fn` | `fix_router=True` | 目标 rank 优先 | 复用修正后的 DeepSeek-V3 固定适配器 |
| `DeepseekV41TopKRouter` | config 中 `moe_router_force_expert_balance=True` | 专家顺序 | 原 sigmoid、softmax 或 sqrtsoftplus，原归一化与缩放 |
| V4.1 crop builder | `fix_router=True` | 专家顺序 | 转换为上述 config 字段，text 与 vision crop 均适用 |

V4.1 的接入对象是 HyperParallel 自有 learned router；hash lookup 和其他模型的自有 router 没有自动覆盖。
固定模式绕过 group 选择结果，因此不代表真实 group-limited 通信行为。需要验证真实训练和通信分布时，保持所有固定路由开关关闭。
