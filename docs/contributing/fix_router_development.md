# HyperParallel fix-router 代码开发

开发基线为 `origin/master` 的 `1dd46230`，特性分支为 `feat/fix-router-20261012`。
本实现新增可复用的固定分配函数，接入通用 MoE 与 V4.1 learned router，并修正已有 DeepSeek-V3 和 Kimi 固定路由的编号与权重对应关系。
实现语义和适用边界见 [特性原理](../guide/fix_router.md)。

## 代码结构

| 文件 | 职责 |
| --- | --- |
| `hyper_parallel/components/functional/fix_router.py` | 生成 `[T,K]` int64 固定编号，校验形状、dtype、device 与专家布局，读取对应无偏权重 |
| `hyper_parallel/components/functional/__init__.py` | 延迟导出两个公共函数，不导入额外 NPU 依赖 |
| `hyper_parallel/components/modules/moe.py` | 增加默认关闭的 `fix_router`，在 Top-K 后替换编号与权重，再计算新编号的专家计数 |
| `hyper_parallel/distributed/expert_parallel/routing.py` | 固定 sigmoid-group 路由在归一化前重新取权重，保留现有固定适配器名称 |
| `hyper_parallel/models/kimi_k26/adapter/distributed/ep_compute.py` | 复用公共固定适配器，移除重复专家编号算法 |
| `hyper_parallel/models/deepseek_v41/modeling_deepseek_v41.py` | 将 MindFormers 同名 config 开关接入 learned router |
| `hyper_parallel/models/deepseek_v41/adapter/validation/cropped_model.py` | 将训练 builder 的 `fix_router` 传至 router config |
| `examples/training_demo/deepseek_v41/train_deepseek_v41*_online.yaml` | 明示默认关闭的 builder 配置 |

公共函数只负责分配和取分数，模型继续拥有评分函数、归一化、缩放以及辅助损失语义。
EP 参数分片、token dispatch、专家计算、shared expert 和 combine 沿用原有链路。
固定编号不新增参数或 buffer，不改变 checkpoint 中模型 tensor 的键与形状。

## 通用 MoE 使用

```python
from hyper_parallel.components.modules.moe import MoE, TokenChoiceTopKRouter

router = TokenChoiceTopKRouter(dim=64, num_experts=8, top_k=2, fix_router=True)
moe = MoE(dim=64, hidden_dim=128, num_experts=8, top_k=2,
          router_kwargs={"fix_router": True})
```

自定义 router 可在正常 Top-K 后、权重归一化前接入：

```python
from hyper_parallel.components.functional import fixed_topk_routing

indices, weights = fixed_topk_routing(unbiased_scores, original_indices)
# Continue the model's own normalization and scaling here.
```

该函数要求无偏分数为 `[T,E]` 浮点张量，原编号为同设备 `[T,K]` int32 或 int64 张量。
`local_expert_count` 默认一；指定每个 EP rank 的连续专家数可使用目标 rank 优先分配。
分数会保持原 dtype 与梯度连接；返回编号统一为 int64。

## EP 与 V4.1 配置

已有 DeepSeek-V3 或 Kimi recipe 在对应 `local_compute_fn` 下设置 `fix_router: true`。
使用通用 DeepSeek-V3 sigmoid-group archetype 的 GLM 模型同样生效。现有 `_target_` 和适配器注册名称无需变化。

V4.1 的 text 和 vision crop recipe 使用模型 builder 参数：

```yaml
model:
  _target_: hyper_parallel.models.deepseek_v41.adapter.validation.cropped_model.build_cropped_deepseek_v41
  config_path: /path/to/DeepSeek-V4.1-Flash
  engram_assets_path: /path/to/scaled-engram.json
  fix_router: true
```

直接构造 `DeepseekV41TopKRouter` 时，在 config 上设置 `moe_router_force_expert_balance=True`。
通用 router 和 V4.1 builder 拒绝字符串形式的 `"false"`，防止配置字符串误开固定路由。

## 验证范围

硬件无关单测覆盖 MindFormers 编号示例、EP 历史编号顺序、整除与余数负载、单 token、empty batch、完整 top-k、
每 token 专家唯一性、dtype/device/shape 错误、分数梯度、text/image 偏置、group mask、归一化与缩放。
完整 MoE 前后向与手工逐专家求和比较，同时覆盖输入加权、输出加权和激活重计算。
原有 Kimi factory 选择、shared expert overlap 与 capacity 参数回归随改动测试一起运行。

```powershell
python -m pip install -e . --no-deps
python -c "import hyper_parallel; print(hyper_parallel.__file__)"
python -m pytest -q tests/ut/components/functional/test_fix_router.py `
  tests/ut/components/modules/test_moe.py `
  tests/ut/distributed/expert_parallel/test_routing.py `
  tests/ut/auto_models/distributed/test_kimi_k26_ep_router.py `
  tests/ut/auto_models/models/deepseek_v41
python .agent/skills/autogit/scripts/autogit.py check
```

本地验证环境为 Windows、Python 3.12、PyTorch 2.7.1 CPU 和 Transformers 5.17.0。
特性相关测试最终通过 28 项测试、64 个子用例；包括 EP 核心和 DTensor 本地边界的相关回归通过 236 项测试、595 个子用例。
额外模型集成回归通过 58 项，另有一项 Windows 路径分隔符断言失败；该项在原工作区可复现，相关源码和测试与开发基线一致。
已有 pytest 平台标记在本地配置中未注册，会产生 warning，不影响测试结果。
多卡 Ascend/CUDA 的通信结果、编译支持、真实性能和显存收益需要在对应设备验证；本地 CPU 数值通过不代表这些指标已经验收。
建议多卡验收分别记录开关关闭与开启时的 gate/专家梯度、各 rank 收发槽位数、专家计数、step time 和峰值显存。
保持容量策略与其他并行配置一致，并分别覆盖 `T*K` 可整除和不可整除 EP 数量的输入。

## 兼容性与检查工具

开关关闭时保留原 Top-K 行为；既有 EP 固定模式保留原专家编号顺序，combine 权重改为对应新专家的真实权重。
因此旧固定模式的 loss 和 grad norm 不作为数值兼容目标。
现有静态 split 优化仍需满足它自身的输入和均匀分配条件，fix-router 开关不替代 split 校验。

仓库部分静态检查规则仍将直接 Torch import 当作旧多后端限制；本次沿用模型已有的逐 import 标注，
实现遵守当前 `AGENTS.md` 的 Torch-only 约束。检查通过情况和实际执行命令保存于归档目录的验证记录。
AutoGit 的提交入口只支持 GitCode，当前 `origin` 是 GitHub；提交使用原生 Git，保留单提交与远端分支推送规则。
