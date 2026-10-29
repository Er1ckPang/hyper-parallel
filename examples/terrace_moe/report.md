# TerraceMoE → HyperParallel：性能实验整合报告

报告日期：2026-10-29。覆盖2026-10-27至2026-10-29已完成的本机实验；所有正式实测均使用die 0–7。本报告整合已有数据，不新增训练，不修改已冻结的原始结果。

## 1. 主要结论

**MegaMoE是本机L-0实验中已经观测到的主要吞吐收益来源。** 最新四臂三轮复测中，global top-k启用Mega后吞吐提高 **56.45%**，T-Route提高 **63.05%**。

**T-Route在当前本机配置下没有体现稳定的独立吞吐收益。** 普通Hyper EP中，T-Route相对global的平均配对吞吐变化为 **-3.98%**；Mega中为 **+0.08%**，且三轮差异方向不一致。

**普通Hyper EP慢于原仓的主要原因是本地重排路径。** 在保持数值轨迹一致的诊断对照中，仅复用重排索引追回 **62.9%** 的耗时差距，融合chunk重排追回 **88.1%**。这些改动仅用于性能归因，尚未合入当前生产迁移实现。

**与原仓比较需要同时说明数值口径。** 10月27日同批次比较中，普通Hyper T-Route吞吐变化-27.87%，但100步loss/norm和最终参数与原仓完全一致；Mega T-Route吞吐提高16.43%，存在BF16加权顺序导致的训练轨迹差异。

![性能实验总览](figures/consolidated_performance_overview.png)

## 2. 模型、环境与测试方法

### 2.1 公共配置

| 项目 | 实际配置 |
|---|---|
| 网络 | 完整L-0：12层，MLA + QK-Norm + YaRN，hidden1024；首层dense、其余11层MoE |
| 专家 | E64；routed专家宽512；1个shared expert；top-4；dense FFN宽2816 |
| 路由 | global为无约束top-4；T-Route为16组选2、每组2专家 |
| 参数量 | 总参数1,496,284,160；EP8下每rank本地527,399,936 |
| 并行与batch | EP8、TP/PP/CP均为1；MBS1、GBS8、seq4096；不重算 |
| 训练 | 每次100个optimizer step；AdamW，LR 3e-4→3e-5，cosine100步；WD0.1，β=(0.9,0.95)，clip1.0 |
| 数据 | FineWeb-Edu真实子集，34,304文档、36,030,319 token；Qwen2.5 tokenizer；每次重放相同100步数据顺序 |
| 精度 | BF16参数/hidden，FP32 router计算和主参数/梯度，BF16派发门控 |
| 确定性 | seed1，torch/NPU/HCCL确定性，PYTHONHASHSEED=1，关闭MatMul K shift |
| 运行环境 | Ascend 910系列NPU，die0–7；Torch2.9.0+cpu + torch-npu2.9.0.post6；CANN9.1.0，驱动25.5.5 |
| 代码 | Hyper基线30c8ce78；迁移提交3c4fe6e8；源仓6fd2dc99，使用与指定2431d215一致的L-0配置/路由hook |

为对齐MLA、初始化、参数分片、辅助损失和分布式optimizer，实验共用原MindSpeed模型与训练器，由Hyper替换路由选择、EP派发/回收与专家计算。这里测量该完整L-0训练路径，不将专家微基准或单算子时间冒充整网吞吐。

数据取自原配方中占57%的FineWeb-Edu，不是完整六源混合语料。真实自由路由、bias更新和辅助loss均开启，不固定专家ID或跨step缓存动态splits。数据及native库完整指纹见附录。

### 2.2 计时与统计口径

每次保留100步完整记录，预先固定统计第21–100步。计时包含取batch、前向、反向、梯度同步、optimizer，以及所有实验共有的实际输入哈希钩子；取每步8个rank中的最大墙时。启动、参数全量指纹/探针、框架日志、参数范数日志和checkpoint不计入该训练核心step计时。

每步全机token数为 `GBS × seq = 8 × 4096 = 32768`。单次吞吐为 `32768 / mean(step_seconds)`；三次run的合并吞吐为累计测量token数除以累计训练墙时。路由比与Mega收益先在同轮计算，再对3轮比值取平均，所以它与直接相除四舍五入后的汇总吞吐可能略有差别。

**正式性能测量关闭profiler。** Profiling仅用于定位，单独采集rank0/rank7的第22–24步；采集时增加的墙时不用于吞吐比较。普通EP在21–100步仍有时延下降趋势，因此这里是同一训练轨迹的固定测量区间，不代表长时间稳态平台。

### 2.3 三批证据及用途

| 批次 | 成功实验规模 | 用途 |
|---|---|---|
| A：10月27日原仓/迁移对比 | 5臂各1次性能run，各100步；另有各1次精度run | 原仓与Hyper EP/Mega的同批次比较 |
| B：10月27日性能归因 | 2次profile、2次无profile基线、3次无profile消融，各100步 | 判断本地重排与通信调度的影响 |
| C：10月29日四臂三轮复测 | 4臂×3次独立100步，共1200步；合计960个测量step | global/T-Route和Mega收益的主要统计依据 |

各批分别统计。原仓比较使用A批；四臂三轮比较使用C批；B批诊断结果单列。不把不同日期或诊断路径的数据合并为同一组平均值。10月29日提交前的整理仅涉及文档、测试元数据和运行后护栏提取，已对照实验快照验证训练计算等价。

## 3. 主要结果：global/T-Route与MegaMoE（三次独立运行）

| 后端与路由 | 平均step ms | run均值的SD ms | 全机token/s |
|---|---:|---:|---:|
| Hyper EP / global | 421.214 | 11.131 | 77,794.2 |
| Hyper EP / T-Route | 438.646 | 2.783 | 74,702.6 |
| MegaMoE / global | 269.236 | 0.681 | 121,707.3 |
| MegaMoE / T-Route | 269.040 | 2.472 | 121,796.0 |

表中SD是3次独立run的平均step耗时的样本标准差，不是单次run中80个step的标准差。

| 配对比较 | 第1轮吞吐比 | 第2轮吞吐比 | 第3轮吞吐比 | 平均变化 |
|---|---:|---:|---:|---:|
| EP T-Route/global | 0.94599× | 0.95198× | 0.98258× | -3.982% |
| Mega T-Route/global | 1.00926× | 0.99775× | 0.99531× | +0.077% |
| global Mega/EP | 1.53459× | 1.54708× | 1.61177× | +56.448% |
| T-Route Mega/EP | 1.63721× | 1.62147× | 1.63267× | +63.045% |

**普通EP：** 三轮T/global均低于1，T-Route在这组测量中比global慢；幅度随run变化，不将平均值解释为所有训练阶段的固定开销。

**MegaMoE：** 三轮T/global分别有高于1和低于1的情况，平均差仅约0.08%，没有稳定的T-Route吞吐收益证据。不能把这个接近零的差异解释为显著加速或严格等价。

**Mega带来的改善：** 同轮global提升约56.45%、T-Route提升约63.05%，方向在三轮中一致。该结论针对本机L-0配置及当前数值口径。

本次三轮实际均使用die0–7，每轮四臂固定同一组设备；运行顺序交错并包含反向顺序。仅有3次独立运行，报告给出逐轮值和SD，不将240个step视作240次独立run来缩小置信区间。

![三轮第21–100步时延](figures/steady_step_times.png)

## 4. 原仓与迁移实现的对比（同批次单次性能run）

| 实验臂 | 平均step ms | 全机token/s | 对原仓T-Route吞吐比 |
|---|---:|---:|---:|
| Source T-Route | 313.688 | 104,460.6 | 1.00000× |
| Hyper EP / T-Route | 434.892 | 75,347.4 | 0.72130× |
| MegaMoE / T-Route | 269.430 | 121,619.6 | 1.16426× |
| Hyper EP / global | 420.776 | 77,875.1 | 0.74550× |
| MegaMoE / global | 269.728 | 121,485.2 | 1.16298× |

同为T-Route时，普通Hyper EP对原仓吞吐下降27.87%；Mega版提高16.43%。原仓是已有融合permutation、Ascend Triton排序和通信/计算重叠的MindSpeed生产路径，不能沿用早期简化研究EP微基准的性能排序。

前次普通EP的T/global变化为−3.25%，Mega为+0.11%；三轮复测分别为−3.98%和+0.08%。两批支持相同的方向判断：普通EP存在路由代价，Mega内差异接近运行波动。原仓没有在10月29日重复三轮，因此不将10月29日Hyper均值与10月27日原仓单次值组成新的同批次加速比。

## 5. 普通Hyper EP为何慢：profiling与消融

### 5.1 算子与同步证据

以下是rank0采集3步的平均值。设备列为同类型任务耗时之和，CPU等待列为主机时间；嵌套范围和跨流任务可能重叠，不能相加为完整step耗时。rank7也观察到相同调用次数和重排热点。

| 观测项，均为每step | 原仓 | Hyper EP |
|---|---:|---:|
| GroupedMatmul调用数 | 66 | 66 |
| GroupedMatmul设备累计ms | 10.204 | 9.734 |
| RepeatInterleave调用数 | 0 | 44 |
| RepeatInterleave设备累计ms | 0.000 | 118.694 |
| 额外AI_CPU TopK累计ms | 0.000 | 13.998 |
| alltoallvAicpuKernel累计ms | 26.184 | 26.432 |
| CPU stream synchronize次数 | 30 | 107 |

两边均为每层2个前向GMM、4个反向GMM，11层合计66次，专家矩阵乘法耗时接近。主要热点位于rank-major到expert-major的本地重排：Hyper分别为token和门控概率生成相同索引，每层生成2次、每次2个repeat_interleave，合计44次/step，并伴随动态shape/total读取和主机同步。原仓采用融合chunk排序，可联合搬运token和概率。

Hyper额外从dense routing map通过int32 topk重建专家ID，落为AI_CPU任务，约14ms/step，属于次要热点。源仓异步发起token/概率A2A并与共享专家或反向计算交错，Hyper的dispatch调度也有优化空间。

逻辑 `hcom_alltoallv_` 累计时间为源55.35ms、Hyper176.44ms，但底层AICPU任务约26.18/26.43ms。逻辑记录包含到达不齐及依赖等待，并与设备任务重叠；不能据此认定Hyper物理传输慢3.2倍。

### 5.2 关闭profiler后的完整训练消融

| 对照 | 平均step ms | 全机token/s |
|---|---:|---:|
| 原仓基线 | 313.582 | 104,495.9 |
| Hyper EP基线 | 435.579 | 75,228.6 |
| Hyper：同一次dispatch复用索引 | 358.884 | 91,305.2 |
| Hyper：融合chunk重排及必要的异步等待 | 328.072 | 99,880.6 |
| 原仓：A2A返回后立即wait | 321.872 | 101,804.4 |

原仓与Hyper基线相差 **121.998ms/step**。仅复用索引减少76.695ms，追回差距62.9%；融合重排减少107.508ms，追回差距88.1%。追回比例的分母是原仓与Hyper基线的耗时差距，不是总step耗时。两项独立对照的收益不能相加。

融合诊断的Hyper吞吐由75,229提高到99,881token/s，提升32.77%。这与“追回88.1%的差距”是不同指标。

融合对照同时替换映射生成、数据搬运和相应反向过程，因此88.1%归于这组本地重排路径，不是RepeatInterleave单个kernel的精确整步占比。它使用MindSpeed的Ascend Triton sorter，仅在诊断worker中生效；当前提交3c4fe6e8和三轮主实验均未包含该优化。

原仓对指定异步A2A返回后立即wait增加8.290ms，约2.64%。该对照只改变这部分A2A的等待位置，不代表取消全部异步重叠；支持本机优先解决本地重排的判断。融合对照仍比原仓慢14.490ms，这部分尚未通过独立消融完全分摊。

全部成功消融均完整训练100步，loss/norm和最终参数与其基线一致。首次融合诊断因异步通信张量进入Triton前缺少显式等待而失败，已保留日志并排除；补齐等待后的版本完整通过。

![代表性CPU/NPU时间线：仅用于定位](../20261027-terrace-hyper-l0/performance-analysis/figures/timeline_step23_rank0.png)

## 6. 性能结果的精度边界

普通Hyper T-Route与原仓的100步loss、梯度范数、参数范数最大差为0，8rank最终全部参数哈希相同。四臂三轮中的12个run，也分别与旧的同臂参考在100步指标、初始化/最终参数和实际输入哈希上严格一致。这证明每个实现的重复可复现，不表示global、T-Route或两个后端之间训练轨迹相同。

源仓和普通Hyper EP在down projection前乘BF16 gate，Mega在专家输出后加权。虽然bias-free线性层在实数中等价，BF16舍入顺序不同。Mega T-Route末步loss比源高0.011818、grad norm高3.56%；完整100步梯度范数相对差的峰值为160.33%，不能仅用末步差说明轨迹等价。Mega与保留的“BF16输出后加权Hyper oracle”100步轨迹相同，差异来源已有独立对照。

因此可以报告Mega当前实现的吞吐收益，但如果验收要求与源仓逐步数值等价，仍需让Mega支持down前加权并重新回归。100步训练损失也不构成长期质量或下游精度结论。

## 7. 结论适用范围与后续重点

| 已有证据支持 | 仍需补充验证 |
|---|---|
| 本机L-0中Mega显著提高吞吐；普通EP的重排开销很大 | 更大模型、不同batch/序列长度下的收益及长期稳态平台 |
| 普通EP三轮T-Route均慢于global；Mega内差异方向不一致 | 跨超节点、慢链路下的路由通信收益 |
| 同dispatch复用索引和融合重排能追回主要差距 | 把诊断优化原生纳入Hyper后的性能、空token/多形状/分布式回归 |
| 现有实现的100步重复可复现 | Mega与原仓逐步数值等价，以及长期训练质量 |

工程优先级是先复用同dispatch的重排映射，再替换通用逐行展开/搬运路径，并联合处理token和门控；随后处理dense mask到专家ID的往返及通信调度。应保持动态自由路由、确定性和正确的异步张量等待，不能用固定splits掩盖主机同步成本。

当前EP8、16逻辑组不满足源quota/T-A2A跨域快路径条件；T-A2A两跳协议也未原生融合进Hyper/Mega。本报告既不把跨超节点收益当作已测结果，也不据本机小差异否定更大规模的潜在收益。后续配置与验收使用[跨超节点任务说明](../20261027-terrace-hyper-l0/CROSS_SUPERNODE.md)。

## 8. 原始结果与复现入口

| 内容 | 文件位置 |
|---|---|
| 原仓及五臂完整报告 | [10月27日报告](../20261027-terrace-hyper-l0/RESULTS.md) |
| 原仓对比性能CSV | [performance_summary.csv](../20261027-terrace-hyper-l0/results/performance_summary.csv) |
| profiling与消融完整说明 | [性能归因报告](../20261027-terrace-hyper-l0/performance-analysis/PERFORMANCE.md) |
| 原始CPU/NPU trace和消融日志 | [performance-analysis/runs](../20261027-terrace-hyper-l0/performance-analysis/runs/) |
| 最新四臂三轮报告 | [10月29日报告](RESULTS.md) |
| 三轮汇总与配对比值 | [performance_summary.csv](results/performance_summary.csv)、[paired_comparisons.csv](results/paired_comparisons.csv) |
| 12次运行明细 | [per_run.csv](results/per_run.csv) |
| 全部1200步指标 | [per_step.csv](results/per_step.csv) |
| 12次精度/确定性守护 | [acceptance.json](results/acceptance.json) |
| 公共计划与顺序 | [plan.json](configs/plan.json) |
| 代码、native库和数据指纹 | [provenance.json](configs/provenance.json) |
| 本整合报告的可复算数据 | [integrated_performance_data.csv](results/integrated_performance_data.csv) |

每个 `runs/round*-*/` 内保存 `manifest.json`、`command.sh`、`train.log`、`metrics.jsonl`、`runtime-rank*.json`、参数和输入哈希。原始数据保留在原目录，本整合报告只读取并引用。

在工作区根目录执行以下命令可重新生成本报告、单文件HTML和总览图，不运行训练：

```bash
/home/ma-user/anaconda3/envs/terrace_hyper_l0/bin/python \
  _codex-docs/20261029-hyper-l0-router-performance/scripts/build_integrated_report.py
```

HTML将正文图表嵌入单文件，可离线阅读或通过浏览器打印为PDF；证据链接仍指向工作区内的原始文件。
