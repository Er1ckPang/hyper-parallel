# Xing4 Engram 训练

在 [Xing4 训练配方](train_single.yaml) 的 `model.config` 中设置 `engram_config_path`，读取架构 JSON 中的
八个 Engram 字段。也可以直接在 `model.config` 中填写这些字段。两处同时设置时，以配方中的字段为准。
附件格式支持 UTF-16。主干、注意力和 mHC 仍按 Xing4 训练配方配置。

对于 `config_arch_demo_1130.json`，使用零基主干层 1 和 11，主干深度至少为 12。复现附件形状时使用
40 层、`hidden_size: 3584`、`vocab_size: 131072` 和 4 路残差。现有训练主干采用 MLA 和可选融合 mHC。

| 字段 | 附件示例 | 用途 |
| --- | --- | --- |
| `engram_layer_ids` | `[1, 11]` | 启用查表的主干层，按列表顺序生成哈希桶 |
| `engram_max_ngram_size` | `3` | 从 bigram 开始计数，覆盖 2、3、4-gram |
| `engram_n_heads` | `8` | 每组 n-gram 的独立哈希头数 |
| `engram_head_dim` | `256` | 每行嵌入宽度，拼接后的投影输入宽度为 `24 × 256 = 6144` |
| `engram_vocab_size` | `16000000` | 每个哈希桶的素数搜索基数 |
| `engram_num_embeddings` | `[384006168, 384016682]` | 各层逻辑表行数，须等于该层桶大小之和 |
| `engram_pad_token_id` | `3` | n-gram 历史不足时使用的原始 token ID |
| `engram_compressed_vocab_size` | `131072` | 哈希输入的压缩词表大小 |

Engram 桶基数可独立于模型词表调整。生成新表时要同步更新逻辑行数。默认可用 `engram_table_pad_multiple`
配置物理表行数向上对齐，默认值为 16。EP 要求物理行数可被 EP 度数整除。对齐不改变哈希结果或 checkpoint
的逻辑行数。

从头训练时，压缩词表等于模型词表可使用恒等 token map。使用压缩 tokenizer 或已有权重时，通过
`model.config.engram_assets_path` 传入原始 JSON 资产，其中包含 token map、逐层倍数和素数桶。
资产中的 `max_ngram_size` 使用最大 n-gram 阶数，附件中的 `engram_max_ngram_size` 使用组数。
恢复训练时保留原始资产和层顺序。开启 Engram 的配方使用 PR1594 的 `GroupedExperts`，旧版 Xing4 Engram
checkpoint 的专家张量需要先转换布局。MindFormers 全量 checkpoint 条件见 [Xing4 训练指南](README.md)。

## 启动

先按 [Xing4 训练指南](README.md) 准备 CANN、隔离 Python 环境和预分词数据。复制训练配方到任务目录，设置
数据与输出路径。从头训练时留空恢复路径。以下命令在 HyperParallel 源码根目录运行。

```bash
# 单卡，YAML 使用缩小的模型和 Engram 表。
torchrun --standalone --nproc_per_node=1 -m examples.training_demo.train_text \
  /path/to/xing4-engram-smoke.yaml

# 单机双卡，EP2 对 Engram 表做行分片，普通参数按 DP2 分片。
torchrun --standalone --nproc_per_node=2 -m examples.training_demo.train_text \
  /path/to/xing4-engram-smoke.yaml \
  --accelerator.ep_size=2 --fsdp_config.dp_shard_size=2 --fsdp_config.edp_shard_size=1 \
  --training.global_batch_size=2
```

Engram 的行分片和并行计算由 `build_model` 声明。缩小配方可以保留示例的两层主干，使用以下参数启用第 1 层。

```yaml
model:
  config:
    engram_layer_ids: [1]
    engram_max_ngram_size: 3
    engram_n_heads: 2
    engram_head_dim: 16
    engram_vocab_size: 17
    engram_num_embeddings: [156]
    engram_pad_token_id: 3
    engram_compressed_vocab_size: 131072
```

附件的两张表合计约 366 GiB BF16 或 732 GiB FP32，另需预留主干参数、梯度和优化器状态。实际部署前根据
设备容量选择拓扑与表规模。自定义训练输入可传入与 CP 本地 token 对齐的 `engram_segment_starts` 和
`engram_token_mask`。前者重置哈希历史，后者选择参与查表的 token。注意力的 packed 边界由注意力入口管理。
