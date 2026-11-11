# Xing4 训练与 MF checkpoint 离线转换

模型复用 `hyper_parallel/models/xing_4/transformer/` 中的官方 Xing4 源文件，高性能模块替换和并行计划由模型 adapter 接入。配置依据为 [MF issue #12](https://gitcode.com/xuxinglei/mindformers-private/issues/12)，字段按语义映射，不要求 MF/HP 键名相同。

本目录只有 [train_single.yaml](train_single.yaml) 和本文档，MF→HP 转换只有 [tools/convert_mf_checkpoint.py](../../../tools/convert_mf_checkpoint.py) 一个入口。训练直接使用框架 `examples.training_demo.train_text`。两个 PR 尚未合入时，按下文在最新官方 master 上依次应用 PR #3、PR #2；依赖与已验证版本见[总 issue #1](https://github.com/xingleixu123/hyper-parallel-personal/issues/1)顶部表格。

## 源码准备：最新主仓 + 两份 PR

应用顺序为 **PR #3（公共 Trainer）→ PR #2（Xing4 模型与转换）**。抓取 PR 当前 head，并应用各自尚未包含在官方 master 中的完整提交范围；PR 追加修复后，只 cherry-pick 最新一个提交会漏掉此前的模型实现。不使用 squash 前的历史提交列表。需要私仓读取权限；以下命令在 Bash 中执行。

```bash
export HP_ROOT=/absolute/path/to/new-hyper-parallel
# 仅首次没有本地仓库时执行 clone；已有仓库直接进入其目录。
git clone https://github.com/mindspore-ai/hyper-parallel.git "$HP_ROOT"
cd "$HP_ROOT"
```

首次及以后更新都使用下面这一组命令，复用专用本地分支 `codex/xing4-local`。先停止使用此目录的训练并保存自己的修改；该分支只保留“官方 master + 两个 PR”，额外个人提交须另行保存。脚本会备份该分支旧 HEAD，然后重新组装，不更新任何远端分支。

```bash
(
set -euo pipefail
test -z "$(git status --porcelain)"
for operation in CHERRY_PICK_HEAD MERGE_HEAD REVERT_HEAD rebase-merge rebase-apply sequencer; do
    test ! -e "$(git rev-parse --git-path "$operation")"
done
git fetch https://github.com/mindspore-ai/hyper-parallel.git \
    +refs/heads/master:refs/remotes/xing-upstream/master
git fetch https://github.com/xingleixu123/hyper-parallel-personal.git \
    +refs/pull/3/head:refs/remotes/xing-pr/pr3 \
    +refs/pull/2/head:refs/remotes/xing-pr/pr2
UPSTREAM_SHA=$(git rev-parse refs/remotes/xing-upstream/master)
PR3_SHA=$(git rev-parse refs/remotes/xing-pr/pr3)
PR2_SHA=$(git rev-parse refs/remotes/xing-pr/pr2)

# 首条提交用于确认 PR 边界，后续追加提交会按顺序包含在范围内。
# 若 PR 经 squash 或改写，核对 Commits 页后更新对应首条 SHA。
PR3_FIRST=609a91c7c8ba6adc169fbc89751736d65a961ed3
PR2_FIRST=3bb9478237d8d668ff36c5899041e8546d010a01
git merge-base --is-ancestor "$PR3_FIRST" "$PR3_SHA"
git merge-base --is-ancestor "$PR2_FIRST" "$PR2_SHA"
git merge-base --is-ancestor "$PR3_FIRST^" "$UPSTREAM_SHA"
git merge-base --is-ancestor "$PR2_FIRST^" "$UPSTREAM_SHA"
mapfile -t PR3_COMMITS < <(git rev-list --reverse --topo-order "$UPSTREAM_SHA..$PR3_SHA")
mapfile -t PR2_COMMITS < <(git rev-list --reverse --topo-order "$UPSTREAM_SHA..$PR2_SHA")
printf 'pr3_commit=%s\n' "${PR3_COMMITS[@]}" | tee -a "$(git rev-parse --git-path xing4-assembly.log)"
printf 'pr2_commit=%s\n' "${PR2_COMMITS[@]}" | tee -a "$(git rev-parse --git-path xing4-assembly.log)"
ASSEMBLY_BRANCH=codex/xing4-local
if git show-ref --verify --quiet "refs/heads/$ASSEMBLY_BRANCH"; then
    OLD_HEAD=$(git rev-parse "refs/heads/$ASSEMBLY_BRANCH")
    BACKUP_REF="refs/xing4-backup/$(date -u +%Y%m%dT%H%M%SZ)-$OLD_HEAD"
    git update-ref "$BACKUP_REF" "$OLD_HEAD"
    printf 'backup=%s\n' "$BACKUP_REF"
fi
git switch -C "$ASSEMBLY_BRANCH" "$UPSTREAM_SHA"
if (( ${#PR3_COMMITS[@]} + ${#PR2_COMMITS[@]} )); then
    git cherry-pick -x "${PR3_COMMITS[@]}" "${PR2_COMMITS[@]}"
fi

git merge-base --is-ancestor "$UPSTREAM_SHA" HEAD
test "$(git rev-list --count "$UPSTREAM_SHA"..HEAD)" -eq "$((${#PR3_COMMITS[@]} + ${#PR2_COMMITS[@]}))"
git diff --stat "$UPSTREAM_SHA" HEAD
git diff --name-status "$UPSTREAM_SHA" HEAD
git status --short
printf 'upstream=%s pr3=%s pr2=%s assembled=%s tree=%s\n' \
    "$UPSTREAM_SHA" "$PR3_SHA" "$PR2_SHA" "$(git rev-parse HEAD)" "$(git rev-parse HEAD^{tree})" \
    | tee -a "$(git rev-parse --git-path xing4-assembly.log)"
)
```

每次 fetch 后分别固定官方 master 和两份 PR 的 SHA；“最新”指本次抓取的版本，不代表运行过程中持续更新。备份位于本地 `refs/xing4-backup/`，不会每次新增工作分支。fetch 的 `+` 只更新本地引用，以接收 PR squash 后的新 head；`git switch -C` 只重建上述专用本地分支，旧提交由备份保留。

cherry-pick 应用提交相对父提交的增量，不会把私仓旧 master 整棵树覆盖到官方主仓。PR #3 的网页 base 是历史依赖分支，所需 #1453 已在当前主仓内。仍须核对实际提交列表和 diff，确认只有两份 PR 的增量，范围外文件没有回退或删除；提交计数不能替代内容审查，也不能证明未来所有主仓版本都兼容。

执行前核对两份 PR 的 Commits 页。遇到空提交或冲突，脚本会停止；先核对是否已被主仓完整或部分吸收，以及接口是否变化，不机械使用 `--ours`、`--theirs` 或 `--skip`。解决冲突后用 `git cherry-pick --continue` 应用队列中剩余提交，再核对与主仓的 diff。撤销当前 cherry-pick 可用 `git cherry-pick --abort`。若某 PR 已完整合入，只应用剩余增量；两者均已合入时直接使用官方 master。

联合选择重计算按[总 issue](https://github.com/xingleixu123/hyper-parallel-personal/issues/1)应用主仓 #1538 的增量。组装成功后继续下文安装和训练，将 `.git/xing4-assembly.log` 中的实际 SHA、配置和环境版本一起归档。历史精度结果对应总 issue 中固定的实际运行版本；主仓或任一 PR 更新后，应针对变化复核，不能自动沿用旧结果宣称新版本已验收。

## HP 环境与安装

| 用途 | 源码 | 运行环境 |
| --- | --- | --- |
| MF 连续单卡基线 | [Somnus master](https://gitcode.com/Somnus2020/mindformers_formal_v1.0) `62ab670a`，MF 侧 HyperParallel [r1.0.0](https://github.com/mindspore-ai/hyper-parallel/tree/r1.0.0) `c95c7514` | Python 3.12.14，[MindSpore 2.10.0 build `6b7c4b02`](https://repo.mindspore.cn/mindspore/mindspore/version/202611/20261104/r2.10_20261104091850_6b7c4b0274392eb09ee3a5652ba14e3ed7a270b4_newest/unified/aarch64/mindspore-2.10.0-cp312-cp312-linux_aarch64.whl)，CANN 9.1.0 B243，Ascend 910B3 |
| HP Python 3.11 复验 | 官方 master `79f1597d` 加私仓 #3、#2 的两条实现提交与 [主仓 #1538](https://github.com/mindspore-ai/hyper-parallel/pull/1538)，运行 tree `0e4b1d99` | Python 3.11.18，PyTorch 2.13.0+cpu，Transformers 5.21.0，MindSpore 3.0.0b0 内置 TorchNPU 2.13.0，CANN 9.2.0 B237 / 9.2.1 B114 |
| HP 独立 PTA 2.13 长窗 | 同一运行 tree `0e4b1d99` | Python 3.12.14 / 3.12.16，PyTorch 2.13.0+cpu，独立 TorchNPU 2.13.0，Transformers 5.21.0，CANN 9.2.0 B237 / 9.2.1 |
| HP 独立 PTA 2.12 历史环境 | 对应[总 issue 的历史记录](https://github.com/xingleixu123/hyper-parallel-personal/issues/1) | Python 3.12.14，PyTorch 2.12.0+cpu，TorchNPU 2.12.0.post3，Transformers 5.21.0，CANN 9.2.0 B237，Ascend 910B3 |

MF 连续训练第 1–2500 步，在第 1500 步更新完成后保存模型、优化器、调度器和数据游标。HP 单卡从该 checkpoint 恢复，逐步对照 MF 的第 1501–2500 步。实际验收状态、固定提交和曲线见[总 issue](https://github.com/xingleixu123/hyper-parallel-personal/issues/1)。

### Python 3.11 与 MindSpore 内置 TorchNPU

在独立的 HP 虚拟环境中安装下面两份 Linux aarch64 wheel。MF 使用表中独立环境。

- [PyTorch 2.13.0+cpu，cp311](https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp311-cp311-manylinux_2_28_aarch64.whl)
- [MindSpore 3.0.0b0，cp311，build `6c261c45`](https://repo.mindspore.cn/mindspore/mindspore/version/202611/20261110/next_gen_20261110141311_6c261c451fc672bc260fa41dc82845a697e5bfb8/ascend/aarch64/mindspore-3.0.0b0-cp311-cp311-linux_aarch64.whl)

```bash
export HP_VENV=/absolute/path/to/hp-py311-venv
export PT_WHEEL=/absolute/path/to/torch-2.13.0+cpu-cp311-cp311-manylinux_2_28_aarch64.whl
export MS_WHEEL=/absolute/path/to/mindspore-3.0.0b0-cp311-cp311-linux_aarch64.whl
python3.11 -m venv "$HP_VENV"
source "$HP_VENV/bin/activate"
printf '%s  %s\n' \
  84453b69508ec79902f899c5ed9495acb9e2bbe9fda5f1d5d6f19e3c3842e1a7 "$PT_WHEEL" \
  526c6149b1b8da432f7e947fa6540f8019231565cefd95ba276b20dd905630fc "$MS_WHEEL" | sha256sum -c -
python -m pip uninstall -y torch-npu
python -m pip install "$PT_WHEEL" "$MS_WHEEL" 'numpy==1.26.4' 'pandas==2.3.3'
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
```

继续安装下文的 HP 和训练依赖。安装后先在虚拟环境内运行 `python -m pip check`，再激活 CANN 9.2 环境并检查实际后端来源。

```bash
python - <<'PY'
from importlib.metadata import version

import torch
import torch_npu

assert torch.__version__ == "2.13.0+cpu"
assert version("mindspore") == "3.0.0b0"
assert "/mindspore/_bundled/torch_npu/" in torch_npu.__file__
print(torch.__version__, version("mindspore"), torch_npu.__version__, torch_npu.__file__)
PY
```

版本来源可查 [CANN 下载](https://www.hiascend.com/cann/download)、[TorchNPU Release](https://gitcode.com/Ascend/pytorch/releases)、[MindSpore 构建目录](https://repo.mindspore.cn/mindspore/mindspore/version/)和[版本配套查询](https://www.hiascend.com/developer/download/compatibility)，按架构、Python ABI 和固定 build 选择。

先按[仓库安装指南](../../../docs/installation.md)准备相互匹配的 HP wheel、工具链和依赖。Python 3.11 复验锁定 NumPy 1.26.4。训练依赖包括 PyYAML 6.0.3、safetensors 0.8.0、pydantic 2.13.5、pybind11 3.1.0 和 setuptools 80.9.0；按固定版本准备，不做无约束升级。融合 mHC 还需要与 PTA/CANN/芯片匹配的 Omni Python 扩展及 custom OPP，以下先安装 HP，再构建 Omni。

```bash
export HP_ROOT=/absolute/path/to/hyper-parallel
export HP_VENV="${HP_VENV:-/absolute/path/to/hp-venv}"
export CANN_ENV=/absolute/path/to/cann/set_env.sh
export RUN_ROOT=/absolute/path/to/new-runs
export HP_CONFIG="$HP_ROOT/examples/training_demo/xing4/train_single.yaml"
export DATA_PREFIX=/absolute/path/to/mmap_deepseekv3_datasets_text_document
source "$HP_VENV/bin/activate"
source "$CANN_ENV"
cd "$HP_ROOT"
python -m pip install --use-pep517 --no-build-isolation --no-deps -e .
./build.sh --multicore off --strict on --jobs 8
python -c 'import torch, torch_npu, transformers, hyper_parallel; print(torch.__version__, torch_npu.__version__, transformers.__version__, hyper_parallel.__file__)'
mkdir -p "$RUN_ROOT"
```

`build.sh` 编译 indexed Dataset 的 native helper，需要匹配 Python ABI 的 pybind11 与 C++ 工具链。仅离线转换也使用同一套 HP 模型依赖：转换只在 CPU 构建 meta 状态并做文件 I/O，不占 NPU 卡，但配置解析会导入 PTA/融合算子的动态库。

更换 HP 版本时，在包含必要依赖的干净 checkout 中固定完整提交，重新安装 editable/wheel，并重建受 ABI 影响的 native 组件。用实际 import 路径、`git rev-parse HEAD`、`pip freeze` 记录生效身份。

## HP Omni 双阶段构建

在隔离源码副本固定 `45919b54b619ac5532ed2dc83fbca82ddd243540`；确认干净且工具链、磁盘和设备匹配。官方源码地址为 https://gitee.com/omniai/omni-ops ，固定提交已被 fetch，源码与先前构建快照的 2887 个源文件逐字节一致，6 项 OPP 已完成 CPU 重建。`OMNI_ROOT`、`CANN_HOME`、`RUN_ROOT` 指向实际路径；该构建示例是历史 B237 适配经验，必须验证实际环境。

```bash
export CANN_HOME=/absolute/path/to/active-cann-9.2
export OMNI_ROOT="$RUN_ROOT/omni-ops"
git clone https://gitee.com/omniai/omni-ops "$OMNI_ROOT"
git -C "$OMNI_ROOT" checkout --detach 45919b54b619ac5532ed2dc83fbca82ddd243540
test "$(git -C "$OMNI_ROOT" rev-parse HEAD)" = 45919b54b619ac5532ed2dc83fbca82ddd243540
cd "$OMNI_ROOT/training/ascendc"
ops='ai_infra_manifold_constrained_hyper_connection_pre;ai_infra_manifold_constrained_hyper_connection_pre_grad;manifold_constrained_hyper_connection_sinkhorn_enhance;ai_infra_sinkhorn_grad;ai_infra_manifold_constrained_hyper_connection_post;ai_infra_mhc_post_grad'
PATH="$CANN_HOME/tools/ccec_compiler/bin:$PATH" OPS_CPU_NUMBER=2 bash build.sh -c ascend910b -p "$CANN_HOME" -n "$ops"
# 仅使用刚生成的唯一安装包。
mapfile -t omni_installers < <(find "$OMNI_ROOT/training/ascendc" -type f -name 'CANN-omni_training_custom_ops*.run')
test "${#omni_installers[@]}" -eq 1
export OMNI_INSTALLER="${omni_installers[0]}"
sha256sum "$OMNI_INSTALLER"
bash "$OMNI_INSTALLER" --quiet --install-path="$RUN_ROOT/vendor-opp"
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
source "$RUN_ROOT/vendor-opp/vendors/omni_training_custom_transformer/bin/set_env.bash"
export CPLUS_INCLUDE_PATH="$RUN_ROOT/vendor-opp/vendors/omni_training_custom_transformer/op_api/include"
export LIBRARY_PATH="$RUN_ROOT/vendor-opp/vendors/omni_training_custom_transformer/op_api/lib"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
export CC="${CC:-gcc}" CXX="${CXX:-g++}"
cd "$OMNI_ROOT/training/ascendc/torch_ops_extension"
# 固定 Omni 版本在 PyTorch 2.13 中使用已移除的头文件。该 include 没有参与算子实现。
if python -c 'import torch; raise SystemExit(not torch.__version__.startswith("2.13."))'; then
  sed -i '/^#include <ATen\/NamedTensorUtils.h>$/d' omni_training_custom_ops/csrc_base/ops_common.h
fi
python -B setup.py build --parallel 1
export OMNI_ENV="$RUN_ROOT/vendor-opp/vendors/omni_training_custom_transformer/bin/set_env.bash"
PY_ABI=$(python -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
export OMNI_PYEXT="$OMNI_ROOT/training/ascendc/torch_ops_extension/build/lib.linux-aarch64-cpython-$PY_ABI"
test -d "$OMNI_PYEXT"
source "$OMNI_ENV"
export PYTHONPATH="$OMNI_PYEXT${PYTHONPATH:+:$PYTHONPATH}"
```

训练前核对实际加载路径、六项 schema 和 NPU 前后向。干净源码加 fresh Omni 的八卡 FSDP 从 1500 恢复至 1503 已 rc0，24 个输入匹配、payload/live 均通过，八 rank 的进程映射使用新 pyext（`e901...`）和新 libcust（`5469...`）。这是短测，不代表 fresh 产物已通过完整 1000 步。注册成功不等于设备 kernel 可用；安装脚本可能记录绝对路径，移动后须检查。

## 数据和训练配置

YAML 各项均有中文注释。共同模型为 1 Dense +1 MoE 主干、1 MTP、MLA、融合mHC，CMHC/clamp关闭；输入4096、训练终止2500、LR总日程7500/warmup1500。模型、optimizer、精度策略及其余训练参数保持实测值。

| 配置 | 单卡默认实测值 | 纯 FSDP8 实测值 |
| --- | --- | --- |
| GBS / LBS | 2 / 1 | 8 / 1 |
| DP / 梯度累积 | 1 / 2 | 8 / 1 |
| TP / CP / EP / PP | 1 / 1 / 1 / 1 | 1 / 1 / 1 / 1 |
| SP / loss parallel | false / false | false / false |
| `fsdp_config.dp_shard_size` | 1 | 8 |
| `fsdp_config.edp_shard_size` | 1 | 8 |
| `activation_checkpoint.mode` | full | off |
| 重计算选择 | 主干 `[0,1]`，不含MTP | 默认选择器，off时不使用 |

两种配置除表内策略和数据/输出/恢复路径外，已与归档的实际运行配置逐项核对。单卡GBS2与FSDP8的GBS8分别建立基线；8个LBS1的DP rank不能使用GBS2。

`model.config.enable_hyper_connections` 控制主干 mHC，`mtp_enable_hyper_connections` 控制 MTP。MTP 开关省略或设为 `null` 时跟随主干。与 MF 一样，开启 MTP mHC 需要先开启主干 mHC。

| 训练结构 | `enable_hyper_connections` | `mtp_enable_hyper_connections` |
| --- | --- | --- |
| 主干与 MTP 均使用 mHC（示例默认） | `true` | `null` 或 `true` |
| 主干使用 mHC，MTP 使用普通残差 | `true` | `false` |
| 主干与 MTP 均使用普通残差 | `false` | `null` 或 `false` |

所有组合均保留 `use_fused_mhc: true`。关闭 mHC 的层采用普通残差，最终模型不包含这些层的 mHC 参数。两处均关闭时，构建入口停用示例中的两条 mHC 专用 FP32 参数规则。融合、CMHC 和 clamp 等选项保留原值，只有存在 mHC 的层执行融合算子。

MF 对照时，两边分别设置相同的主干和 MTP 开关。凡有普通残差的层，MF 配方使用 `hidden_dropout: 0.0` 和 `apply_residual_connection_post_layernorm: false`。续训和离线转换都需要源、目标采用匹配的主干和 MTP 结构；MF 的主干开、MTP 关 checkpoint 对应表中第二行。

数据输入是已有的同前缀 `.bin`、`.idx`，不是转换脚本生成的新数据。设置 `DATA_PREFIX` 或 YAML `dataset.data_path` 更换数据；同时核对词表131072、EOD2、pad3、序列长度4096及实际样本。默认 `Xing4MFRecordDataset` 逐文档截断/补齐到4097后移位，不跨文档packing；普通预训练可按数据需要改用框架原生GPT Dataset，MF续训对照需保持实际样本语义一致。

更换模型权重时，HP原生DCP直接加载；MF native checkpoint先按下文转换。模型规模改变需使用匹配结构的 YAML 和权重，不能只把HF发布权重目录填入原生训练checkpoint字段。

## 观察与校验工具

先展开文末“生成观察与 CPU 校验脚本”并执行，脚本仅写入新的 `RUN_ROOT`，不改变模型、优化器或 checkpoint 算法。观察器使用同步 CPU 复制与摘要，影响性能；训练中必须全程开启才能取得未经 `.6g` 截断的指标。运行前确认 RUN_ROOT 为空闲新目录、足够存储，确保不同实验输出目录互不覆盖。历史源码重放的最终 HEAD 会因提交元数据变化而不同，下面 `HP_EXPERIMENT_HEAD` 记录真实值，不能硬编码旧 worktree 身份。

## MF 连续 2500 步基线

MF 与 HP 使用独立虚拟环境及 CANN；在新的 shell 中启动 MF，避免混用动态库。按权限从源仓检出冻结提交：

```bash
export MF_ROOT=/absolute/path/to/mf-clean HP_MS_ROOT=/absolute/path/to/hp-ms-clean
git clone https://gitcode.com/Somnus2020/mindformers_formal_v1.0.git "$MF_ROOT"
git -C "$MF_ROOT" checkout --detach 62ab670a6a86c3f42ef2e52284b289eba69238d4
git clone https://gitcode.com/mindspore/hyper-parallel.git "$HP_MS_ROOT"
git -C "$HP_MS_ROOT" checkout --detach c95c7514053abdd9460b6b55cf4b4021ffbc246d
# 激活已有的独立 Python 3.12 / MindSpore 固定环境，然后执行下文版本校验。
python -m pip check
```

按冻结源码安装剩余依赖时避免自动升级或替换 MindSpore；CANN 设备 kernel 和源码权限需事先准备。下面先验证实际版本，不能仅凭 wheel 文件名判定。

目标Linux aarch64/910B、Python3.12.14、MF62ab、HP-MS c95、MindSpore **2.10.0 build 6b7c4b02**、CANN **9.1.0 B243**。这是指定实验栈，配套状态另依官方查询；CANN9.2是另一套HP运行环境，不能混入MF。优先复用完全匹配的环境；如需新环境，必须从官方构建目录取得精确 `6b7c4b02` 的 aarch64/cp312 wheel 并独立核验，上表给出已核验的官方 wheel 直链。不得无约束升级 MindSpore。MF/HP-MS其余依赖按冻结源和环境单独安装并锁版本，不用可能替换MS的无约束依赖安装；`pip check`与实际import验证。目标环境需C++/Python开发头、`ninja`、`make`、`pybind11`，检查 `command -v ninja` 后再编译；不要假设python虚拟环境中自带ninja。

```bash
# 均为用户实际路径；MF_ROOT/HP_MS_ROOT为独立干净checkout，DATA_PREFIX为现成数据。
export MF_ROOT=/path/to/mf-clean HP_MS_ROOT=/path/to/hp-ms-clean
export RUN_ROOT=/path/to/new-run CANN_HOME=/path/to/cann-9.1.0
export DATA_PREFIX=/path/to/mmap_deepseekv3_datasets_text_document
export HYPER_PARALLEL_PLATFORM=mindspore
source "$CANN_HOME/set_env.sh"
command -v python; command -v ninja; command -v c++
test "$(git -C "$MF_ROOT" rev-parse HEAD)" = 62ab670a6a86c3f42ef2e52284b289eba69238d4
test "$(git -C "$HP_MS_ROOT" rev-parse HEAD)" = c95c7514053abdd9460b6b55cf4b4021ffbc246d
test -z "$(git -C "$MF_ROOT" status --porcelain)"
test -z "$(git -C "$HP_MS_ROOT" status --porcelain)"
python - <<'PY'
from pathlib import Path
import mindspore as ms
assert ms.__version__ == '2.10.0'
assert '6b7c4b02' in (Path(ms.__file__).parent / '.commit_id').read_text()
print('MS:', ms.__version__, ms.__file__)
PY
# 实际检查CANN安装info为 version=9.1.0 / V100R001C11SPC001B243。
cat "$CANN_HOME/aarch64-linux/ascend_toolkit_install.info"
```

MF native indexed helper按MF源码的 `mindformers/dataset/blended_datasets/utils.py:compile_helpers` 使用 `make -C "$MF_ROOT/mindformers/dataset/blended_datasets"`；该构建会在源码树产生未跟踪扩展，先记录clean commit，构建后记录产物hash/状态，或在独立可写源码副本编译。确认`python3`/`python3-config`与激活的3.12 ABI一致且`python3 -m pybind11 --includes`可用；不要混用另一个环境的helper。

### MF 21 源 host 扩展构建

以下源自历史CustomOpBuilder成功链，使用当前clean HP-MS源码的基础21文件；环境可选增强indexer的额外依赖未覆盖。`BUILD_DIR`必须是任务独立新目录，不动活跃source `build/lib` 或现有.so；保存命令stdout/stderr、21文件SHA、生成ninja与so hash。不要把最终hash要求固定成历史689d（构建路径/工具链可变）。

```bash
export BUILD_DIR="$RUN_ROOT/mf-native-extension"
mkdir -p "$RUN_ROOT"
test ! -e "$BUILD_DIR"
mkdir "$BUILD_DIR"
export MAX_JOBS=4 ASCEND_RT_VISIBLE_DEVICES=""
python - <<'PY'
import ast, hashlib, os
from pathlib import Path
import mindspore as ms
src = Path(os.environ['HP_MS_ROOT']) / 'hyper_parallel/platform/mindspore/custom_ops'
tree = ast.parse((src / 'custom_op_impl.py').read_text())
item = next(x for x in tree.body if isinstance(x, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == '_CUSTOM_OP_SOURCES' for t in x.targets))
files = [src / x.args[1].value for x in item.value.elts]
assert len(files) == 21 and len(set(files)) == 21 and all(p.is_file() for p in files)
for p in files:
    print('source', p.name, hashlib.sha256(p.read_bytes()).hexdigest())
m = ms.ops.CustomOpBuilder('hyper_parallel_custom_ops_ms', [str(p) for p in files],
                           backend='Ascend', build_dir=os.environ['BUILD_DIR']).load()
p = Path(m.__file__).resolve()
assert p.is_relative_to(Path(os.environ['BUILD_DIR']).resolve())
for n in ('npu_situ_glu', 'npu_mhc_pre_sinkhorn', 'npu_mhc_pre_sinkhorn_backward',
          'npu_mhc_post', 'npu_mhc_post_backward'):
    assert callable(getattr(m, n, None)), n
print('extension', p, hashlib.sha256(p.read_bytes()).hexdigest())
PY
```

构建仅验证CPU导入。正式运行前设置 `PYTHONPATH="$BUILD_DIR:$MF_ROOT:$HP_MS_ROOT${PYTHONPATH:+:$PYTHONPATH}"`，普通导入 `hyper_parallel.platform.mindspore.custom_ops.custom_op_impl` 并打印 `impl._custom_ops.__file__`、SHA及五个exports；其源码`build/lib`优先，若存在旧so可能覆盖新目录，先检查并明确加载身份。**禁止预先向sys.modules注入so掩盖实际路径**。loader缺`npu_situ_glu`可能自动构建，`HP_CUSTOM_OPS_NO_AUTO_REBUILD=1`不是通用禁止构建；如load失败必须排查。本轮正常loader已记录新b04c扩展路径及SHA并完成三步设备训练；新设备、新环境仍需验证实际加载和前后向。MS wrapper与HP PTA Omni pyext是两条依赖链；目标CANN mHC设备kernel/OPP可用性需环境核验，文档不可暗示源码host so独立提供kernel。


### 写入 MF 运行配置

以下通过安全替换环境变量，不执行 shell 展开 YAML；先 `export RUN_ROOT DATA_PREFIX`。

```bash
mkdir -p "$RUN_ROOT/mf"
cat > "$RUN_ROOT/mf-template.yaml" <<'YAML'
checkpoint:
  enable_save: true
  save_path: ${RUN_ROOT}/mf/checkpoints
  save_max: 1000
  save_interleaved_steps: 1500
  no_save_optim: false
  async_save: false
  prefix: mf-single
  remove_redundancy: true
  load_path: ''
  load_balanced: false
  no_load_optim: false
  load_worker_number: 8
training:
  steps: 2500
  local_batch_size: 1
  global_batch_size: 2
  max_norm: 1.0
  seed: 1234
  deterministic: false
optimizer:
  type: Muon
  adamw_betas:
  - 0.9
  - 0.95
  adamw_eps: 1.0e-08
  weight_decay: 0.1
  matched_adamw_rms: 0.2
  momentum: 0.95
  nesterov: true
  ns_steps: 5
  qk_clip_enabled: true
  qk_clip_threshold: 100.0
  comm_strategy: allgather_deredundency
  use_fused_adamw: true
  adamw_include:
  - '*word_embeddings*'
  - '*output_layer*'
  phase0_batch_memory_gb: 1.0
  ns_batch_memory_gb: 1.0
  phase4_batch_memory_gb: 4.0
lr_scheduler:
  type: CosineWithWarmUpLR
  learning_rate: 4.0e-05
  lr_end: 2.0e-05
  warmup_lr_init: 0.0
  warmup_steps: 1500
  total_steps: 7500
callbacks:
- type: MFLossMonitor
  per_print_times: 1
parallelism:
  data_parallel_shard: 1
  data_parallel_shard_strategy: optim_grads_params
  reshard_after_forward_policy: always
  cpu_offload: false
  tensor_parallel: 1
  sequence_parallel: false
  enable_mc2: false
  enable_loss_parallel: false
  expert_parallel: 1
  npu_nums_per_device: 1
  moe_token_dispatcher_type: alltoall
  expert_parallel_async_d2h: false
  expert_parallel_use_safe_tokens: true
  context_parallel: 1
  context_parallel_method: colossal
  context_parallel_async: false
  pipeline_parallel: 1
recompute:
  mode: full
  full_recompute_layer:
  - 0-39
  exclude_op:
    mlp\.router:
    - 0-39
recompute_comm:
  enable: false
train_dataset:
  balance_enabled: true
  dataloader:
    type: OrderedIndexDataLoader
    data_path: ${DATA_PREFIX}
    shuffle: false
    sequence_length: 4096
    reset_position_ids: false
    eod_mask_loss: false
    create_attention_mask: false
    reset_attention_mask: true
    create_compressed_eod_mask: true
    compressed_eod_mask_length: 384
    eod_token: 2
    pad_token: 3
    create_seq_len_vector: false
    column_names:
    - input_ids
    - labels
    - loss_mask
    - position_ids
    - actual_seq_len
    python_multiprocessing: false
    use_cache: false
    cache_dir: null
  drop_remainder: true
  num_parallel_workers: 2
  prefetch_size: 1
  use_distribute_dataset: true
model:
  chunk_loss_num: 8
  calculate_per_token_loss: false
  model_type: deepseek_v3
  architectures: DeepseekV3ForCausalLM
  vocab_size: 131072
  seq_length: 4096
  hidden_size: 3584
  intermediate_size: 9216
  num_hidden_layers: 2
  max_position_embeddings: 32768
  hidden_act: fusedswiglu
  num_attention_heads: 32
  rms_norm_eps: 1.0e-06
  add_bias_linear: false
  attention_bias: false
  use_flash_attention: true
  use_eod_attn_mask_compression: true
  multi_latent_attention: true
  mla_qkv_concat: true
  kv_lora_rank: 512
  q_lora_rank: 768
  qk_rope_head_dim: 64
  v_head_dim: 128
  qk_nope_head_dim: 128
  qk_layernorm: true
  attention_dropout: 0.0
  hidden_dropout: 0.0
  params_dtype: bfloat16
  embedding_params_dtype: bfloat16
  layernorm_params_dtype: bfloat16
  moe_router_params_dtype: float32
  compute_dtype: bfloat16
  layernorm_compute_dtype: float32
  softmax_compute_dtype: float32
  rotary_dtype: float32
  router_dense_type: float32
  fp32_residual_connection: true
  initializer_range: 0.02
  num_nextn_predict_layers: 1
  mtp_loss_scaling_factor: 0.1
  position_embedding_type: yarn
  rotary_scaling_factor: 16.0
  beta_fast: 32
  beta_slow: 1
  mscale: 1
  mscale_all_dim: 1
  rope_theta: 100000.0
  apply_rope_fusion: true
  bias_swiglu_fusion: false
  enable_hyper_connections: true
  mhc_compute_dtype: bfloat16
  enable_hc_head: false
  num_residual_streams: 4
  mhc_sinkhorn_iterations: 20
  mhc_init_gating_factor: 0.01
  enable_fused_triton_sinkhorn: false
  use_fused_mhc: true
  mhc_megatron_compatibility: true
  hc_sinkhorn_clamp: null
  gated_linear_unit: true
  moe_intermediate_size: 1024
  routed_scaling_factor: 2.0
  first_k_dense_replace: 1
  n_routed_experts: 64
  num_experts_per_tok: 4
  n_shared_experts: 1
  moe_shared_expert_intermediate_size: 1024
  norm_topk_prob: true
  scoring_func: sigmoid
  moe_grouped_gemm: true
  moe_permute_fusion: true
  topk_group: 0
  n_group: 0
  moe_z_loss_coeff: 0.0
  use_pad_tokens: true
  moe_router_force_expert_balance: false
  moe_router_load_balancing_type: seq_aux_loss
  moe_aux_loss_coeff: 0.0005
  moe_token_drop_policy: false
  moe_router_enable_expert_bias: true
  moe_router_bias_update_rate: 0.0001
  enable_cmhc: false
  experimental_attention_variant: null
context:
  mode: 1
  device_target: Ascend
profiler:
  enable_profiling: false
  start_step: 2
  end_step: 2
  output_path: ${RUN_ROOT}/mf/profiler
  profiler_rank:
  - 0
  - 256
  - 512
  - 768
  - 784
  - 800
  - 816
  - 832
  - 848
  - 864
  - 880
  profiler_level: 1
  profile_memory: false
  profile_cpu: false
  with_stack: false
YAML
python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['RUN_ROOT'])
s = (p / 'mf-template.yaml').read_text()
for key in ('RUN_ROOT', 'DATA_PREFIX'):
    s = s.replace('${' + key + '}', os.environ[key])
(p / 'mf.yaml').write_text(s)
PY
```

### MF 原始指标和样本观察

同步采样会影响时序；观察器不改变训练数据。它依赖当前 MF 版本的 `_next_batch`，样本哈希与 HP 对照仍需独立验证。

```bash
cat > "$RUN_ROOT/observe_mf.py" <<'PY'
"""Observe native MF training through its callback and unchanged batch fetch."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import mindspore as ms
import numpy as np

from mindformers.pynative.callback.callback import TrainerCallback
from mindformers.pynative.tools.monitor_utils import get_monitor_data_tracker
from mindformers.pynative.trainer.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--device-id', type=int, default=0)
    args = parser.parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='Ascend', device_id=args.device_id)
    trainer = Trainer(args.config)
    # MF has no public callback for each fetched microbatch.
    original_next = trainer._next_batch
    pending = []

    def observe_next():
        batch = original_next()
        observed = {}
        for key in ('input_ids', 'labels', 'loss_mask', 'position_ids', 'actual_seq_len'):
            array = batch[key].asnumpy()
            item = {'shape': list(array.shape), 'dtype': str(array.dtype),
                    'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}
            logical = array.astype(np.int64)
            if array.ndim == 2:
                item['canonical_chunks_256'] = [
                    [hashlib.sha256(row[start:start + 256].tobytes()).hexdigest()
                     for start in range(0, row.size, 256)] for row in logical]
            observed[key] = item
        pending.append(observed)
        return batch

    class Record(TrainerCallback):
        def __init__(self, stream):
            self.stream = stream
            self.last_step = 0
            self.last_samples = 0

        def on_train_begin(self, config, state, **kwargs):
            del config, kwargs
            self.last_step = int(state.global_step)
            self.last_samples = int(state.consumed_samples)
            Path(args.ledger + '.initial.json').write_text(json.dumps({
                'global_step': self.last_step, 'consumed_samples': self.last_samples,
                'gradient_accumulation': trainer.num_accumulation_steps,
                'global_batch_size': trainer.global_batch_size}, indent=2))

        def on_step_end(self, config, state, **kwargs):
            del config, kwargs
            tracker = get_monitor_data_tracker()
            values = tracker.data
            step = int(state.global_step)
            metrics = [values.get('loss'), values.get('mtp-1-loss'),
                       values.get('load-balancing-loss')]
            if (step != self.last_step + 1 or len(pending) != trainer.num_accumulation_steps or
                    int(state.consumed_samples) != self.last_samples + trainer.global_batch_size or
                    tracker.training_state.get('step') != step or
                    any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics)):
                raise RuntimeError('Missing or inconsistent native step observation')
            row = {'step': step, 'consumed_samples': int(state.consumed_samples),
                   'loss': metrics[0], 'mtp_raw': metrics[1], 'aux_raw': metrics[2],
                   'microbatches': list(pending)}
            self.stream.write(json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n')
            self.stream.flush()
            pending.clear()
            self.last_step = step
            self.last_samples = int(state.consumed_samples)

    with Path(args.ledger).open('x', encoding='utf-8') as stream:
        observer = Record(stream)
        trainer.callback_handler.add_callback(observer)
        trainer._next_batch = observe_next
        try:
            trainer.train()
            if pending or observer.last_step != int(trainer.state.max_steps):
                raise RuntimeError('Native train returned without complete observations')
        finally:
            trainer._next_batch = original_next
            trainer.callback_handler.remove_callback(observer)


if __name__ == '__main__':
    main()
PY
export PYTHONPATH="$BUILD_DIR:$MF_ROOT:$HP_MS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$MF_ROOT"
set -o pipefail
python "$RUN_ROOT/observe_mf.py" --config "$RUN_ROOT/mf.yaml" --ledger "$RUN_ROOT/mf/observations.jsonl" --device-id 0 2>&1 | tee "$RUN_ROOT/mf/native.log"
```

## MF checkpoint 转换

关闭 MF shell，在新的 HP shell 重新激活 HP 环境并 source 表中对应的 CANN 9.2 环境，开启融合 mHC 时加载同 ABI 的 Omni 扩展，设置 `HP_ROOT`、`HP_CONFIG`、`RUN_ROOT`、`DATA_PREFIX` 和 `OMNI_PYEXT`，确认 `hyper_parallel.__file__` 指向组装后的 checkout。不要将 MF 的 `PYTHONPATH` 或 CANN 9.1 动态库带入转换和 HP 训练进程。

MF 的 checkpoint callback 会记录并捕获部分保存错误；训练结束不等于保存成功。先确认 step 1500 原生保存成功且该 checkpoint 无写入者，再运行下方 closure 审计；基线不必结束到 2500，只要状态闭合、下一批样本对应，就可在另一空闲卡组并行续训与基线后续步骤。源必须是已完整写完的某个 MF iteration 目录，含模型、FP32主权重、optimizer槽、`common.json`；多rank分片还需原生 `metadata.json` 及其引用的全部文件。无需训练观察器、额外实验manifest或源MF YAML，也无需手填源/目标TP、EP等并行度。

| 参数 | 必填/默认 | 含义 |
| --- | --- | --- |
| `--src` / `--load-dir` | 必填 | 完成保存的MF iteration目录，不是多个iteration的父目录。 |
| `--dst` / `--save-dir` | 必填 | 新的HP DCP输出目录，不能覆盖已有目录。 |
| `--hp-config` | 必填 | 目标HP YAML，用于复用模型、模块替换、optimizer参数组、日程和DataLoader构建。 |
| `--mf-optimizer` | 完整续训必填 | `muon`、`adamw` 或 `fused-adamw`，解释源更新计数约定。 |
| `--source-metadata` | 可选 | 源metadata另存路径，默认`src/metadata.json`。 |
| `--data-path` | 可选 | 转换机器的数据前缀覆盖，默认取HP YAML的`dataset.data_path`。 |
| `--resume-state` | 可选 | 同一保存点的HP格式额外状态，单份`.pt`或含原生逐rank文件的目录。 |
| `--weights-only` | 默认关闭 | 只转换模型张量，不构建optimizer、scheduler及数据状态。 |
| `--shard-size-mb` | 默认512 | 分批输出预算；单个大张量及变换缓冲可以超过该预算。 |

MF Muon即使启用融合AdamW fallback也填`muon`；只有独立融合AdamW才填`fused-adamw`。训练步数取`common.json.global_step`，数据进度取`consumed_samples`，optimizer更新次数取其保存值及源约定，不一律假定样本数等于step×GBS。目标HP YAML定义对应的实际GBS/LBS、逻辑矩阵分组、精度和LR日程，不比较两份配置的键名或整份内容相等。

```bash
set -euo pipefail
export MF_CKPT="$RUN_ROOT/mf/checkpoints/iteration_00001500"
test -f "$MF_CKPT/common.json"
mkdir -p "$RUN_ROOT/proofs"
python "$RUN_ROOT/scripts/mf_checkpoint_closure.py" --src "$MF_CKPT" --ledger "$RUN_ROOT/mf/observations.jsonl" --log "$RUN_ROOT/mf/native.log" --output "$RUN_ROOT/proofs/mf1500-closure-v2.json"
export HP_CKPT="$RUN_ROOT/mf1500-hp"
python -m tools.convert_mf_checkpoint \
  --src "$MF_CKPT" --dst "$HP_CKPT" --hp-config "$HP_CONFIG" \
  --mf-optimizer muon --data-path "$DATA_PREFIX"
```

输出是完整原生HP DCP目录：共享`.metadata`、safetensors/bytes、嵌入式公共训练状态及来源收据`conversion.json`。MF的分片序号依据`axis_fragmentations`转为逐轴元素坐标，再由框架reader/planner按区域交集重组；源已使用元素坐标时保留。目标运行时由原生DCP重分片，同一产物不需要按每种目标并行度重复转换。

完整恢复原生 DataLoader 要求源与目标 `global_batch_size` 不变；不能关闭数据恢复来绕过验收。实测 MF GBS2 → FSDP8 GBS8 被 elastic DP 正确拒绝，即使八 rank 的模型与槽 after_load 已通过，也不是完整恢复。可选保持 GBS2 的合法 TP/CP 拓扑；两卡 TP2 从 MF1500 恢复到1503 已短测 rc0，不代表完整1000步通过。自动数据恢复适用于可从全局游标恢复的原生定长loader；跨DP使用`single`或`cyclic + data_sharding=false`。数据源内部快照、流式缓冲和兼容的Torch RNG需提供真实逐rank `--resume-state`，不承诺任意迁移其私有拓扑。MF RNG字节不能当作Torch RNG，默认保留HP配置的运行时种子；当前dropout/router jitter为0，完整对照仍需核实实际下一批样本。

### 只加载模型权重

使用新的输出目录，增加`--weights-only`，不需要源optimizer计数或数据位置入参：

```bash
python -m tools.convert_mf_checkpoint \
  --src "$MF_CKPT" --dst "$RUN_ROOT/mf-weights" \
  --hp-config "$HP_CONFIG" --weights-only
```

原生训练加载时将`restore_from`设为该目录，并追加以下三项；这是从权重开始新训练状态，不是完整续训：

```bash
# 追加到相应的原生训练命令：
--checkpoint.restore_optimizer=false \
--checkpoint.restore_train_state=false \
--checkpoint.restore_dataloader_state=false
```

## HP 单卡从零和续训

```bash
cat > "$RUN_ROOT/run1.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case_name=${1:?baseline|mf_resume|self_resume}
END_STEP=${END_STEP:-2500}
: "${RUN_ROOT:?}" "${HP_ROOT:?}" "${DATA_PREFIX:?}" "${HP_CONFIG:?}"
case "$case_name" in
 baseline) restore=null; save=true ;;
 mf_resume) : "${HP_CKPT:?}"; restore=$HP_CKPT; save=false ;;
 self_resume) : "${SINGLE_CKPT:?}"; restore=$SINGLE_CKPT; save=false ;;
 *) exit 2 ;;
esac
save=${SAVE_CKPT:-$save}
if [[ $case_name == baseline ]]; then case_name=single_baseline; fi
out="$RUN_ROOT/runs/${case_name}${RUN_SUFFIX:-}"; test ! -e "$out"; mkdir -p "$out"
export HP_EXPERIMENT_HEAD="$(git -C "$HP_ROOT" rev-parse HEAD)" HP_OBSERVATIONS="$out/observations"
cd "$HP_ROOT"
python -m torch.distributed.run --standalone --nproc_per_node=1 "$RUN_ROOT/scripts/observe_train.py" "$HP_CONFIG" \
  "--dataset.data_path=$DATA_PREFIX" "--training.train_iters=$END_STEP" "--checkpoint.restore_from=$restore" "--checkpoint.save_ckpt=$save" \
  "--checkpoint.checkpoint_dir=$out/checkpoints" 2>&1 | tee "$out/native.log"
SH
```

按需单独选择流程。先设置 `export ASCEND_RT_VISIBLE_DEVICES=0`；从零短跑可用 `SAVE_CKPT=false END_STEP=3 RUN_SUFFIX=-fresh3 bash "$RUN_ROOT/run1.sh" baseline`，不会保存终止 checkpoint。从零连续训练：

```bash
bash "$RUN_ROOT/run1.sh" baseline
```

MF 转换后直接从 MF 1500 续训，不依赖单卡从零训练：

```bash
export HP_MF_REFERENCE_LEDGER="$RUN_ROOT/mf/observations.jsonl"
bash "$RUN_ROOT/run1.sh" mf_resume
unset HP_MF_REFERENCE_LEDGER
```

从已有 HP 单卡 1500 自恢复：

```bash
export SINGLE_CKPT="$RUN_ROOT/runs/single_baseline/checkpoints/global_step_1500"
bash "$RUN_ROOT/run1.sh" self_resume
```

`HP_MF_REFERENCE_LEDGER` 强制比较 1501 步每微批输入、shift labels、位置、监督 mask；对齐通过仍不代替全窗口比较。

## HP 八卡连续及跨配置续训

最终 YAML 的 CP wrapper `ulysses_degree: null` 按 CP 大小自动选择全 Ulysses；TP>1 且 CP>1 时须使用全 Ulysses 并满足注意力头数整除等要求。`ulysses_degree: 1` 的 KV AllGather/hybrid 仅 TP=1 时可选，结果须与自动策略分开记录。MoE gate 和 shared experts 保持 EP 复制，而 `experts.gate_up_proj` 与 `experts.down_proj` 保持 TP 复制、CP 复制、EP `shard(0)`，EP=1 时不发生实际切分。这些是并行计划，不改变训练超参数或模型数学实现。具体配置的完整验收结果和版本身份见[总 issue #1](https://github.com/xingleixu123/hyper-parallel-personal/issues/1)。

| case | TP / CP / EP | DP shard / expert DP shard | 重计算 |
| --- | --- | --- | --- |
| baseline、fsdp8 | 1 / 1 / 1 | 8 / 8 | off |
| hsdp4 | 1 / 1 / 1 | 4 / 1 | off |
| tp2 | 2 / 1 / 1 | 4 / 1 | off |
| ep2 | 1 / 1 / 2 | 8 / 4 | off |
| cp2 | 1 / 2 / 1 | 8 / 1 | off |
| sac、recompute_moe、recompute_all | 1 / 1 / 1 | 8 / 1 | selective、full [1]、full [0,1,2] |
| tp2_ep2 | 2 / 1 / 2 | 4 / 4 | off |
| tp2_cp2_ep2、tp2_cp2_ep2_all | 2 / 2 / 2 | 4 / 4 | off、full [0,1,2] |

以下启动器应用对应 recipe，以新 YAML 内的 TP expert 复制规则运行；每个候选从同一个 FSDP8 的 1500 完整状态独立加载，GBS=8、LBS=1 和 LR 7500/warmup1500 不变。`BASE_CKPT` 为连续基线实际生成的 `global_step_1500` 目录；先确认保存完成且源不再写入。不同机器需自行设置空闲设备、HCCL 和端口；1500 保存已闭合且下一批样本对应后，可在互不重叠的空闲卡组与基线余步并行。以下一组八卡的顺序循环只是资源限制。`train_iters=2500` 是绝对步数。

```bash
cat > "$RUN_ROOT/run8.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case_name=${1:?baseline|fsdp8|hsdp4|tp2|ep2|cp2|sac|recompute_moe|recompute_all|tp2_ep2|tp2_cp2_ep2|tp2_cp2_ep2_all}
END_STEP=${END_STEP:-2500}
: "${RUN_ROOT:?}" "${HP_ROOT:?}" "${DATA_PREFIX:?}" "${HP_CONFIG:?}"
case "$case_name" in
  baseline|fsdp8) tp=1; cp=1; ep=1; ds=8; es=8; selection=off ;;
  hsdp4) tp=1; cp=1; ep=1; ds=4; es=1; selection=off ;;
  tp2) tp=2; cp=1; ep=1; ds=4; es=1; selection=off ;;
  ep2) tp=1; cp=1; ep=2; ds=8; es=4; selection=off ;;
  cp2) tp=1; cp=2; ep=1; ds=8; es=1; selection=off ;;
  sac) tp=1; cp=1; ep=1; ds=8; es=1; selection=selective ;;
  recompute_moe) tp=1; cp=1; ep=1; ds=8; es=1; selection='[1]' ;;
  recompute_all) tp=1; cp=1; ep=1; ds=8; es=1; selection='[0,1,2]' ;;
  tp2_ep2) tp=2; cp=1; ep=2; ds=4; es=4; selection=off ;;
  tp2_cp2_ep2) tp=2; cp=2; ep=2; ds=4; es=4; selection=off ;;
  tp2_cp2_ep2_all) tp=2; cp=2; ep=2; ds=4; es=4; selection='[0,1,2]' ;;
  *) echo "Unknown case: $case_name" >&2; exit 2 ;;
esac
if [[ $case_name == baseline ]]; then restore=null; save=true
else : "${BASE_CKPT:?set BASE_CKPT to completed FSDP8 step 1500 directory}"; restore=$BASE_CKPT; save=false; fi
save=${SAVE_CKPT:-$save}
out="$RUN_ROOT/runs/${case_name}${RUN_SUFFIX:-}"
test ! -e "$out"
mkdir -p "$out"
export HP_EXPERIMENT_HEAD="$(git -C "$HP_ROOT" rev-parse HEAD)"
export HP_OBSERVATIONS="$out/observations"
if ((tp > 1)); then sp=true; else sp=false; fi
if [[ $selection == off ]]; then ac=("--activation_checkpoint.mode='off'" '--activation_checkpoint.selection={}')
elif [[ $selection == selective ]]; then ac=(--activation_checkpoint.mode=selective '--activation_checkpoint.selection={}')
else ac=(--activation_checkpoint.mode=full --activation_checkpoint.selection.source=model_adapter_safe_regions "--activation_checkpoint.selection.layer_indices=$selection"); fi
cd "$HP_ROOT"
python -m torch.distributed.run --standalone --nproc_per_node=8 "$RUN_ROOT/scripts/observe_train.py" "$HP_CONFIG" \
  "--dataset.data_path=$DATA_PREFIX" --training.global_batch_size=8 "--training.train_iters=$END_STEP" \
  "--checkpoint.restore_from=$restore" "--checkpoint.save_ckpt=$save" "--checkpoint.checkpoint_dir=$out/checkpoints" \
  "--accelerator.tp_size=$tp" "--accelerator.cp_size=$cp" "--accelerator.ep_size=$ep" --accelerator.pp_size=1 \
  "--accelerator.sequence_parallel=$sp" "--accelerator.loss_parallel=$sp" \
  "--fsdp_config.dp_shard_size=$ds" "--fsdp_config.edp_shard_size=$es" "${ac[@]}" 2>&1 | tee "$out/native.log"
SH
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset HP_MF_REFERENCE_LEDGER
bash "$RUN_ROOT/run8.sh" baseline
export BASE_CKPT="$RUN_ROOT/runs/baseline/checkpoints/global_step_1500"
test -f "$BASE_CKPT/.metadata"
```

从零的八卡短测可使用 `SAVE_CKPT=false END_STEP=3 RUN_SUFFIX=-fresh3 bash "$RUN_ROOT/run8.sh" baseline`，不会保存终止 checkpoint。短程恢复检查可先运行以下命令：输出写入 `fsdp8-smoke`，从相同 1500 状态执行到绝对步 1503；检查无误后再运行完整 case。短程输出不能用于 1000 步统计。`END_STEP` 只改变终止步，LR 仍为 7500/warmup1500。单卡已设置 `HP_CKPT` 后可同样使用 `RUN_SUFFIX=-smoke END_STEP=1503 bash "$RUN_ROOT/run1.sh" mf_resume`。

```bash
RUN_SUFFIX=-smoke END_STEP=1503 bash "$RUN_ROOT/run8.sh" fsdp8
```

短跑无误后按需单独运行各配置，或顺序执行完整窗口：

```bash
for case_name in fsdp8 hsdp4 tp2 ep2 cp2 sac recompute_moe recompute_all tp2_ep2 tp2_cp2_ep2 tp2_cp2_ep2_all; do
  bash "$RUN_ROOT/run8.sh" "$case_name"
done
```

实际 checkpoint 名称以输出目录为准。确认 1500 checkpoint 已写入完成且独立核验完整性后即可开始候选；八卡基线与 MF 单卡 GBS2 不直接逐步比较。TP/CP 的 DP rank 和梯度累积按 mesh 自动计算；EP 不额外作为 DP 除数。`hsdp4` 为 DP shard 4 × replica 2。先在目标硬件短跑验证全部算子和内存，再执行完整窗口；短跑用独立后缀并仅修改绝对终止步，不改变调度器。

## 恢复和完整窗口核验

训练完成后读取实际观察目录和完整源 checkpoint（只处理可信 checkpoint）。payload 和 live 是两个独立检查；后者 `PASS_WITH_UNCHECKED` 不是全通过。Muon 非 owner 清理须有实际 assignment、replicate 进程组、owner 分片及哈希证据；另一个同状态补采若必须使用，可指定 `--ownership-observations`，脚本会逐 rank 核对原快照。payload 校验默认严格。仅 MF 转换后的目标 TP 非 leader，若实际 `tp_rank>0`、live loader 为 None 且加载的 extra loader 为空，可通过 `--allow-nonleader-no-loader` 按原生目标 skeleton 行为核验；原生逐 rank 状态仍严格精确比较。比较器失败或缺文件时停止验收。

以下包装器为每个 case 选择分析布局。TP 非 leader 的 loader 在运行完成后由实际 `inputs_rank*.jsonl` 中的 `tp_rank` 识别；`--allow-nonleader-no-loader` 只有确实观察到 `tp_rank>0` 才允许无 loader，无需推测 global rank。TP rank 缺失时不得假定通过。`expected-parameters=61` 是此模型固定结构的默认值，修改模型后须独立确认并显式调整。

```bash
set -euo pipefail
cat > "$RUN_ROOT/verify8.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
case_name=${1:?case name}
: "${RUN_ROOT:?}" "${BASE_CKPT:?}"
case "$case_name" in
  fsdp8|hsdp4|ep2|sac|recompute_moe|recompute_all) layout=fsdp8; live_extra=() ;;
  tp2|tp2_ep2) layout=tp2; live_extra=(--allow-nonleader-no-loader) ;;
  cp2) layout=cp2; live_extra=() ;;
  tp2_cp2_ep2|tp2_cp2_ep2_all) layout=tp2_cp2; live_extra=(--allow-nonleader-no-loader) ;;
  *) echo "Unknown case: $case_name" >&2; exit 2 ;;
esac
obs="$RUN_ROOT/runs/$case_name/observations"
mkdir -p "$RUN_ROOT/proofs"
python "$RUN_ROOT/scripts/compare_hp_restored_payload.py" "$BASE_CKPT" "$obs" --ranks 8 --output "$RUN_ROOT/proofs/$case_name-payload.json"
python "$RUN_ROOT/scripts/compare_hp_live_restore.py" "$obs" --ranks 8 --output "$RUN_ROOT/proofs/$case_name-live.json" "${live_extra[@]}"
python "$RUN_ROOT/scripts/analyze_hp_comparison.py" "$RUN_ROOT/runs/baseline/observations" "$obs" --layout "$layout" --output-dir "$RUN_ROOT/results/$case_name"
SH
for case_name in fsdp8 hsdp4 tp2 ep2 cp2 sac recompute_moe recompute_all tp2_ep2 tp2_cp2_ep2 tp2_cp2_ep2_all; do
  bash "$RUN_ROOT/verify8.sh" "$case_name"
done
```

`--layout` 的 `fsdp8` 用于相同 DP8 输入布局（FSDP8、EP2、HSDP、重计算）；TP2 用 `tp2`，CP2 用 `cp2`，TP2+CP2 用 `tp2_cp2`。检查各 case 的实际观察 shape/坐标与分析假设一致；EP 与 HSDP 的独立 rank 映射也必须核实，不能把 `--layout` 成功推断为所有拓扑的正确性。CP 规范化块将 IGNORE_INDEX 映射为 0，并非原始 labels 字节等价。所有候选实际输入、resume 和三指标必须分别通过；不要将未经验证的联合配置当作已通过。

指标是 main foundation loss、weighted MTP 除以 0.1 得到 raw MTP、raw router AUX；HP 比较器输出 `comparison.csv`，其 baseline/candidate 是每步未经 `.6g` 格式化的观察值、delta 是两者的浮点差值，累计列是到该步的 signed mean。HP 比较器的 `statistics.json` 在完整 1501–2500 共 1000 步上用 `math.fsum` 计算 candidate−baseline 的 signed mean、其绝对值、MAE、最大绝对差及正负零计数，验收为 `abs(signed mean) <= 0.005`，不平滑，不以 MAE 替代。下方生成器将 CSV 原始值转换为可识别关键词的日志，保留源 SHA 和列映射，再在 [log.javazero.top](https://log.javazero.top/) 上传日志作图。

MF 单卡对 HP 从 MF 1500 恢复还需要对源 checkpoint、映射、payload 和 live 分别生成证据：

```bash
set -euo pipefail
python "$RUN_ROOT/scripts/audit_mf_to_hp_independent.py" --source "$MF_CKPT" --target "$HP_CKPT" --report "$RUN_ROOT/proofs/mf1500-independent-mapping-v1.json"
python "$RUN_ROOT/scripts/compare_hp_restored_payload.py" "$HP_CKPT" "$RUN_ROOT/runs/mf_resume/observations" --ranks 1 --output "$RUN_ROOT/proofs/hp-single-payload-proof-v1.json"
python "$RUN_ROOT/scripts/compare_hp_live_restore.py" "$RUN_ROOT/runs/mf_resume/observations" --ranks 1 --source-rng-not-saved --output "$RUN_ROOT/proofs/hp-single-live-proof-v2.json"
python "$RUN_ROOT/scripts/analyze_mf_hp_single.py" --mf-run "$RUN_ROOT/mf" --hp-run "$RUN_ROOT/runs/mf_resume" --data-prefix "$DATA_PREFIX" --proof-root "$RUN_ROOT/proofs" --output "$RUN_ROOT/results/mf-hp-single"
```

独立映射审计针对单 rank 源与逻辑目标，仅审计模型/主权重/槽，不审计 extra state；源完成保存及没有写入进程也需单独核查。MF 比较器输出 `per-step.csv` 和 `summary.json`；`per-step.csv` 来源是连续 MF 观察与 HP 恢复观察，两者按 step 配对，含原值及差值；同时报告 LR 打印精度限制、样本检查和不可检查字段。MF 统计脚本只读取四份 proof 并记录状态，不将其状态合入自身 PASS 判定；必须逐一审阅 proof 的结果，不能仅因统计通过或文件存在判定通过。MF 与 HP 的 RNG 格式不同，MF RNG 不转换；仍需验证样本、精度和随机性假设。

## 从原始 CSV 生成绘图日志

下面生成器只读取完整的 1501–2500 窗口，校验无重复、缺失或非有限值，并按原始数值输出，不平滑。HP 使用 `comparison.csv` 的 `*_baseline/candidate`；MF 对 HP 使用 `per-step.csv` 的 `*_mf/hp`。输出 `overlay` 的两个日志分别为参考和候选，`signed_delta` 为 candidate−reference 与零线，`cumulative` 为每步前缀的 `math.fsum(delta)/n`、零线及 ±0.005。逐步或累计曲线越界只是诊断，±0.005 只约束**最终完整窗口**的有符号均值；零线标示有符号差值 0；分母为 0 的相对误差不采用。`manifest.json` 记录输入 CSV 的 SHA256、标签与字段映射。

```bash
cat > "$RUN_ROOT/prepare_plot_input.py" <<'PY'
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('reference_label')
parser.add_argument('candidate_label')
args = parser.parse_args()
for label in (args.reference_label, args.candidate_label):
    if not label or label in ('.', '..') or '/' in label or '\\' in label or Path(label).name != label:
        raise ValueError('labels must be single file names')
if args.reference_label == args.candidate_label:
    raise ValueError('reference and candidate labels must differ')
raw = args.csv_path.read_bytes()
with args.csv_path.open(encoding='utf-8-sig', newline='') as stream:
    reader = csv.DictReader(stream)
    fields = set(reader.fieldnames or ())
    hp = {'LOSS': ('loss_baseline', 'loss_candidate'),
          'MTPLOSS': ('mtp_raw_baseline', 'mtp_raw_candidate'),
          'AUXLOSS': ('aux_baseline', 'aux_candidate')}
    mf = {'LOSS': ('loss_mf', 'loss_hp'),
          'MTPLOSS': ('mtp_raw_mf', 'mtp_raw_hp'),
          'AUXLOSS': ('aux_raw_mf', 'aux_raw_hp')}
    mapping = next((m for m in (hp, mf) if all(x in fields for pair in m.values() for x in pair)), None)
    if mapping is None or 'step' not in fields:
        raise ValueError('CSV has neither supported metric mapping')
    rows = {}
    for row in reader:
        step = int(row['step'])
        if step in rows or not 1501 <= step <= 2500:
            raise ValueError(f'duplicate or unexpected step: {step}')
        values = {key: tuple(float(row[column]) for column in pair) for key, pair in mapping.items()}
        if any(not math.isfinite(v) for pair in values.values() for v in pair):
            raise ValueError(f'nonfinite value at step {step}')
        rows[step] = values
if set(rows) != set(range(1501, 2501)):
    raise ValueError('expected exactly one row for each step 1501 through 2500')
if args.output.exists():
    raise FileExistsError(f'choose a new output directory: {args.output}')
args.output.mkdir(parents=True)
for group in ('overlay', 'signed_delta', 'cumulative'):
    (args.output / group).mkdir()
delta_label = args.candidate_label + '-minus-' + args.reference_label
file_mapping = {
    'overlay': {'reference': args.reference_label, 'candidate': args.candidate_label},
    'signed_delta': {'delta': delta_label, 'zero': 'zero'},
    'cumulative': {'mean': 'Mean_' + delta_label, 'zero': 'zero',
                   'positive_limit': 'positive_limit', 'negative_limit': 'negative_limit'},
}
streams = {}
for group, names in file_mapping.items():
    for role, filename in names.items():
        streams[group, role] = (args.output / group / (filename + '.log')).open('x', encoding='utf-8')

def emit(group, name, step, values):
    metrics = ' '.join(f'{key}: {value:.17g}' for key, value in values.items())
    streams[group, name].write(f'step: {step} {metrics}\n')

try:
    diffs = {key: [] for key in mapping}
    for step in sorted(rows):
        values = rows[step]
        reference = {key: pair[0] for key, pair in values.items()}
        candidate = {key: pair[1] for key, pair in values.items()}
        delta = {key: candidate[key] - reference[key] for key in mapping}
        for key in mapping:
            diffs[key].append(delta[key])
        mean = {key: math.fsum(diffs[key]) / len(diffs[key]) for key in mapping}
        zero = dict.fromkeys(mapping, 0.0)
        emit('overlay', 'reference', step, reference)
        emit('overlay', 'candidate', step, candidate)
        emit('signed_delta', 'delta', step, delta)
        emit('signed_delta', 'zero', step, zero)
        emit('cumulative', 'mean', step, mean)
        emit('cumulative', 'zero', step, zero)
        emit('cumulative', 'positive_limit', step, dict.fromkeys(mapping, 0.005))
        emit('cumulative', 'negative_limit', step, dict.fromkeys(mapping, -0.005))
finally:
    for stream in streams.values():
        stream.close()
manifest = {'source_csv': str(args.csv_path), 'sha256': hashlib.sha256(raw).hexdigest(),
            'window': [1501, 2500], 'rows': len(rows), 'reference_label': args.reference_label,
            'candidate_label': args.candidate_label, 'field_mapping_reference_candidate': mapping,
            'delta': 'candidate minus reference', 'smoothing': False, 'file_mapping': file_mapping}
(args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY
cat > "$RUN_ROOT/run_plot_input.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
: "${RUN_ROOT:?}"
python "$RUN_ROOT/prepare_plot_input.py" "$@"
SH
# 根据自己的结果选择一个，输出目录必须不存在。
bash "$RUN_ROOT/run_plot_input.sh" "$RUN_ROOT/results/cp2/comparison.csv" "$RUN_ROOT/plots/cp2" FSDP8 CP2
bash "$RUN_ROOT/run_plot_input.sh" "$RUN_ROOT/results/mf-hp-single/per-step.csv" "$RUN_ROOT/plots/mf-hp" MF HP
```

在网站将 X 最小值设为 `1501`、最大值设为 `2500`，差值模式选有符号的 `normal`，不启用平滑。分别上传一个目录内的 `.log` 文件；每组用 `step` 作为横轴，分别选择 `LOSS`、`MTPLOSS`、`AUXLOSS` 关键词作图，日志文件直接使用标签命名，字段与文件映射记录在 manifest。overlay、signed_delta、cumulative 分开画；保留日志和 manifest 以便追溯，实际网站界面和解析结果需核对。

## 运行与验证边界

MF clean 62ab/c95 源码重建的 host 扩展完成三步普通 loader 训练，历史完整 2500 步基线使用另一构建产物；两者身份不可混同。fresh Omni 八卡恢复短测的边界如上，完整窗口结果见总 issue。MS 2.10.0 build `6b7c4b02` 的精确 wheel、CANN、设备算子、私有源码权限和数据是前置条件；请按官方渠道核对，不执行升级 MS 的无约束安装。

## 附录：观察与校验脚本

<details><summary>生成观察与 CPU 校验脚本</summary>

```bash
mkdir -p "$RUN_ROOT/scripts"
cat > "$RUN_ROOT/scripts/observe_train.py" <<'PY'
"""Observe the native Trainer without replacing training or checkpoint methods."""

import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch

from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.core.distributed_checkpoint.planner import WriteItemType
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardSavePlanner
from hyper_parallel.core.optimizer.muon import Muon
from hyper_parallel.tools.model_integration.optimizer_layout import (
    optimizer_leaves, optimizer_state_for_parameter, snapshot_training_layout,
)
from hyper_parallel.tools.model_integration.integration_validator import collect_activation_checkpoint_wrappers
from hyper_parallel.trainer.callbacks.base import Callback
from hyper_parallel.trainer.config.parser import parse_training_args
from hyper_parallel.trainer.text_trainer import TextTrainer


def digest(value):
    return hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def local_digest(value):
    local = value.to_local() if callable(getattr(value, 'to_local', None)) else value
    return digest(local.detach().cpu())


def json_value(value):
    if torch.is_tensor(value):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': local_digest(value)}
    if isinstance(value, dict):
        return {str(key): json_value(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(child) for child in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Observation(Callback):
    """Write actual ordered forward inputs and finalized, unrounded step metrics."""

    def __init__(self, trainer, directory):
        super().__init__(trainer)
        self.rank = trainer.global_rank
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.inputs = (self.directory / f'inputs_rank{self.rank}.jsonl').open('x')
        self.metrics = (self.directory / f'metrics_rank{self.rank}.jsonl').open('x')
        self.micro = 0
        self.step = None
        self.handle = trainer.model.register_forward_pre_hook(self.forward_inputs, with_kwargs=True)
        if trainer.config.checkpoint.restore_from is not None:
            integration = trainer.model_integration
            original_capture = integration.capture_checkpoint_payload

            def capture(event, path, payload):
                original_capture(event, path, payload)
                if event != 'after_load':
                    return
                planner = StandardSavePlanner(enable_plan_caching=False, remove_redundancy=False)
                planner.configure_planner({name: payload[name] for name in ('model', 'optimizer')},
                                          rank=self.rank, use_collectives=False)
                tensors, values = [], {}
                for item in planner.build_local_plan().items:
                    value = planner.get_data(item)
                    if item.type == WriteItemType.TENSOR:
                        chunk = item.tensor_data['chunk']
                        tensors.append({'key': item.index.fqn, 'offsets': list(chunk.offsets),
                                        'sizes': list(chunk.sizes), 'global_shape': list(item.tensor_data['size']),
                                        'dtype': str(value.dtype), 'sha256': digest(value)})
                    else:
                        values[item.index.fqn] = json_value(value)
                (self.directory / f'loaded_payload_rank{self.rank}.json').write_text(json.dumps({
                    'checkpoint': path, 'tensors': tensors, 'values': values,
                    'extra_state': json_value(payload.get('extra_state'))}, indent=2))

            # Reuse the existing checkpoint observation hook; loading is unchanged.
            integration.capture_checkpoint_payload = capture

    def on_train_begin(self, state, **kwargs):
        parallel = self.trainer.get_batch.parallel_context
        value = {'global_step': state.global_step, 'epoch': state.epoch,
                 'num_micro_batches': self.trainer.num_micro_batches,
                 'source_head': os.environ['HP_EXPERIMENT_HEAD'],
                 'parallel_coordinates': {key: getattr(parallel, key, None)
                                          for key in ('dp_rank', 'cp_rank', 'tp_rank')},
                 'activation_checkpoint': collect_activation_checkpoint_wrappers(self.trainer.model)}
        value['checkpoint_wrapper_mro'] = [
            {'fqn': name, 'type': type(module).__qualname__,
             'mro': [base.__qualname__ for base in type(module).__mro__]}
            for name, module in self.trainer.model.named_modules(remove_duplicate=False)
            if any(base.__name__ == 'CheckpointWrapper' for base in type(module).__mro__)]
        (self.directory / f'restore_rank{self.rank}.json').write_text(json.dumps(value, indent=2))
        names = {id(getattr(p, 'main_param', p)): name
                 for name, p in self.trainer.model.named_parameters(remove_duplicate=False)}
        assignments = []
        for leaf in optimizer_leaves(self.trainer.optimizer):
            if type(leaf) is not Muon:
                continue
            for group_id, info in leaf._hsdp_assignment_batches.items():
                for batch_id, batch in enumerate(info['batch_groups']):
                    for sub_id, assignment in enumerate(batch['sub_batches']):
                        assignments.append({'group': group_id, 'batch': batch_id, 'sub_batch': sub_id,
                            'is_replicated': assignment.is_replicated,
                            'replicate_group_ranks': list(assignment.replicate_group_ranks),
                            'replicate_sizes': list(assignment.replicate_sizes),
                            'replicate_pgs': [torch.distributed.get_process_group_ranks(pg)
                                              for pg in assignment.replicate_pgs],
                            'shard_pgs': [torch.distributed.get_process_group_ranks(pg)
                                          for pg in assignment.shard_pgs],
                            'records': [{'name': names[id(record.param)], 'index': record.index,
                                'owner': assignment.owner_by_index.get(record.index),
                                'is_owned': assignment.is_owned(record)}
                                for record in assignment.all_records]})
        (self.directory / f'muon_assignments_rank{self.rank}.json').write_text(
            json.dumps(assignments, indent=2))
        if self.trainer.config.checkpoint.restore_from is not None:
            trainer = self.trainer
            snapshot = snapshot_training_layout(trainer.model, trainer.optimizer)
            for name, parameter in trainer.model.named_parameters(remove_duplicate=False):
                main = getattr(parameter, 'main_param', parameter)
                entry = snapshot['parameters'][name]
                entry['model_sha256'] = local_digest(parameter)
                entry['main_sha256'] = local_digest(main)
                entry['main_is_model'] = main is parameter
                entry['optimizer_state_hashes'] = {
                    key: json_value(child) for key, child in
                    optimizer_state_for_parameter(trainer.optimizer, main).items()}
            snapshot['scheduler'] = json_value(trainer.lr_scheduler.state_dict())
            loader = trainer.train_dataloader
            snapshot['dataloader'] = None if loader is None else json_value(loader.state_dict())
            snapshot['rng'] = {'cpu': local_digest(torch.get_rng_state()),
                               'device': local_digest(torch.npu.get_rng_state()),
                               'python': json_value(random.getstate())}
            (self.directory / f'live_restore_rank{self.rank}.json').write_text(json.dumps(snapshot, indent=2))
            gate = os.environ.get('HP_RESTORE_GATE')
            if gate:
                deadline = time.monotonic() + 900
                while not Path(gate).is_file():
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Independent restore verification did not finish within 15 minutes')
                    time.sleep(0.2)
                if json.loads(Path(gate).read_text()).get('status') != 'PASS':
                    raise RuntimeError('Independent restore verification rejected this state')

    def on_step_begin(self, state, **kwargs):
        self.step, self.micro = state.global_step + 1, 0

    def forward_inputs(self, module, args, kwargs):
        del module
        if args:
            raise RuntimeError('Observation expects the native keyword forward contract')
        context = self.trainer.get_batch.parallel_context
        row = {'step': self.step, 'micro': self.micro, 'rank': self.rank,
               'dp_rank': getattr(context, 'dp_rank', None),
               'cp_rank': getattr(context, 'cp_rank', None),
               'tp_rank': getattr(context, 'tp_rank', None), 'fields': {}}
        for name in ('input_ids', 'labels', 'shift_labels', 'loss_mask', 'position_ids'):
            value = kwargs.get(name)
            if value is None:
                continue
            cpu = value.detach().cpu().contiguous()
            logical = cpu.to(torch.int64)
            if name in ('labels', 'shift_labels'):
                logical = logical.masked_fill(logical == IGNORE_INDEX, 0)
            pieces = []
            for sample in logical.reshape(-1, logical.shape[-1]):
                pieces.append([digest(piece) for piece in sample.split(256)])
            row['fields'][name] = {'shape': list(cpu.shape), 'dtype': str(cpu.dtype),
                                   'sha256': digest(cpu), 'canonical_chunks_256': pieces}
            if name == 'position_ids':
                row['fields'][name]['starts'] = logical.reshape(-1, logical.shape[-1])[:, 0].tolist()
        if 'input_ids' not in row['fields'] or 'shift_labels' not in row['fields']:
            raise RuntimeError('The actual forward is missing training inputs or shifted targets')
        reference = os.environ.get('HP_MF_REFERENCE_LEDGER')
        if reference and self.step == 1501:
            with Path(reference).open() as stream:
                expected = next(json.loads(line) for line in stream if json.loads(line)['step'] == 1501)
            micro = expected['microbatches'][self.micro]
            for hp_name, mf_name in (('input_ids', 'input_ids'), ('shift_labels', 'labels'),
                                      ('position_ids', 'position_ids')):
                if row['fields'][hp_name]['canonical_chunks_256'] != micro[mf_name]['canonical_chunks_256']:
                    raise RuntimeError(f'Actual MF/HP step1501 micro{self.micro} differs in {hp_name}')
            labels = kwargs['shift_labels'].detach().cpu().contiguous()
            mask = (labels != IGNORE_INDEX).to(torch.int64)
            chunks = [[digest(part) for part in sample.split(256)] for sample in mask]
            if chunks != micro['loss_mask']['canonical_chunks_256']:
                raise RuntimeError(f'Actual MF/HP step1501 micro{self.micro} supervision mask differs')
        self.inputs.write(json.dumps(row, allow_nan=False) + '\n')
        self.inputs.flush()
        self.micro += 1

    def on_step_end(self, state, loss, loss_dict, grad_norm, **kwargs):
        del loss, loss_dict, kwargs
        if self.micro != self.trainer.num_micro_batches:
            raise RuntimeError('Observed forward count differs from gradient accumulation')
        row = {'step': state.global_step, 'rank': self.rank, 'micro_batches': self.micro,
               'grad_norm': float(grad_norm), 'metrics': dict(self.trainer.step_env_metrics)}
        self.metrics.write(json.dumps(row, allow_nan=False) + '\n')
        self.metrics.flush()

    def on_train_end(self, state, **kwargs):
        self.handle.remove()
        self.inputs.close()
        self.metrics.close()
        (self.directory / f'complete_rank{self.rank}.json').write_text(
            json.dumps({'global_step': state.global_step, 'epoch': state.epoch}, indent=2))


config = parse_training_args()
trainer = TextTrainer(config)
# The existing lifecycle list has no public registration method.
trainer.base._callbacks.append(Observation(trainer.base, os.environ['HP_OBSERVATIONS']))
trainer.train()
PY
cat > "$RUN_ROOT/scripts/analyze_hp_comparison.py" <<'PY'
#!/usr/bin/env python3
"""Compare eight-rank FSDP8 observer outputs over steps 1501..2500.

No framework/device dependency. Samples: strict same rank, DP/TP/CP coordinate,
step, micro, field, shape, dtype and observed hash equality. This intentionally
does not claim cross-topology sample validation.
"""
import argparse
import csv
import json
import math
from pathlib import Path

FIRST, LAST, RANKS = 1501, 2500, 8
METRICS = {'loss': ('training/foundation_loss', 1.0),
           'mtp_raw': ('training/foundation_loss/mtp', 0.1),
           'aux': ('training/router_aux_loss', 1.0)}


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except Exception as exc:
                raise ValueError(f'{path}:{line_no}: {exc}') from exc


def load_metrics(directory, rank):
    rows={}
    for row in read_jsonl(directory/f'metrics_rank{rank}.jsonl'):
        step=row['step']
        if not FIRST <= step <= LAST: continue
        if row['rank'] != rank or step in rows:
            raise ValueError(f'{directory}: invalid/duplicate metric rank={rank}, step={step}')
        metrics={}
        for name,(field,scale) in METRICS.items():
            value=row['metrics'][field]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(f'{directory}: nonfinite/invalid {field}, step={step}, rank={rank}')
            metrics[name]=float(value)/scale
        rows[step]=metrics
    if set(rows) != set(range(FIRST,LAST+1)):
        raise ValueError(f'{directory}: rank {rank} metric window missing {sorted(set(range(FIRST,LAST+1))-set(rows))[:20]}')
    return rows


def load_inputs(directory, rank, micro_count=1):
    rows={}
    for row in read_jsonl(directory/f'inputs_rank{rank}.jsonl'):
        step=row['step']
        if not FIRST <= step <= LAST: continue
        key=(step,row['micro'])
        if row['rank'] != rank or key in rows:
            raise ValueError(f'{directory}: invalid/duplicate input rank={rank}, key={key}')
        if not {'input_ids','shift_labels'}.issubset(row['fields']):
            raise ValueError(f'{directory}: missing required forward fields rank={rank}, key={key}')
        rows[key]=row
    # FSDP8: DP8, LBS1, GBS8 -> exactly one observed microbatch per rank.
    expected={(step,micro) for step in range(FIRST,LAST+1) for micro in range(micro_count)}
    if set(rows)!=expected:
        raise ValueError(f'{directory}: rank {rank} input window missing/extra {sorted(expected-set(rows))[:10]} / {sorted(set(rows)-expected)[:10]}')
    return rows


def cp_fields_match(left, right, cp_rank):
    """Compare observed, 256-token-aligned CP halves; no raw tensors inferred."""
    required={'input_ids','labels','shift_labels','position_ids'}
    lf,rf=left['fields'],right['fields']
    if set(lf)!=set(rf) or not required.issubset(rf): return False
    if lf['position_ids'].get('starts') != [0] or rf['position_ids'].get('starts') != [cp_rank*2048]: return False
    for field in sorted(lf):
        a,b=lf[field],rf[field]
        if a.get('shape')!=[1,4096] or b.get('shape')!=[1,2048] or a.get('dtype')!=b.get('dtype'):
            return False
        full=a.get('canonical_chunks_256'); part=b.get('canonical_chunks_256')
        if not isinstance(full,list) or len(full)!=1 or len(full[0])!=16: return False
        if not isinstance(part,list) or len(part)!=1 or len(part[0])!=8: return False
        if full[0][cp_rank*8:(cp_rank+1)*8] != part[0]: return False
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('baseline',type=Path,help='baseline observations directory')
    ap.add_argument('candidate',type=Path,help='self-resume observations directory')
    ap.add_argument('--output-dir',type=Path,required=True,help='new, independent results directory')
    ap.add_argument('--threshold',type=float,default=0.005)
    ap.add_argument('--layout',choices=('fsdp8','tp2','cp2','tp2_cp2'),default='fsdp8')
    args=ap.parse_args()
    base=args.baseline.resolve(strict=True); case=args.candidate.resolve(strict=True); out=args.output_dir.resolve()
    if out.is_relative_to(base) or out.is_relative_to(case) or out.exists():
        ap.error('output-dir must be new and outside both inputs')
    if not math.isfinite(args.threshold) or args.threshold<0: ap.error('invalid threshold')
    errors=[]; metrics={}; samples={'status':'PASS','compared_rank_microbatches':0,'mismatches':[]}
    coordinates={}
    base0=case0=None; base_rank_metrics={}; case_rank_metrics={}
    baseline_inputs={rank:load_inputs(base,rank) for rank in range(RANKS)}
    for rank in range(RANKS):
        bm=load_metrics(base,rank); cm=load_metrics(case,rank)
        base_rank_metrics[rank]=bm; case_rank_metrics[rank]=cm
        if rank == 0: base0,case0=bm,cm
        micro_count={'fsdp8':1,'tp2':2,'cp2':2,'tp2_cp2':4}[args.layout]
        ci=load_inputs(case,rank,micro_count)
        for key in sorted(ci):
            right=ci[key]
            if args.layout=='fsdp8':
                left=baseline_inputs[rank][key]
                equal=left==right
                baseline_rank=rank
            else:
                dp,tp,cp=right.get('dp_rank'),right.get('tp_rank'),right.get('cp_rank')
                dp_size={'tp2':4,'cp2':4,'tp2_cp2':2}[args.layout]
                tp_size={'tp2':2,'cp2':1,'tp2_cp2':2}[args.layout]
                cp_size={'tp2':1,'cp2':2,'tp2_cp2':2}[args.layout]
                if (not isinstance(dp,int) or isinstance(dp,bool) or dp not in range(dp_size)
                        or not isinstance(tp,int) or isinstance(tp,bool) or tp not in range(tp_size)
                        or not isinstance(cp,int) or isinstance(cp,bool) or cp not in range(cp_size)):
                    samples['status']='FAIL'
                    samples['mismatches'].append({'kind':'invalid_coordinates','rank':rank,'step':key[0],'micro':key[1],'coords':[dp,tp,cp]})
                    continue
                coordinates.setdefault(key,[]).append((dp,tp,cp))
                baseline_rank=key[1]*dp_size+dp
                left=baseline_inputs[baseline_rank][(key[0],0)]
                base_valid=(left.get('dp_rank')==baseline_rank and left.get('cp_rank')==0 and left.get('tp_rank')==0)
                required={'input_ids','labels','shift_labels','position_ids'}
                if cp_size==1:
                    equal=(base_valid and set(left['fields'])==set(right['fields'])
                           and required.issubset(right['fields']) and left['fields']==right['fields'])
                else:
                    equal=base_valid and cp_fields_match(left,right,cp)
            if not equal:
                samples['status']='FAIL'
                samples['mismatches'].append({'rank':rank,'baseline_rank':baseline_rank,'step':key[0],'micro':key[1],
                    'baseline':left,'candidate':right})
            samples['compared_rank_microbatches']+=1
    if args.layout!='fsdp8':
        dp_size={'tp2':4,'cp2':4,'tp2_cp2':2}[args.layout]
        tp_size={'tp2':2,'cp2':1,'tp2_cp2':2}[args.layout]
        cp_size={'tp2':1,'cp2':2,'tp2_cp2':2}[args.layout]
        micro_count=8//dp_size
        wanted={(dp,tp,cp) for dp in range(dp_size) for tp in range(tp_size) for cp in range(cp_size)}
        for key in ((step,micro) for step in range(FIRST,LAST+1) for micro in range(micro_count)):
            found=coordinates.get(key,[])
            if len(found)!=8 or set(found)!=wanted:
                samples['status']='FAIL'
                samples['mismatches'].append({'kind':'coordinate_coverage','step':key[0],'micro':key[1],'observed':found})
    rows=[]; deltas={name:[] for name in METRICS}
    for step in range(FIRST,LAST+1):
        row={'step':step}
        for name in METRICS:
            b=base0[step][name]; c=case0[step][name]; diff=c-b
            deltas[name].append(diff)
            row.update({f'{name}_baseline':b,f'{name}_candidate':c,f'{name}_delta':diff,
                        f'{name}_cumulative_signed_mean':math.fsum(deltas[name])/len(deltas[name])})
        rows.append(row)
    for name,values in deltas.items():
        signed_mean=math.fsum(values)/len(values)
        metrics[name]={'count':len(values),'signed_mean':signed_mean,'abs_signed_mean':abs(signed_mean),
            'mae':math.fsum(map(abs,values))/len(values),'max_abs':max(map(abs,values)),
            'positive':sum(v>0 for v in values),'negative':sum(v<0 for v in values),
            'zero':sum(v==0 for v in values),'threshold':args.threshold,
            'pass':abs(signed_mean)<=args.threshold}
    rank_statistics={}
    for rank in range(RANKS):
        per_metric={}
        for name in METRICS:
            # Rank0 is the published native baseline. Report and gate each
            # candidate rank on the same user-specified complete-window rule.
            diffs=[case_rank_metrics[rank][step][name]-base0[step][name] for step in range(FIRST,LAST+1)]
            mean=math.fsum(diffs)/len(diffs)
            per_metric[name]={'count':len(diffs),'signed_mean':mean,'abs_signed_mean':abs(mean),
                'mae':math.fsum(map(abs,diffs))/len(diffs),'max_abs':max(map(abs,diffs)),
                'positive':sum(v>0 for v in diffs),'negative':sum(v<0 for v in diffs),
                'zero':sum(v==0 for v in diffs),'threshold':args.threshold,
                'pass':abs(mean)<=args.threshold}
            if not per_metric[name]['pass']:
                errors.append({'kind':'rank_metric_threshold_exceeded','rank':rank,'metric':name})
        rank_statistics[str(rank)]=per_metric
    rank_spread={}
    for label,all_ranks in [('baseline',base_rank_metrics),('candidate',case_rank_metrics)]:
        rank_spread[label]={}
        for name in METRICS:
            spans=[max(all_ranks[r][step][name] for r in range(RANKS))-
                   min(all_ranks[r][step][name] for r in range(RANKS)) for step in range(FIRST,LAST+1)]
            rank_spread[label][name]={'max_step_spread':max(spans),'mean_step_spread':math.fsum(spans)/len(spans)}
    # Explicit TP-pair spread is diagnostic only; no extra bitwise gate.
    if args.layout=='tp2':
        rank_spread['candidate_tp_pairs']={}
        for first in (0,2,4,6):
            pair={}
            for name in METRICS:
                diffs=[case_rank_metrics[first+1][step][name]-case_rank_metrics[first][step][name]
                       for step in range(FIRST,LAST+1)]
                pair[name]={'max_abs_step_diff':max(map(abs,diffs)),
                            'signed_mean_diff':math.fsum(diffs)/len(diffs)}
            rank_spread['candidate_tp_pairs'][f'{first}-{first+1}']=pair
    if samples['status']!='PASS': errors.append({'kind':'observed_samples_mismatch','count':len(samples['mismatches'])})
    if any(not v['pass'] for v in metrics.values()): errors.append({'kind':'metric_threshold_exceeded'})
    report={'status':'PASS' if not errors else 'FAIL','baseline':str(base),'candidate':str(case),
            'window':[FIRST,LAST],'metric_sources':METRICS,'metrics':metrics,'rank_statistics':rank_statistics,
            'rank_spread':rank_spread,'samples':samples,'errors':errors,
            'sample_scope':('FSDP8 same-layout' if args.layout=='fsdp8' else 'observed coordinates and ordered canonical 256-token chunks; labels IGNORE_INDEX normalized; CP layouts assume verified contiguous half-slices'), 'layout':args.layout}
    out.mkdir(parents=True)
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out/'statistics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'errors':len(errors),'output':str(out)}))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__': raise SystemExit(main())




PY
cat > "$RUN_ROOT/scripts/analyze_mf_hp_single.py" <<'PY'
#!/usr/bin/env python3
"""Read-only observed MF/HP single-rank 1501..2500 comparison, CPU only."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
import numpy as np
from hyper_parallel.data.constants import IGNORE_INDEX
from hyper_parallel.data.indexed.indexed_data_reader import IndexedDataReader

FIRST,LAST=1501,2500

def sha(data): return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()
def chunks(a):
    a=np.asarray(a,dtype=np.int64).reshape(1,-1)
    return [[sha(row[i:i+256]) for i in range(0,row.size,256)] for row in a]
def load_jsonl(path):
    with path.open(encoding='utf8') as f:
        for n,line in enumerate(f,1):
            try: yield json.loads(line)
            except Exception as e: raise ValueError(f'{path}:{n}: {e}') from e

def window(path,keyfn):
    out={}
    for row in load_jsonl(path):
        step=row['step']
        if FIRST<=step<=LAST:
            key=keyfn(row)
            if key in out: raise ValueError(f'duplicate {path} {key}')
            out[key]=row
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mf-run',type=Path,required=True); ap.add_argument('--hp-run',type=Path,required=True)
    ap.add_argument('--data-prefix',required=True); ap.add_argument('--proof-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    out=args.output.resolve()
    if out.exists(): raise SystemExit('output already exists; choose a new directory')
    if out.is_relative_to(args.mf_run.resolve()) or out.is_relative_to(args.hp_run.resolve()): raise SystemExit('output inside input run')
    mf=window(args.mf_run/'observations.jsonl',lambda r:r['step'])
    hp=window(args.hp_run/'observations/metrics_rank0.jsonl',lambda r:r['step'])
    hi=window(args.hp_run/'observations/inputs_rank0.jsonl',lambda r:(r['step'],r['micro']))
    steps=set(range(FIRST,LAST+1)); micros={(s,m) for s in steps for m in (0,1)}
    if set(mf)!=steps or set(hp)!=steps or set(hi)!=micros: raise ValueError('window incomplete or has invalid micro identities')
    pat=re.compile(r'step:\[\s*(\d+)\s*/\s*2500\].*?consumed_samples:\s*(\d+).*?\blr:\s*([0-9.eE+-]+)')
    mf_lr={}
    with (args.mf_run/'native.log').open(encoding='utf8',errors='replace') as f:
        for line in f:
            match=pat.search(line)
            if match:
                step=int(match[1])
                if step in steps:
                    if step in mf_lr: raise ValueError(f'duplicate MF native LR at {step}')
                    mf_lr[step]=(int(match[2]),match[3])
    if set(mf_lr)!=steps: raise ValueError(f'MF native LR missing {len(steps-set(mf_lr))} steps')
    reader=IndexedDataReader(args.data_prefix,mmap=True)
    sample_errors=[]; sample_counts={k:0 for k in ('mf_input','hp_input','mf_labels','hp_labels','hp_shift_labels','mf_mask','hp_derived_mask','mf_positions','hp_positions')}
    stat_errors=[]; rows=[]; ds={x:[] for x in ('loss','mtp_raw','aux_raw')}; lr_max=0.; lr_bad=[]
    for step in range(FIRST,LAST+1):
        mr,hr=mf[step],hp[step]
        if mr.get('consumed_samples')!=step*2 or mf_lr[step][0]!=step*2 or hr.get('rank')!=0 or hr.get('micro_batches')!=2:
            stat_errors.append({'step':step,'kind':'progress_mismatch'})
        mm=hr['metrics']
        if mm.get('data/step_samples')!=2 or mm.get('data/consumed_samples')!=(step-1500)*2:
            stat_errors.append({'step':step,'kind':'hp_session_progress','observed':mm.get('data/consumed_samples')})
        hplr=mm['training/lr']; text=mf_lr[step][1]; mflr=float(text)
        # MF native log has six decimal digits after the decimal in scientific notation.
        # Formatting the actual HP value to the same precision tests the printed interval.
        lr_max=max(lr_max,abs(hplr-mflr))
        if format(hplr,'.6e') != text.lower(): lr_bad.append({'step':step,'hp':hplr,'mf_printed':text})
        row={'step':step,'mf_consumed_samples':mr['consumed_samples'],'hp_session_consumed_samples':mm.get('data/consumed_samples'),'mf_lr_printed':text,'hp_lr':hplr}
        for name,mkey,hkey,scale in [('loss','loss','training/foundation_loss',1),('mtp_raw','mtp_raw','training/foundation_loss/mtp',.1),('aux_raw','aux_raw','training/router_aux_loss',1)]:
            a=mr[mkey]; b=mm[hkey]/scale
            if not all(isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x) for x in (a,b)): raise ValueError(f'nonfinite {step} {name}')
            d=b-a; ds[name].append(d); row.update({f'{name}_mf':a,f'{name}_hp':b,f'{name}_delta':d,f'{name}_cumulative_signed_mean':math.fsum(ds[name])/len(ds[name])})
        rows.append(row)
        if len(mr['microbatches'])!=2: raise ValueError(f'MF {step} has wrong micro count')
        for micro in (0,1):
            index=(step-1)*2+micro; h=hi[(step,micro)]; m=mr['microbatches'][micro]
            if (h.get('rank'),h.get('dp_rank'),h.get('tp_rank'),h.get('cp_rank'))!=(0,0,0,0): raise ValueError(f'HP coords invalid {step}/{micro}')
            length=min(int(reader.sequence_lengths[index]),4097)
            textrow=np.full(4097,3,dtype=np.int64); textrow[:length]=reader.get(index,length=length)
            inputs=textrow[:-1].copy(); labels=textrow[1:].copy()
            mask=(labels!=3).astype(np.int64); hp_labels=labels.copy(); hp_labels[mask==0]=IGNORE_INDEX
            mf_labels=labels.copy(); mf_labels[mask==0]=0; inputs[inputs==3]=0
            pos=np.arange(4096,dtype=np.int64)
            fields=h['fields']; tests=[
              ('mf_input',m['input_ids'],inputs,np.int32),('hp_input',fields['input_ids'],inputs,np.int64),
              ('mf_labels',m['labels'],mf_labels,np.int32),('hp_labels',fields['labels'],hp_labels,np.int64),
              ('hp_shift_labels',fields['shift_labels'],hp_labels,np.int64),('mf_mask',m['loss_mask'],mask,np.int32),
              ('mf_positions',m['position_ids'],pos,np.int32),('hp_positions',fields['position_ids'],pos,np.int64)]
            for label,observed,array,dtype in tests:
                valid=(observed.get('shape')==[1,4096] and observed.get('sha256')==sha(array.astype(dtype)))
                # For labels HP observer normalizes ignore values before chunk hashing.
                normalized=mf_labels if label in ('hp_labels','hp_shift_labels') else array
                valid=valid and observed.get('canonical_chunks_256')==chunks(normalized)
                if not valid: sample_errors.append({'step':step,'micro':micro,'record_index':index,'field':label})
                else: sample_counts[label]+=1
            # Actual HP raw labels hash matched reconstructed IGNORE_INDEX labels;
            # its runtime loss_mask is (labels >= 0), independently reconstructable.
            if np.array_equal((hp_labels>=0).astype(np.int64),mask): sample_counts['hp_derived_mask']+=1
            else: sample_errors.append({'step':step,'micro':micro,'field':'hp_derived_mask'})
    metrics={}
    for name,values in ds.items():
        mean=math.fsum(values)/len(values)
        metrics[name]={'count':len(values),'signed_mean':mean,'abs_signed_mean':abs(mean),'mae':math.fsum(map(abs,values))/len(values),'max_abs':max(map(abs,values)),'positive':sum(x>0 for x in values),'negative':sum(x<0 for x in values),'zero':sum(x==0 for x in values),'threshold':.005,'pass':abs(mean)<=.005}
    proofs={}
    for name in ('mf1500-closure-v2.json','mf1500-independent-mapping-v1.json','hp-single-payload-proof-v1.json','hp-single-live-proof-v2.json'):
        path=args.proof_root/name; raw=path.read_bytes(); data=json.loads(raw)
        proofs[name]={'sha256':hashlib.sha256(raw).hexdigest(),'status':data.get('status',data.get('result', 'PASS' if data.get('passed') is True else 'UNKNOWN'))}
    sample_audit={'microbatches':2000,'checked':sample_counts,'mismatches':sample_errors,'sequence_length':4096,
        'method':'Reconstruct sequential source records 3000..4999 from native IndexedDataReader; independently check actual MF/HP raw tensor hashes and canonical chunks; derive HP mask from verified actual shift_labels using labels>=0, compare MF observed mask.',
        'limitations':['MF actual_seq_len is observed but not independently compared to an HP field','HP mask is derived from byte-verified actual labels and framework contract, not separately observed at loss boundary']}
    summary={'status':('FAIL' if stat_errors or sample_errors or not all(x['pass'] for x in metrics.values()) else ('PASS_WITH_LR_PRINT_DIFFERENCES' if lr_bad else 'PASS')),
      'window':[FIRST,LAST],'metrics':metrics,'sample_status':'PASS' if not sample_errors else 'FAIL','sample_counts':sample_counts,
      'progress_errors':stat_errors,'lr':{'comparison':'HP post-step scalar formatted to MF native scientific six decimal digits; differences reported, no invented tolerance gate','steps':len(mf_lr),'printed_value_mismatch_count':len(lr_bad),'mismatches':lr_bad,'max_abs_to_printed':lr_max},
      'proofs':proofs,'rng_scope':'MF RNG is not converted to Torch; HP runtime default RNG, no claim of MS RNG restore',
      'inputs':{'mf_observations':str(args.mf_run/'observations.jsonl'),'hp_observations':str(args.hp_run/'observations'),'indexed_prefix':args.data_prefix}}
    out.mkdir(parents=True)
    with (out/'per-step.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (out/'samples.json').write_text(json.dumps(sample_audit,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'status':summary['status'],'sample_errors':len(sample_errors),'lr_errors':len(lr_bad),'progress_errors':len(stat_errors),'metrics':metrics},ensure_ascii=False))
    return 0 if summary['status'] in ('PASS','PASS_WITH_LR_PRINT_DIFFERENCES') else 1
if __name__=='__main__': raise SystemExit(main())

PY
cat > "$RUN_ROOT/scripts/compare_hp_restored_payload.py" <<'PY'
#!/usr/bin/env python3
"""Compare trusted HP DCP checkpoint with eight after_load observer payloads.

CPU only, no process group. Reads the full DCP once (~25 GB for this
case); checkpoint pickle and torch.save inputs must be trusted. No writes to
checkpoint or observer directories. This verifies the captured after_load
payload, before optimizer application, not live aliasing or later training.
"""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardLoadPlanner
from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemReader
from hyper_parallel.core.distributed_checkpoint.metadata import TensorStorageMetadata
from hyper_parallel.core.distributed_checkpoint.utils import str_to_dtype, set_element


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def json_value(value):
    # Exactly mirrors observe_train.py json_value, including list/tuple treatment.
    if torch.is_tensor(value):
        return {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': digest(value)}
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def same_value(a, b):
    # Preserve bool/int type distinction, but JSON tuples become lists.
    return json.dumps(a, sort_keys=True, ensure_ascii=False, allow_nan=False) == json.dumps(b, sort_keys=True, ensure_ascii=False, allow_nan=False)


def compare_saved_subset(saved, observed, path='extra_state'):
    """Verify saved leaves and report destination-only skeleton values."""
    failures, additions = [], []
    if isinstance(saved, dict):
        if not isinstance(observed, dict): return [path + ': type mismatch'], additions
        for key, value in saved.items():
            if key not in observed: failures.append(path + '.' + key + ': missing')
            else:
                e, a = compare_saved_subset(value, observed[key], path + '.' + key)
                failures.extend(e); additions.extend(a)
        additions.extend(path + '.' + key for key in observed.keys()-saved.keys())
    elif not same_value(saved, observed): failures.append(path + ': value mismatch')
    return failures, additions

def overlaps(a, b):
    ao, az = a; bo, bz = b
    return all(x < y+n and y < x+m for x,m,y,n in zip(ao,az,bo,bz))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('observations', type=Path, help='directory containing loaded_payload_rankN.json')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ranks', type=int, default=8)
    parser.add_argument('--allow-nonleader-no-loader', action='store_true',
                        help='Allow absent embedded loader fields only for a proven TP nonleader without loader')
    args = parser.parse_args()
    ckpt = args.checkpoint.resolve(strict=True)
    obs = args.observations.resolve(strict=True)
    output = args.output.resolve()
    if output.is_relative_to(ckpt) or output.is_relative_to(obs):
        parser.error('output must be outside checkpoint and observations')
    result = {'checkpoint': str(ckpt), 'observations': str(obs),
              'scope': 'DCP full CPU read vs after_load payload; not applied optimizer/live aliases or training',
              'errors': [], 'ranks': [], 'coverage': {}, 'missing_source_policy': 'No unverified exceptions allowed'}
    errors = result['errors']
    metadata = FileSystemReader(ckpt).load_metadata()
    # Use the standard native planner/reader over a flat FQN skeleton. The
    # convenience merge planner is broken for nested planner_data and is not
    # involved in this comparison. Bytes placeholders are loaded in place.
    need_embedded = any(not (ckpt/'extra_state'/f'extra_state_rank_{rank}.pt').is_file() for rank in range(args.ranks))
    embedded_keys = {key for key in metadata.state_dict_metadata if key == 'extra_state' or key.startswith('extra_state.')} if need_embedded else set()
    if need_embedded and not embedded_keys:
        raise RuntimeError('Missing rank-local extra_state and embedded extra_state metadata')
    source = {
        key: torch.empty(tuple(meta.size), dtype=str_to_dtype(meta.properties.dtype), device='cpu')
        if isinstance(meta, TensorStorageMetadata) else None
        for key, meta in metadata.state_dict_metadata.items()
        if key.startswith(('model.', 'optimizer.')) or key in embedded_keys
    }
    reader = FileSystemReader(ckpt)
    planner = StandardLoadPlanner(allow_partial_load=False)
    planner.configure_planner(source, metadata, is_coordinator=True, rank=0,
                              flatten_state_dict=False, use_collectives=False,
                              broadcast_replicated_tensors=False)
    reader.configure_reader(metadata, is_coordinator=True, rank=0)
    plan = reader.optimize_local_plan(planner.build_local_plan())
    plan = reader.optimize_global_plan(planner.build_global_plan([plan]))
    plan = planner.finalize_plan(plan)
    reader.execute_read(plan, planner)
    embedded_root = {}
    for key in sorted(embedded_keys):
        path = (metadata.planner_data or {}).get(key)
        if path is None:
            raise RuntimeError(f'Embedded extra_state lacks planner_data mapping for {key}')
        if not path or path[0] != 'extra_state':
            raise RuntimeError(f'Invalid embedded extra_state mapping for {key}: {path}')
        set_element(embedded_root, path, source.pop(key))
    embedded_extra = embedded_root.get('extra_state')
    if need_embedded and not isinstance(embedded_extra, dict):
        raise RuntimeError('Embedded extra_state did not decode into a dict')
    metadata_keys = {key for key in metadata.state_dict_metadata if key.startswith(('model.', 'optimizer.'))}
    if set(source) != metadata_keys:
        errors.append({'kind':'source_fqn_mismatch', 'missing':sorted(metadata_keys-set(source)), 'extra':sorted(set(source)-metadata_keys)})
    seen = set(); boxes = defaultdict(set)
    for rank in range(args.ranks):
        rank_errors=[]
        entry={'rank':rank, 'tensor_compared':0, 'value_compared':0, 'errors':rank_errors}
        result['ranks'].append(entry)
        try:
            data=json.loads((obs/f'loaded_payload_rank{rank}.json').read_text(encoding='utf-8'))
            if Path(data['checkpoint']).name != ckpt.name:
                rank_errors.append({'kind':'checkpoint_name','observed':data['checkpoint'],'expected_name':ckpt.name})
            # Same rank-local extra_state that the native checkpointer reads.
            rank_path=ckpt/'extra_state'/f'extra_state_rank_{rank}.pt'
            saved_extra=(torch.load(rank_path,map_location='cpu',weights_only=False)
                         if rank_path.is_file() else embedded_extra)
            entry['extra_state_source']='per_rank' if rank_path.is_file() else 'embedded'
            saved_json = json_value(saved_extra)
            observed_extra = data.get('extra_state')
            if rank_path.is_file():
                if not same_value(saved_json, observed_extra):
                    rank_errors.append({'kind':'extra_state_mismatch'})
                else:
                    entry['extra_state']='exact_json_match'
            else:
                # A TP nonleader legitimately has no source DataLoader. Require
                # both observed coordinates and live absence; never infer from
                # rank parity or treat another missing field as acceptable.
                if args.allow_nonleader_no_loader and 'train_dataloader' in saved_json:
                    live_path=obs/f'live_restore_rank{rank}.json'
                    input_path=obs/f'inputs_rank{rank}.jsonl'
                    live=json.loads(live_path.read_text(encoding='utf-8')) if live_path.is_file() else None
                    with input_path.open(encoding='utf-8') as f:
                        first=json.loads(next(f))
                    tp_rank=first.get('tp_rank')
                    if (isinstance(tp_rank,int) and not isinstance(tp_rank,bool) and tp_rank>0
                            and isinstance(live,dict) and live.get('dataloader') is None
                            and isinstance(observed_extra,dict) and not observed_extra.get('train_dataloader')):
                        saved_json=dict(saved_json)
                        saved_json.pop('train_dataloader')
                        entry['loader_source']='not applicable: observed TP nonleader without DataLoader'
                failures, additions = compare_saved_subset(saved_json, observed_extra)
                if failures: rank_errors.append({'kind':'embedded_extra_state_mismatch','paths':failures})
                entry['extra_state']={'saved_fields':'match' if not failures else 'mismatch',
                                      'destination_only_skeleton_fields':additions,
                                      'note':'destination-only values were not present in checkpoint and are not verified'}
            rank_seen=set()
            for item in data['tensors']:
                key=item['key']; offsets=tuple(item['offsets']); sizes=tuple(item['sizes'])
                token=(key,offsets,sizes)
                if token in rank_seen:
                    rank_errors.append({'kind':'duplicate_tensor_box','key':key,'offsets':offsets,'sizes':sizes}); continue
                rank_seen.add(token)
                value=source.get(key)
                if key not in source:
                    rank_errors.append({'kind':'missing_source_tensor','key':key,'offsets':offsets,'sizes':sizes,
                                        'note':'No absence is accepted without independently verified reason'}); continue
                if not torch.is_tensor(value):
                    rank_errors.append({'kind':'source_not_tensor','key':key}); continue
                shape=tuple(value.shape)
                if tuple(item['global_shape']) != shape or len(offsets)!=len(shape) or len(sizes)!=len(shape) or any(o<0 or n<0 or o+n>s for o,n,s in zip(offsets,sizes,shape)):
                    rank_errors.append({'kind':'invalid_tensor_box','key':key,'source_shape':shape,'item':item}); continue
                if item['dtype'] != str(value.dtype):
                    rank_errors.append({'kind':'dtype_mismatch','key':key,'expected':str(value.dtype),'observed':item['dtype']}); continue
                selection=tuple(slice(o,o+n) for o,n in zip(offsets,sizes))
                shard=value[selection] if shape else value
                actual=digest(shard)
                if actual != item['sha256']:
                    rank_errors.append({'kind':'tensor_sha256_mismatch','key':key,'offsets':offsets,'sizes':sizes,'expected':actual,'observed':item['sha256']})
                else:
                    entry['tensor_compared']+=1
                seen.add(key); boxes[key].add((offsets,sizes))
            for key, observed in data['values'].items():
                if key not in source:
                    rank_errors.append({'kind':'missing_source_value','key':key,'note':'No absence is accepted without independently verified reason'}); continue
                if torch.is_tensor(source[key]):
                    rank_errors.append({'kind':'source_value_is_tensor','key':key}); continue
                expected=json_value(source[key])
                if not same_value(expected,observed):
                    rank_errors.append({'kind':'value_mismatch','key':key,'expected':expected,'observed':observed})
                else:
                    entry['value_compared']+=1
                seen.add(key)
        except Exception as exc:
            rank_errors.append({'kind':'rank_exception','type':type(exc).__name__,'message':str(exc)})
        if rank_errors: errors.append({'kind':'rank_failed','rank':rank,'count':len(rank_errors)})
    for key in sorted(metadata_keys-seen):
        errors.append({'kind':'source_never_observed','key':key})
    # Across ranks, repeated identical boxes are replicas. Any partial overlaps
    # are rejected; otherwise summing unique volumes proves full global coverage.
    for key, meta in metadata.state_dict_metadata.items():
        if not key.startswith(('model.','optimizer.')) or not isinstance(meta,TensorStorageMetadata): continue
        unique=list(boxes.get(key,set())); total=sum(math.prod(size) for _,size in unique)
        overlap=any(overlaps(a,b) for i,a in enumerate(unique) for b in unique[:i])
        expected=math.prod(meta.size)
        result['coverage'][key]={'unique_boxes':len(unique),'covered_elements':total,'expected_elements':expected,'overlap':overlap}
        if total!=expected or overlap:
            errors.append({'kind':'tensor_coverage','key':key,**result['coverage'][key]})
    result['result']='PASS' if not errors else 'FAIL'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'result':result['result'],'errors':len(errors),'ranks':len(result['ranks']),'output':str(output)}))
    return 1 if errors else 0

if __name__=='__main__':
    raise SystemExit(main())
PY
cat > "$RUN_ROOT/scripts/compare_hp_live_restore.py" <<'PY'
#!/usr/bin/env python3
"""Compare captured after_load and on_train_begin JSON, without checkpoint IO.

Results cover observed local parameter hashes/aliases/slots and selected
scheduler/loader/RNG state. They do not establish a full training acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path


def leaves(value, prefix=''):
    if isinstance(value,dict):
        for key,child in value.items():
            yield from leaves(child, f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(value,list):
        for i,child in enumerate(value): yield from leaves(child,f'{prefix}.{i}')
    else:
        yield prefix,value


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def build_owner_evidence(root, owner_root, ranks):
    """Load real assignments; an external capture must match original snapshots."""
    evidence={'directory':str(owner_root),'external':owner_root != root,'files':{},'gaps':[]}
    contexts=[]
    for rank in range(ranks):
        paths={name:owner_root/f'{name}_rank{rank}.json' for name in
               ('muon_assignments','loaded_payload','live_restore','restore')}
        for name,path in paths.items():
            evidence['files'][f'{name}_rank{rank}']=hashlib.sha256(path.read_bytes()).hexdigest()
        assignments=read_json(paths['muon_assignments']); payload=read_json(paths['loaded_payload'])
        live=read_json(paths['live_restore']); started=read_json(paths['restore'])
        if owner_root != root:
            original_payload=read_json(root/f'loaded_payload_rank{rank}.json')
            original_live=read_json(root/f'live_restore_rank{rank}.json')
            original_started=read_json(root/f'restore_rank{rank}.json')
            for key in ('tensors','values','extra_state'):
                if payload.get(key)!=original_payload.get(key):
                    raise ValueError(f'rank {rank}: external ownership payload differs: {key}')
            for key in ('parameters','scheduler','dataloader'):
                if live.get(key)!=original_live.get(key):
                    raise ValueError(f'rank {rank}: external ownership live differs: {key}')
            for key,value in original_live.get('rng',{}).items():
                if live.get('rng',{}).get(key)!=value:
                    raise ValueError(f'rank {rank}: external ownership RNG differs: {key}')
            for key in set(live.get('rng',{}))-set(original_live.get('rng',{})):
                evidence['gaps'].append({'rank':rank,'field':'original live RNG.'+key,'note':'only in separate ownership capture'})
            for key in ('global_step','epoch','num_micro_batches','source_head'):
                if started.get(key)!=original_started.get(key):
                    raise ValueError(f'rank {rank}: external ownership restore identity differs: {key}')
            coords=original_started.get('parallel_coordinates')
            if coords is None:
                inp=root/f'inputs_rank{rank}.jsonl'
                if not inp.is_file(): raise ValueError(f'rank {rank}: no original coordinates evidence')
                with inp.open(encoding='utf-8') as f: first=json.loads(next(f))
                coords={k:first.get(k) for k in ('dp_rank','tp_rank','cp_rank')}
            if coords != started.get('parallel_coordinates'):
                raise ValueError(f'rank {rank}: external ownership coordinates differ')
        mapping={}
        for group in assignments:
            for record in group['records']:
                name=record['name']
                if name in mapping: raise ValueError(f'rank {rank}: duplicate owner assignment {name}')
                mapping[name]=(group,record)
        contexts.append({'assignments':mapping,'payload':payload,'live':live})
    return evidence,contexts


def prove_nonowner(contexts, rank, name, item):
    """Prove a retained owner in the actual replicate-PG connected component."""
    pair=contexts[rank]['assignments'].get(name)
    if pair is None: raise ValueError('missing assignment')
    group,record=pair; owner=record.get('owner'); coord=group.get('replicate_group_ranks')
    if not group.get('is_replicated') or not owner or record.get('is_owned') is not False or coord==owner:
        raise ValueError('record is not a proven replicated non-owner')
    signature=(group.get('group'),group.get('batch'),group.get('sub_batch'),record.get('index'))
    reachable={rank}; queue=[rank]
    while queue:
        current=queue.pop(); found=contexts[current]['assignments'].get(name)
        if found is None: raise ValueError(f'rank {current} missing assignment in replica component')
        g,r=found
        if (g.get('group'),g.get('batch'),g.get('sub_batch'),r.get('index'))!=signature or r.get('owner')!=owner or not g.get('is_replicated'):
            raise ValueError(f'rank {current} inconsistent owner assignment')
        coords=g.get('replicate_group_ranks'); pgs=g.get('replicate_pgs'); sizes=g.get('replicate_sizes')
        if not (isinstance(coords,list) and isinstance(pgs,list) and isinstance(sizes,list) and len(coords)==len(pgs)==len(sizes)==len(owner)):
            raise ValueError(f'rank {current} incomplete multi-dimensional replica evidence')
        if r.get('is_owned') != (coords==owner): raise ValueError(f'rank {current} owner coordinate inconsistency')
        for members,size in zip(pgs,sizes):
            if not isinstance(members,list) or current not in members or len(members)!=size or len(set(members))!=len(members):
                raise ValueError(f'rank {current} invalid replica group')
            for peer in members:
                if not isinstance(peer,int) or not 0<=peer<len(contexts): raise ValueError('replica group outside observed ranks')
                if peer not in reachable: reachable.add(peer); queue.append(peer)
    owners=[]
    key='optimizer.state.'+name+'.momentum_buffer'
    for peer in sorted(reachable):
        g,r=contexts[peer]['assignments'][name]
        if not r['is_owned']: continue
        payload_items=[x for x in contexts[peer]['payload']['tensors'] if x['key']==key]
        state=contexts[peer]['live']['parameters'][name].get('optimizer_state_hashes',{}).get('momentum_buffer')
        if len(payload_items)!=1 or not isinstance(state,dict): continue
        other=payload_items[0]
        if all(other.get(k)==item.get(k) for k in ('offsets','sizes','global_shape','dtype','sha256')) and state.get('sha256')==item['sha256'] and state.get('dtype')==item['dtype']:
            owners.append(peer)
    if len(owners)!=1: raise ValueError(f'expected one matching owner in replica component; found {owners}')
    return {'owner_rank':owners[0],'replica_component':sorted(reachable),'owner_coord':owner,
            'offsets':item['offsets'],'sizes':item['sizes'],'dtype':item['dtype'],'sha256':item['sha256']}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('observations',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--ranks',type=int,default=1)
    ap.add_argument('--expected-parameters',type=int,default=61)
    ap.add_argument('--expected-step',type=int,default=1500)
    ap.add_argument('--allow-nonleader-no-loader',action='store_true',help='Treat missing loader as not applicable only with nonzero TP rank evidence')
    ap.add_argument('--tp-nonleader-ranks',default='',help='Comma-separated ranks, only when independently verified from actual mesh; permits no loader before first forward')
    ap.add_argument('--source-rng-not-saved',action='store_true',help='Source checkpoint lacks compatible RNG; compare runtime state while marking its provenance')
    ap.add_argument('--ownership-observations',type=Path,help='Separate same-state native owner capture, verified against this run before use')
    args=ap.parse_args()
    verified_nonleaders={int(x) for x in args.tp_nonleader_ranks.split(',') if x.strip()}
    root=args.observations.resolve(strict=True); out=args.output.resolve()
    if out.is_relative_to(root): ap.error('output must be outside observations')
    result={'ranks':[],'errors':[],'scope':'Captured after_load to live on_train_begin, local hashes and alias identity; no disk readback'}
    owner_root=args.ownership_observations.resolve(strict=True) if args.ownership_observations else root
    if out.is_relative_to(owner_root): ap.error('output must be outside ownership observations')
    owner_contexts=None; owner_failure=None
    for rank in range(args.ranks):
        rr={'rank':rank,'errors':[],'parameters_compared':0,'slots_compared':0,'unchecked':[]}
        result['ranks'].append(rr); errors=rr['errors']
        try:
            payload=json.loads((root/f'loaded_payload_rank{rank}.json').read_text(encoding='utf-8'))
            live=json.loads((root/f'live_restore_rank{rank}.json').read_text(encoding='utf-8'))
            started=json.loads((root/f'restore_rank{rank}.json').read_text(encoding='utf-8'))
            if started.get('global_step')!=args.expected_step:
                errors.append({'kind':'global_step','observed':started.get('global_step')})
            tensors={}
            for item in payload['tensors']:
                key=item['key']
                if key in tensors:
                    errors.append({'kind':'ambiguous_payload_key','key':key}); continue
                tensors[key]=item
            values=payload['values']; params=live['parameters']
            if len(params)!=args.expected_parameters:
                errors.append({'kind':'parameter_count','expected':args.expected_parameters,'observed':len(params)})
            seen_slots=set()
            for name,entry in params.items():
                model_key='model.'+name
                if model_key not in tensors:
                    errors.append({'kind':'missing_model','key':model_key}); continue
                expected=tensors[model_key]
                if entry.get('model_sha256')!=expected['sha256']:
                    errors.append({'kind':'model_hash','key':model_key})
                if entry.get('main_is_model') is not True:
                    errors.append({'kind':'expected_fp32_main_alias','name':name,'observed':entry.get('main_is_model')})
                if entry.get('main_sha256') != entry.get('model_sha256'):
                    errors.append({'kind':'main_model_hash','name':name})
                main_key='optimizer._mixed_precision_optimizer.fp32_from_fp16_params.'+name
                if main_key in tensors and tensors[main_key]['sha256']!=entry.get('main_sha256'):
                    errors.append({'kind':'main_payload_hash','key':main_key})
                if main_key not in tensors:
                    model_dtype=entry.get('model_parameter',{}).get('dtype')
                    main_dtype=entry.get('main_parameter',{}).get('dtype')
                    if (entry.get('main_is_model') is True and model_dtype=='torch.float32'
                            and main_dtype=='torch.float32'
                            and expected.get('dtype')=='torch.float32'
                            and entry.get('model_sha256')==expected.get('sha256')
                            and entry.get('main_sha256')==expected.get('sha256')):
                        rr['fp32_main_alias_verified']=rr.get('fp32_main_alias_verified',0)+1
                    else:
                        errors.append({'kind':'missing_main_without_proven_fp32_alias','name':name,
                                       'model_dtype':model_dtype,'main_dtype':main_dtype,
                                       'payload_dtype':expected.get('dtype')})
                prefix='optimizer.state.'+name+'.'
                expected_slots={k[len(prefix):]:v for k,v in tensors.items() if k.startswith(prefix)}
                expected_values={k[len(prefix):]:v for k,v in values.items() if k.startswith(prefix)}
                live_slots=entry.get('optimizer_state_hashes',{})
                for slot, value in live_slots.items():
                    if slot in expected_slots:
                        if not isinstance(value,dict) or value.get('sha256')!=expected_slots[slot]['sha256']:
                            errors.append({'kind':'slot_tensor_hash','name':name,'slot':slot})
                        else: rr['slots_compared']+=1
                        seen_slots.add(prefix+slot)
                    elif slot in expected_values:
                        if value!=expected_values[slot]: errors.append({'kind':'slot_value','name':name,'slot':slot,'expected':expected_values[slot],'actual':value})
                        else: rr['slots_compared']+=1
                        seen_slots.add(prefix+slot)
                    else:
                        errors.append({'kind':'live_slot_missing_payload','name':name,'slot':slot})
                for slot in set(expected_slots)|set(expected_values):
                    if slot in live_slots: continue
                    if slot=='momentum_buffer' and slot in expected_slots:
                        if owner_contexts is None and owner_failure is None:
                            try:
                                result['ownership_evidence'],owner_contexts=build_owner_evidence(root,owner_root,args.ranks)
                            except Exception as exc:
                                owner_failure=f'{type(exc).__name__}: {exc}'
                        try:
                            if owner_failure is not None: raise ValueError(owner_failure)
                            receipt=prove_nonowner(owner_contexts,rank,name,expected_slots[slot])
                            rr.setdefault('proven_nonowner_slots',[]).append({'name':name,**receipt})
                            seen_slots.add(prefix+slot)
                        except Exception as exc:
                            errors.append({'kind':'payload_slot_missing_live','name':name,'slot':slot,'owner_error':str(exc)})
                    else:
                        errors.append({'kind':'payload_slot_missing_live','name':name,'slot':slot})
                rr['parameters_compared']+=1
            unassigned=[k for k in (*tensors,*values) if k.startswith('optimizer.state.') and k not in seen_slots]
            if unassigned: errors.append({'kind':'unassigned_payload_slots','keys':sorted(unassigned)})
            extra=payload.get('extra_state') or {}
            saved_scheduler=extra.get('lr_scheduler')
            if saved_scheduler:
                if saved_scheduler != live.get('scheduler'): errors.append({'kind':'scheduler_mismatch','expected':saved_scheduler,'actual':live.get('scheduler')})
                else: rr['scheduler']='exact_match'
            else: rr['unchecked'].append({'kind':'scheduler_absent_in_extra'})
            saved_loader=extra.get('train_dataloader')
            actual_loader=live.get('dataloader')
            if actual_loader is None:
                # Require actual observed TP coordinates, never infer from global rank.
                inputs=root/f'inputs_rank{rank}.jsonl'
                tp_rank=None
                if inputs.exists():
                    with inputs.open(encoding='utf-8') as f:
                        first=f.readline()
                    if first: tp_rank=json.loads(first).get('tp_rank')
                if args.allow_nonleader_no_loader and ((isinstance(tp_rank,int) and tp_rank>0) or rank in verified_nonleaders):
                    rr['loader']={'status':'not_applicable_tp_nonleader','evidence':'input tp_rank' if isinstance(tp_rank,int) and tp_rank>0 else 'caller-verified rank list'}
                    if tp_rank == 0: errors.append({'kind':'nonleader_evidence_conflict','tp_rank':tp_rank})
                else: errors.append({'kind':'loader_missing_unverified_owner','tp_rank':tp_rank})
            elif saved_loader:
                saved_c={k:v for k,v in leaves(saved_loader) if k.split('.')[-1]=='consumed_samples'}
                actual_c={k:v for k,v in leaves(actual_loader) if k.split('.')[-1]=='consumed_samples'}
                if not saved_c or not actual_c or set(saved_c.values())!=set(actual_c.values()):
                    errors.append({'kind':'loader_cursor','saved':saved_c,'actual':actual_c})
                else: rr['loader']={'status':'cursor_match','saved':saved_c,'actual':actual_c}
            else: rr['unchecked'].append({'kind':'saved_loader_empty'})
            saved_rng=extra.get('rng_state') or {}
            if args.source_rng_not_saved:
                rr['rng_provenance']='HP runtime skeleton/default; source checkpoint did not save compatible RNG; no MindSpore RNG restoration claimed'
            for src,dst in [('torch_cpu','cpu'),('torch_device','device')]:
                if src not in saved_rng or saved_rng[src] is None:
                    rr['unchecked'].append({'kind':'rng_not_saved','key':src}); continue
                expected=saved_rng[src]
                if not isinstance(expected,dict) or expected.get('sha256')!=live.get('rng',{}).get(dst):
                    errors.append({'kind':'rng_hash','key':src})
                else: rr.setdefault('rng_match',[]).append(src)
            if saved_rng.get('python') is not None:
                if 'python' not in live.get('rng',{}):
                    rr['unchecked'].append({'kind':'python_rng_not_observed_in_original'})
                elif saved_rng['python'] != live['rng']['python']:
                    errors.append({'kind':'rng_python_mismatch'})
                else: rr.setdefault('rng_match',[]).append('python')
            else: rr['unchecked'].append({'kind':'rng_not_saved','key':'python'})
        except Exception as exc:
            errors.append({'kind':'exception','type':type(exc).__name__,'message':str(exc)})
        if errors: result['errors'].append({'rank':rank,'count':len(errors)})
    result['status']='FAIL' if result['errors'] else ('PASS_WITH_UNCHECKED' if any(r['unchecked'] for r in result['ranks']) else 'PASS')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'errors':len(result['errors']),'output':str(out)}))
    return 1 if result['errors'] else 0

if __name__=='__main__': raise SystemExit(main())
PY
cat > "$RUN_ROOT/scripts/mf_checkpoint_closure.py" <<'PY'
"""CPU-only closure audit for a native MF iteration; never changes source files."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

from safetensors import safe_open
from tools.convert_mf_checkpoint import MindFormersCheckpoint


def stat(path):
    x = path.stat()
    return {'device': x.st_dev, 'inode': x.st_ino, 'size': x.st_size, 'mtime_ns': x.st_mtime_ns}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--log', type=Path, required=True, help='native trainer log containing save-return message')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-step', type=int, default=1500)
    parser.add_argument('--expected-samples', type=int, default=3000)
    args = parser.parse_args()
    src = args.src.resolve(strict=True)
    output = args.output.resolve()
    if src == output or src in output.parents:
        parser.error('output must be outside source checkpoint directory')
    if args.expected_step <= 0 or args.expected_samples < 0:
        parser.error('invalid expected step or sample count')
    if output.exists():
        parser.error('output already exists')
    result = {'status': 'failed', 'source': str(src), 'expected_step': args.expected_step,
              'expected_samples': args.expected_samples, 'errors': [],
              'writer_fd_check': 'not performed; host-side read-only check required', 'files': [], 'headers': {}}
    try:
        if not src.is_dir():
            raise ValueError('source is not a directory')
        files = sorted(p for p in src.rglob('*') if p.is_file())
        if not files or not (src / 'common.json').is_file():
            raise ValueError('missing checkpoint files or common.json')
        before = {str(p.relative_to(src)): stat(p) for p in files}
        ledger_hash = hashlib.sha256()
        ledger_rows = 0
        with args.ledger.open('rb') as ledger:
            for raw in ledger:
                row = json.loads(raw)
                ledger_rows += 1
                if row.get('step') != ledger_rows:
                    raise ValueError(f'ledger missing/duplicate step {ledger_rows}')
                ledger_hash.update(raw)
                if ledger_rows == args.expected_step:
                    if int(row.get('consumed_samples', -1)) != args.expected_samples:
                        raise ValueError('ledger consumed_samples differs at target')
                    for key in ('loss', 'mtp_raw', 'aux_raw'):
                        val = row.get(key)
                        if not isinstance(val, (int, float)) or not math.isfinite(val):
                            raise ValueError(f'ledger invalid {key} at target')
                    break
        if ledger_rows != args.expected_step:
            raise ValueError(f'ledger ends at {ledger_rows}, before target')
        result['ledger'] = {'path': str(args.ledger), 'verified_contiguous_rows': ledger_rows,
                            'prefix_sha256': ledger_hash.hexdigest(), 'last_step': ledger_rows}
        pattern = re.compile(r'Checkpoint saved at step\s+' + str(args.expected_step) + r'\b')
        with args.log.open('r', encoding='utf-8', errors='replace') as log:
            matches = [line.rstrip() for line in log if pattern.search(line)]
        result['save_log'] = {'path': str(args.log), 'matching_lines': matches}
        if not matches:
            raise ValueError('native save-return log not found')
        source = MindFormersCheckpoint(src)
        result['common'] = source.common
        if int(source.common.get('global_step', -1)) != args.expected_step:
            raise ValueError('common.global_step differs')
        if int(source.common.get('consumed_samples', -1)) != args.expected_samples:
            raise ValueError('common.consumed_samples differs')
        names = sorted(source.keys())
        if 'global_step' not in names:
            raise ValueError('optimizer global_step tensor missing')
        value = source.read('global_step')
        if value.numel() != 1 or int(value.item()) != args.expected_step:
            raise ValueError('optimizer tensor global_step differs')
        result['optimizer_global_step'] = int(value.item())
        result['tensor_names'] = {'all': names, 'model': [], 'master': [], 'muon_m': [],
                                  'adam_m': [], 'adam_v': [], 'max_exp_avg_sq': []}
        prefixes = {'master': 'main_param.', 'muon_m': 'muon_m.', 'adam_m': 'adam_m.',
                    'adam_v': 'adam_v.', 'max_exp_avg_sq': 'max_exp_avg_sq.'}
        for name in names:
            if name == 'global_step':
                continue
            group = next((key for key, prefix in prefixes.items() if name.startswith(prefix)), 'model')
            result['tensor_names'][group].append(name)
        result['tensor_counts'] = {key: len(value) for key, value in result['tensor_names'].items()}
        if not result['tensor_names']['model'] or not any(result['tensor_names'][k] for k in ('muon_m', 'adam_m', 'adam_v')):
            raise ValueError('model or optimizer slots not found')
        for path in files:
            rel = str(path.relative_to(src))
            entry = {'path': rel, **before[rel], 'sha256': digest(path)}
            if path.suffix == '.safetensors':
                with safe_open(path, framework='pt', device='cpu') as reader:
                    result['headers'][rel] = {name: {'shape': list(reader.get_slice(name).get_shape()),
                                                     'dtype': reader.get_slice(name).get_dtype()}
                                              for name in sorted(reader.keys())}
            result['files'].append(entry)
        after_files = sorted(p for p in src.rglob('*') if p.is_file())
        after = {str(p.relative_to(src)): stat(p) for p in after_files}
        if before != after:
            raise ValueError('checkpoint file set or stat changed during audit')
        result['status'] = 'passed'
    except Exception as exc:  # Persist a diagnostic even when a gate fails.
        result['errors'].append(f'{type(exc).__name__}: {exc}')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent,
                                     prefix=f'.{output.name}.', delete=False) as f:
        temp = Path(f.name)
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, output)
    print(json.dumps({'status': result['status'], 'errors': result['errors'], 'output': str(output)}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
PY
cat > "$RUN_ROOT/scripts/audit_mf_to_hp_independent.py" <<'PY'
#!/usr/bin/env python3
"""Read-only, single-rank MF-to-HP checkpoint tensor audit.

Usage: python audit_conversion.py [--source DIR] [--target DIR] [--report FILE]
Assumes unsharded source and unsharded logical target tensors. Fails closed on
unknown mappings, duplicates, chunks and unaccounted source/target tensors.
Only a tensor and its comparison blocks are materialized at a time.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from hyper_parallel.core.distributed_checkpoint.filesystem_storage import FileSystemReader
from hyper_parallel.core.distributed_checkpoint.standard_planner import StandardLoadPlanner
from hyper_parallel.core.distributed_checkpoint.metadata import TensorStorageMetadata
from hyper_parallel.core.distributed_checkpoint.utils import str_to_dtype

SLOTS = {'momentum_buffer': 'muon_m.', 'exp_avg': 'adam_m.',
         'exp_avg_sq': 'adam_v.', 'max_exp_avg_sq': 'max_exp_avg_sq.'}
MASTER = 'optimizer._mixed_precision_optimizer.fp32_from_fp16_params.'
FIELDS = {
    'input_layernorm.weight': 'input_layernorm.weight',
    'post_attention_layernorm.weight': 'pre_mlp_layernorm.weight',
    'self_attn.kv_a_layernorm.weight': 'self_attention.k_layernorm.weight',
    'self_attn.q_a_layernorm.weight': 'self_attention.q_layernorm.weight',
    'self_attn.linear_qkv.weight': 'self_attention.linear_qkv.weight',
    'self_attn.q_b_proj.weight': 'self_attention.linear_qb.weight',
    'self_attn.kv_b_proj.weight': 'self_attention.linear_kvb.weight',
    'self_attn.o_proj.weight': 'self_attention.linear_proj.weight',
    'mlp.gate.weight': 'mlp.router.weight',
    'mlp.gate.tid2eid': 'mlp.router.tid2eid',
    'mlp.gate.e_score_correction_bias': 'mlp.expert_bias',
    'mlp.gate._token_counts': 'mlp.tokens_per_expert',
    'mlp.experts.gate_up_proj': 'mlp.experts.weight1',
    'mlp.experts.down_proj': 'mlp.experts.weight2',
}


def index_files(directory):
    files = sorted(directory.glob('*.safetensors'))
    if not files:
        raise ValueError(f'No safetensors files in {directory}')
    index, duplicates = {}, []
    for path in files:
        with safe_open(str(path), framework='pt', device='cpu') as f:
            for key in f.keys():
                if key in index:
                    duplicates.append(key)
                index[key] = path
    return index, duplicates


def reference(name):
    """Map an HP model FQN to MF name(s) and a coordinate operation."""
    top = {'model.embed_tokens.weight': 'embedding.word_embeddings.weight',
           'model.norm.weight': 'decoder.final_layernorm.weight',
           'lm_head.weight': 'output_layer.weight'}
    if name in top:
        return [top[name]], 'identity'
    mtp = re.fullmatch(r'mtp\.layers\.(\d+)\.(eh_proj|enorm|hnorm|post_norm)\.weight', name)
    if mtp:
        field = 'final_layernorm' if mtp[2] == 'post_norm' else mtp[2]
        return [f'mtp.layers.{mtp[1]}.{field}.weight'], 'identity'
    layer = re.fullmatch(r'model\.layers\.(\d+)\.(.+)', name)
    if layer:
        base, field = f'decoder.layers.{layer[1]}', layer[2]
    else:
        layer = re.fullmatch(r'mtp\.layers\.(\d+)\.mtp_block\.(.+)', name)
        if not layer:
            raise ValueError(f'No reference mapping for {name}')
        base, field = f'mtp.layers.{layer[1]}.transformer_layer', layer[2]
    if field in FIELDS:
        return [f'{base}.{FIELDS[field]}'], 'identity'
    dense = re.fullmatch(r'(mlp(?:\.shared_experts)?)\.(linear_fc1|gate_proj|up_proj|linear_fc2|down_proj)\.weight', field)
    if dense:
        part, kind = dense.groups()
        old = 'linear_fc2' if kind in ('linear_fc2', 'down_proj') else 'linear_fc1'
        op = {'linear_fc1': 'gate_up', 'gate_proj': 'gate', 'up_proj': 'up',
              'linear_fc2': 'identity', 'down_proj': 'identity'}[kind]
        return [f'{base}.{part}.{old}.weight'], op
    hc = re.fullmatch(r'(attn_hc|ffn_hc)\.(hc_base|hc_fn|hc_scale)', field)
    if hc:
        part, kind = hc.groups()
        prefix = f'{base}.{part}'
        if kind == 'hc_base':
            return [f'{prefix}.bias'], 'identity'
        if kind == 'hc_fn':
            return [f'{prefix}.mapping_proj.weight'], 'identity'
        return [f'{prefix}.alpha_{suffix}' for suffix in ('pre', 'post', 'res')], 'concat'
    raise ValueError(f'No reference mapping for {name}')


def read(index, key):
    with safe_open(str(index[key]), framework='pt', device='cpu') as f:
        return f.get_tensor(key)


def summary(value, depth=0):
    if isinstance(value, torch.Tensor):
        if value.numel() <= 8:
            return {'shape': list(value.shape), 'dtype': str(value.dtype), 'value': value.tolist()}
        return {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): summary(v, depth + 1) for k, v in value.items()} if depth < 8 else {'keys': list(map(str, value))}
    if isinstance(value, (list, tuple)):
        return [summary(v, depth + 1) for v in value] if len(value) <= 20 and depth < 8 else {'type': type(value).__name__, 'length': len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {'type': type(value).__name__}


def compare(actual, expected):
    if tuple(actual.shape) != tuple(expected.shape):
        return False, None
    # Conversion intentionally casts to the target dtype. Work in bounded blocks.
    a, b = actual.reshape(-1), expected.reshape(-1)
    equal, maxdiff = True, 0.0
    for i in range(0, a.numel(), 1024 * 1024):
        x = a[i:i + 1024 * 1024]
        y = b[i:i + 1024 * 1024].to(dtype=actual.dtype)
        if not torch.equal(x, y):
            equal = False
        if x.numel():
            d = (x.to(torch.float64) - y.to(torch.float64)).abs()
            if not torch.isfinite(d).all().item():
                maxdiff = None
            elif maxdiff is not None:
                maxdiff = max(maxdiff, d.max().item())
    return equal, maxdiff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--report', type=Path, default=Path('audit_conversion.json'))
    args = parser.parse_args()
    source, source_dup = index_files(args.source)
    reader = FileSystemReader(args.target)
    metadata = reader.load_metadata()
    target = {key: value for key, value in metadata.state_dict_metadata.items()
              if isinstance(value, TensorStorageMetadata) and key.startswith(('model.', 'optimizer.'))}
    target_dup = []
    reader.configure_reader(metadata, is_coordinator=True, rank=0)

    def read_target(key):
        meta = target[key]
        out = {key: torch.empty(tuple(meta.size), dtype=str_to_dtype(meta.properties.dtype), device='cpu')}
        planner = StandardLoadPlanner(allow_partial_load=False)
        planner.configure_planner(out, metadata, is_coordinator=True, rank=0,
                                  flatten_state_dict=False, use_collectives=False,
                                  broadcast_replicated_tensors=False)
        plan = reader.optimize_local_plan(planner.build_local_plan())
        plan = reader.optimize_global_plan(planner.build_global_plan([plan]))
        reader.execute_read(planner.finalize_plan(plan), planner)
        return out[key]
    result = {'source': str(args.source), 'target': str(args.target), 'tensors': [],
              'duplicates': {'source': source_dup, 'target': target_dup}, 'errors': [], 'extra_state': {}}
    used = set()
    for key in sorted(target):
        record = {'target': key}
        try:
            prefix = ''
            if key.startswith('model.'):
                model_key = key[len('model.'):]
                kind = 'model'
            elif key.startswith(MASTER):
                model_key = key[len(MASTER):]
                prefix, kind = 'main_param.', 'master'
            elif key.startswith('optimizer.state.'):
                model_key, sep, slot = key[len('optimizer.state.'):].rpartition('.')
                if not sep or slot not in SLOTS:
                    raise ValueError(f'Unknown optimizer slot: {key}')
                prefix, kind = SLOTS[slot], 'slot'
            else:
                raise ValueError(f'Unknown target tensor: {key}')
            names, op = reference(model_key)
            actual = read_target(key)
            # For FP32 trainable target weights, the converter selects the source
            # FP32 master. BF16 target model weights use the compute copy.
            if kind == 'model' and actual.dtype == torch.float32:
                names = [('main_param.' + n) if 'main_param.' + n in source else n for n in names]
            else:
                names = [prefix + n for n in names]
            missing = [n for n in names if n not in source]
            if missing:
                raise ValueError(f'Missing source tensors: {missing}')
            used.update(names)
            inputs = [read(source, n) for n in names]
            if op == 'concat':
                expected = torch.cat([value.reshape(-1) for value in inputs], dim=0)
            elif op == 'gate':
                expected = inputs[0][0::2]
            elif op == 'up':
                expected = inputs[0][1::2]
            elif op == 'gate_up':
                expected = torch.cat((inputs[0][0::2], inputs[0][1::2]), dim=0)
            else:
                expected = inputs[0]
            equal, diff = compare(actual, expected)
            record.update({'source': names, 'operation': op, 'source_shapes': [list(t.shape) for t in inputs],
                           'source_dtypes': [str(t.dtype) for t in inputs], 'target_shape': list(actual.shape),
                           'target_dtype': str(actual.dtype), 'equal': equal, 'max_abs_diff': diff})
            if not equal:
                result['errors'].append(f'Mismatch: {key}')
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            result['errors'].append(f'{key}: {record["error"]}')
        result['tensors'].append(record)
    # mHC rms weights are omitted only for the fused unit-normalization mapping.
    # Source compute copies replaced by FP32 master are retained as shadows and
    # reported separately: training can save a rounded copy, not a second HP tensor.
    result['source_omissions'] = []
    for key in sorted(set(source) - used):
        item = {'source': key}
        if re.fullmatch(r'(?:decoder\.layers\.\d+|mtp\.layers\.\d+\.transformer_layer)\.(?:attn_hc|ffn_hc)\.rms_weight', key):
            tensor = read(source, key)
            unit = bool(torch.all(tensor == 1).item())
            item.update({'reason': 'omitted fused mHC unit RMS weight', 'shape': list(tensor.shape),
                         'dtype': str(tensor.dtype), 'equal_to_one': unit})
            if not unit:
                result['errors'].append(f'Non-unit mHC RMS weight: {key}')
        elif key == 'global_step':
            tensor = read(source, key)
            item.update({'reason': 'optimizer/scheduler counter', 'value': summary(tensor)})
        elif 'main_param.' + key in used:
            compute = read(source, key)
            master = read(source, 'main_param.' + key)
            same, diff = compare(compute, master)
            item.update({'reason': 'MF compute copy superseded by FP32 master in FP32 target model',
                         'shape': list(compute.shape), 'dtype': str(compute.dtype),
                         'master_cast_equal': same, 'master_cast_max_abs_diff': diff,
                         'note': 'Informational only: the FP32 master and compute copy can differ.'})
        else:
            item['reason'] = 'unaccounted source training tensor'
            result['errors'].append(f'Unaccounted source tensor: {key}')
        result['source_omissions'].append(item)
    result['extra_state'] = {'status': 'not checked; separate independent state gate required'}
    result['other_tensor_metadata_keys'] = sorted(k for k, v in metadata.state_dict_metadata.items()
                                                if isinstance(v, TensorStorageMetadata) and k not in target)
    result['non_tensor_metadata_keys'] = sorted(k for k, v in metadata.state_dict_metadata.items()
                                                if not isinstance(v, TensorStorageMetadata))
    result['scope'] = 'independent tensor mapping and values only; no scheduler, loader, live aliases or training'
    result['counts'] = {'source': len(source), 'target': len(target), 'checked': len(result['tensors']),
                        'errors': len(result['errors']) + len(source_dup) + len(target_dup)}
    result['passed'] = result['counts']['errors'] == 0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'counts': result['counts'], 'report': str(args.report)}, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
PY
```
</details>
