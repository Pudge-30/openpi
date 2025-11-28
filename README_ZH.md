# openpi

openpi 包含由 [Physical Intelligence team](https://www.physicalintelligence.company/) 发布的用于机器人的开源模型和软件包。

目前，此仓库包含三种类型的模型：
- [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)，一种基于流的视觉-语言-动作模型 (VLA)。
- [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast)，一种基于 FAST 动作分词器的自回归 VLA。
- [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)，π₀ 的升级版本，通过[知识隔离](https://www.physicalintelligence.company/research/knowledge_insulation)训练，具有更好的开放世界泛化能力。请注意，在此仓库中，我们目前仅支持流匹配头 (flow matching head) 用于 $\pi_{0.5}$ 的训练和推理。

对于所有模型，我们提供_基础模型_检查点（在 10k+ 小时的机器人数据上预训练），以及开箱即用或在您自己的数据集上微调它们的示例。

这是一项实验：$\pi_0$ 是为我们自己的机器人开发的，这些机器人不同于广泛使用的平台（如 [ALOHA](https://tonyzhaozh.github.io/aloha/) 和 [DROID](https://droid-dataset.github.io/)）。虽然我们乐观地认为研究人员和从业者能够进行创造性的新实验，将 $\pi_0$ 改编到他们自己的平台上，但我们并不期望每一次尝试都能成功。总而言之：$\pi_0$ 可能适合您，也可能不适合，但欢迎您尝试看看！

## 更新

- [2025年9月] 我们在 openpi 中发布了 PyTorch 支持。
- [2025年9月] 我们发布了 pi05，这是 pi0 的升级版，具有更好的开放世界泛化能力。
- [2025年9月]: 我们为 DROID 训练添加了[改进的空闲过滤器](examples/droid/README_train.md#data-filtering)。
- [2025年6月]: 我们添加了[说明](examples/droid/README_train.md)，用于使用 `openpi` 在完整的 [DROID 数据集](https://droid-dataset.github.io/)上训练 VLA。这是用于训练 pi0-FAST-DROID 的训练流程的近似开源实现。

## 要求

要运行此仓库中的模型，您需要至少具有以下规格的 NVIDIA GPU。这些估算假设使用单个 GPU，但您也可以通过在训练配置中配置 `fsdp_devices` 来使用具有模型并行性的多个 GPU，以降低每个 GPU 的内存需求。另请注意，当前的训练脚本尚不支持多节点训练。

| 模式               | 所需内存        | 示例 GPU           |
| ------------------ | --------------- | ------------------ |
| 推理 (Inference)   | > 8 GB          | RTX 4090           |
| 微调 (LoRA)        | > 22.5 GB       | RTX 4090           |
| 微调 (Full)        | > 70 GB         | A100 (80GB) / H100 |

该仓库已在 Ubuntu 22.04 上测试过，我们目前不支持其他操作系统。

## 安装

克隆此仓库时，请确更新子模块：

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 或者如果您已经克隆了仓库：
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/) 来管理 Python 依赖项。请参阅 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/) 进行设置。安装 uv 后，运行以下命令来设置环境：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意：需要 `GIT_LFS_SKIP_SMUDGE=1` 以拉取 LeRobot 作为依赖项。

**Docker**: 作为 uv 安装的替代方案，我们提供了使用 Docker 安装 openpi 的说明。如果您在系统设置中遇到问题，请考虑使用 Docker 来简化安装。有关更多详细信息，请参阅 [Docker 设置](docs/docker.md)。

## 模型检查点

### 基础模型
我们提供多个基础 VLA 模型检查点。这些检查点已在 10k+ 小时的机器人数据上进行了预训练，可用于微调。

| 模型         | 用例        | 描述                                                                                                        | 检查点路径                                     |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | 微调        | 用于微调的基础 [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)                                 | `gs://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | 微调        | 用于微调的基础自回归 [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast)                 | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| $\pi_{0.5}$  | 微调        | 用于微调的基础 [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)                              | `gs://openpi-assets/checkpoints/pi05_base`     |

### 微调模型
我们还为各种机器人平台和任务提供“专家”检查点。这些模型是从上述基础模型微调而来的，旨在直接在目标机器人上运行。这些模型可能适用于也可能不适用于您的特定机器人。由于这些检查点是在相对较小的数据集上微调的（这些数据集是用更广泛可用的机器人收集的，例如 ALOHA 和 DROID Franka 设置），它们可能无法泛化到您的特定设置，尽管我们在实践中发现其中一些（特别是 DROID 检查点）具有相当广泛的泛化能力。

| 模型                     | 用例              | 描述                                                                                                                                                                                              | 检查点路径                                            |
| ------------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | 推理              | 在 [DROID 数据集](https://droid-dataset.github.io/)上微调的 $\pi_0$-FAST 模型：可以在 DROID 机器人平台上对新场景中的各种简单桌面操作任务进行 0-shot 执行                                          | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | 微调              | 在 [DROID 数据集](https://droid-dataset.github.io/)上微调的 $\pi_0$ 模型：推理速度比 $\pi_0$-FAST-DROID 快，但可能无法很好地遵循语言指令                                                          | `gs://openpi-assets/checkpoints/pi0_droid`            |
| $\pi_0$-ALOHA-towel      | 推理              | 在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调的 $\pi_0$ 模型：可以在 ALOHA 机器人平台上 0-shot 折叠各种毛巾                                                                      | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | 推理              | 在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调的 $\pi_0$ 模型：可以从特百惠容器中取出食物                                                                                         | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | 推理              | 在公共 [ALOHA](https://dit-policy.github.io/) 数据上微调的 $\pi_0$ 模型：可以拔掉笔帽                                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| $\pi_{0.5}$-LIBERO       | 推理              | 针对 [LIBERO](https://libero-project.github.io/datasets) 基准微调的 $\pi_{0.5}$ 模型：获得最先进的性能（参见 [LIBERO README](examples/libero/README.md)）                                         | `gs://openpi-assets/checkpoints/pi05_libero`          |
| $\pi_{0.5}$-DROID        | 推理 / 微调       | 在 [DROID 数据集](https://droid-dataset.github.io/)上通过[知识隔离](https://www.physicalintelligence.company/research/knowledge_insulation)微调的 $\pi_{0.5}$ 模型：推理速度快且语言遵循能力良好 | `gs://openpi-assets/checkpoints/pi05_droid`           |


默认情况下，检查点会自动从 `gs://openpi-assets` 下载，并在需要时缓存在 `~/.cache/openpi` 中。您可以通过设置 `OPENPI_DATA_HOME` 环境变量来覆盖下载路径。

## 运行预训练模型的推理

我们的预训练模型检查点可以通过几行代码运行（这里以我们的 $\pi_0$-FAST-DROID 模型为例）：
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 创建已训练的策略
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 在虚拟示例上运行推理
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
您也可以在[示例笔记本](examples/inference.ipynb)中进行测试。

我们提供了详细的步骤示例，用于在 [DROID](examples/droid/README.md) 和 [ALOHA](examples/aloha_real/README.md) 机器人上运行我们预训练检查点的推理。

**远程推理**：我们提供了**远程**运行我们模型推理的[示例和代码](docs/remote_inference.md)：模型可以在不同的服务器上运行，并通过 websocket 连接将动作流式传输到机器人。这使得在机器人之外使用更强大的 GPU 以及保持机器人和策略环境分离变得容易。

**无机器人测试推理**：我们提供了一个[脚本](examples/simple_client/README.md)用于在没有机器人的情况下测试推理。此脚本将生成随机观察结果并使用模型运行推理。有关更多详细信息，请参阅[此处](examples/simple_client/README.md)。

## 在您自己的数据上微调基础模型

我们将以在 [LIBERO 数据集](https://libero-project.github.io/datasets)上微调 $\pi_{0.5}$ 模型为例，说明如何在您自己的数据上微调基础模型。我们将解释三个步骤：
1. 将您的数据转换为 LeRobot 数据集（我们用于训练的数据集格式）
2. 定义训练配置并运行训练
3. 启动策略服务器并运行推理

### 1. 将您的数据转换为 LeRobot 数据集

我们在 [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py) 中提供了一个将 LIBERO 数据转换为 LeRobot 数据集的最小示例脚本。您可以轻松修改它以转换您自己的数据！您可以从[此处](https://huggingface.co/datasets/openvla/modified_libero_rlds)下载原始 LIBERO 数据集，并使用以下命令运行脚本：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**注意：** 如果您只想在 LIBERO 上进行微调，可以跳过此步骤，因为我们的 LIBERO 微调配置指向已预转换的 LIBERO 数据集。此步骤仅作为一个示例，您可以将其调整为适用于您自己的数据。

### 2. 定义训练配置并运行训练

要在您自己的数据上微调基础模型，您需要定义数据处理和训练的配置。我们在下面提供了带有详细注释的 LIBERO 示例配置，您可以根据自己的数据集进行修改：

- [`LiberoInputs` 和 `LiberoOutputs`](src/openpi/policies/libero_policy.py)：定义从 LIBERO 环境到模型的数据映射，反之亦然。将用于训练和推理。
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py)：定义如何处理来自 LeRobot 数据集的原始 LIBERO 数据以进行训练。
- [`TrainConfig`](src/openpi/training/config.py)：定义微调超参数、数据配置和权重加载器。

我们提供了在 LIBERO 数据上微调 [π₀](src/openpi/training/config.py)、[π₀-FAST](src/openpi/training/config.py) 和 [π₀.₅](src/openpi/training/config.py) 的示例配置。

在运行训练之前，我们需要计算训练数据的归一化统计信息。使用您的训练配置名称运行以下脚本：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

现在我们可以使用以下命令开始训练（如果在具有相同配置的情况下重新运行微调，则使用 `--overwrite` 标志覆盖现有检查点）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

该命令将把训练进度记录到控制台，并将检查点保存到 `checkpoints` 目录。您也可以在 Weights & Biases 仪表板上监控训练进度。为了最大限度地利用 GPU 内存，请在运行训练之前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` —— 这使 JAX 能够使用高达 90% 的 GPU 内存（默认值为 75%）。

**注意：** 我们提供了从预训练中*重新加载*状态/动作归一化的归一化统计信息的功能。如果您正在微调机器人上的新任务，而该机器人是我们预训练混合数据的一部分，这将非常有益。有关如何重新加载归一化统计信息的更多详细信息，请参阅 [norm_stats.md](docs/norm_stats.md) 文件。

### 3. 启动策略服务器并运行推理

训练完成后，我们可以通过启动策略服务器，然后从 LIBERO 评估脚本查询它来运行推理。启动模型服务器很容易（在此示例中，我们使用迭代 20,000 的检查点，请根据需要进行修改）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

这将启动一个服务器，监听 8000 端口并等待发送给它的观察结果。然后我们可以运行查询服务器的评估脚本（或机器人运行时）。

特别是对于运行 LIBERO 评估，我们提供（并建议使用）一个 Docker 化的工作流，它可以同时处理策略服务器和评估脚本。有关更多详细信息，请参阅 [LIBERO README](examples/libero/README.md)。

如果您想将策略服务器调用嵌入到您自己的机器人运行时中，我们在[远程推理文档](docs/remote_inference.md)中提供了一个关于如何执行此操作的最小示例。

### 更多示例

我们在以下 README 中提供了更多关于如何在 ALOHA 平台上微调和运行我们的模型推理的示例：
- [ALOHA Simulator](examples/aloha_sim)
- [ALOHA Real](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch 支持

openpi 现在提供 π₀ 和 π₀.₅ 模型的 PyTorch 实现以及原始的 JAX 版本！PyTorch 实现已在 LIBERO 基准测试（推理和微调）上得到验证。目前不支持一些功能（将来可能会更改）：

- π₀-FAST 模型
- 混合精度训练
- FSDP (全分片数据并行) 训练
- LoRA (低秩适应) 训练
- 训练期间的 EMA (指数移动平均) 权重

### 设置
1. 确保您已安装所有依赖项的最新版本：`uv sync`

2. 再次检查您是否安装了 transformers 4.53.2：`uv pip show transformers`

3. 应用 transformers 库补丁：
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

这将覆盖 transformers 库中的几个文件以进行必要的模型更改：1) 支持 AdaRMS，2) 正确控制激活的精度，以及 3) 允许在不更新的情况下使用 KV 缓存。

**警告**：在默认的 uv 链接模式（硬链接）下，这将永久影响 uv 缓存中的 transformers 库，这意味着这些更改将在重新安装 transformers 后仍然存在，甚至可能传播到使用 transformers 的其他项目。要完全撤消此操作，您必须运行 `uv cache clean transformers`。

### 将 JAX 模型转换为 PyTorch

要将 JAX 模型检查点转换为 PyTorch 格式：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 运行推理

PyTorch 实现使用与 JAX 版本相同的 API - 您只需更改检查点路径以指向转换后的 PyTorch 模型：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# 创建已训练的策略（自动检测 PyTorch 格式）
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 运行推理（与 JAX API 相同）
action_chunk = policy.infer(example)["actions"]
```

### 使用 PyTorch 的策略服务器

策略服务器与 PyTorch 模型的工作方式完全相同 - 只需指向转换后的检查点目录：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 进行微调

要在 PyTorch 中进行微调：

1. 将 JAX 基础模型转换为 PyTorch 格式：
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. 在您的配置中使用 `pytorch_weight_path` 指定转换后的 PyTorch 模型路径

3. 使用以下模式之一启动训练：

```bash
# 单 GPU 训练：
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# 示例：
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # 从最新检查点恢复

# 多 GPU 训练（单节点）：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# 示例：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# 多节点训练：
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### 精度设置

JAX 和 PyTorch 实现处理精度的方式如下：

**JAX:**
1. 推理：大多数权重和计算使用 bfloat16，为了稳定性少数计算使用 float32。
2. 训练：默认为混合精度：权重和梯度为 float32，（大多数）激活和计算为 bfloat16。您可以通过在配置中将 `dtype` 设置为 float32 来更改为全 float32 训练。

**PyTorch:**
1. 推理：与 JAX 匹配 -- 大多数权重和计算使用 bfloat16，为了稳定性少数权重转换为 float32。
2. 训练：支持全 bfloat16（默认）或全 float32。您可以通过在配置中设置 `pytorch_training_precision` 来更改它。与 float32 相比，bfloat16 使用的内存更少，但损失可能更高。目前尚不支持混合精度。

使用 torch.compile 时，推理速度在 JAX 和 PyTorch 之间是相当的。

## 故障排除

我们将在此处收集常见问题及其解决方案。如果您遇到问题，请先检查此处。如果找不到解决方案，请在仓库中提交 issue（有关指南，请参阅[此处](CONTRIBUTING.md)）。

| 问题 | 解决方案 |
| --- | --- |
| `uv sync` 失败并出现依赖冲突 | 尝试删除虚拟环境目录（`rm -rf .venv`）并再次运行 `uv sync`。如果问题仍然存在，请检查您是否安装了最新版本的 `uv`（`uv self update`）。 |
| 训练耗尽 GPU 内存 | 确保在运行训练之前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`（或更高），以允许 JAX 使用更多 GPU 内存。您还可以使用 `--fsdp-devices <n>`，其中 `<n>` 是您的 GPU 数量，以启用[全分片数据并行](https://engineering.fb.com/2021/07/15/open-source/fsdp/)，这会减少内存使用但会降低训练速度（减慢程度取决于您的具体设置）。如果您仍然内存不足，您可能需要考虑禁用 EMA。 |
| 策略服务器连接错误 | 检查服务器是否正在运行并在预期的端口上监听。验证客户端和服务器之间的网络连接和防火墙设置。 |
| 训练时缺少归一化统计信息错误 | 在开始训练之前，使用您的配置名称运行 `scripts/compute_norm_stats.py`。 |
| 数据集下载失败 | 检查您的互联网连接。对于 HuggingFace 数据集，请确保您已登录（`huggingface-cli login`）。 |
| CUDA/GPU 错误 | 验证 NVIDIA 驱动程序是否已正确安装。对于 Docker，请确保已安装 nvidia-container-toolkit。检查 GPU 兼容性。您**不需要**在系统级别安装 CUDA 库 —— 它们将通过 uv 安装。如果您遇到 CUDA 问题，您甚至可能想要尝试*卸载*系统 CUDA 库，因为系统库有时会导致冲突。 |
| 运行示例时导入错误 | 确保您已使用 `uv sync` 安装了所有依赖项。某些示例可能在其 README 中列出了额外的要求。 |
| 动作维度不匹配 | 验证您的数据处理转换是否与机器人的预期输入/输出维度匹配。检查策略类中的动作空间定义。 |
| 训练损失发散 | 检查数据集 `norm_stats.json` 中的 `q01`、`q99` 和 `std` 值。某些很少使用的维度最终可能会有非常小的 `q01`、`q99` 或 `std` 值，导致归一化后的状态和动作非常大。您可以手动调整归一化统计信息作为一种解决方法。 |

