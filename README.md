
<div align="center"> 
<!-- 居中盒子-->

# Lightning-Hydra-Template

+ [创建简洁、一致、易读的徽章：concise, consistent, legible badges](https://shields.io/)
- 构建状态（动态badge）、版本信息、下载次数、许可证、测试覆盖率
+ [管理和维护多语言的钩子hooks: 代码提交之前执行的预定义的检查等任务（自动化脚本或操作）]((https://github.com/pre-commit/pre-commit))
[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)

[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)

[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
+ [强制性的代码格式化工具 "Black"](https://black.readthedocs.io/en/stable/) 
+ [ Python 代码文件中的导入语句进行排序，按字母顺序排列，同时自动将导入语句分组并按照它们的类型进行分类 "isort"](https://pycqa.github.io/isort/) 

[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/) <br>
+ `测试、代码质量、Codecov可以用来监视和报告测试覆盖率`<br>
[![tests](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/test.yml)
[![code-quality](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/ashleve/lightning-hydra-template/actions/workflows/code-quality-main.yaml)
[![codecov](https://codecov.io/gh/ashleve/lightning-hydra-template/branch/main/graph/badge.svg)](https://codecov.io/gh/ashleve/lightning-hydra-template) <br>

+ `证书、PR、贡献者`<br>
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/ashleve/lightning-hydra-template/pulls)
[![contributors](https://img.shields.io/github/contributors/ashleve/lightning-hydra-template.svg)](https://github.com/ashleve/lightning-hydra-template/graphs/contributors)

+ `整洁的、干净的模板，意味着模板本身的代码结构和布局是清晰的，没有多余的冗余内容。 `<br> 
A clean template to kickstart your deep learning project 🚀⚡🔥<br>
Click on [<kbd>Use this template</kbd>](https://github.com/ashleve/lightning-hydra-template/generate) to initialize new repository.

_Suggestions are always welcome!_

</div>

<br>

## 📌  Introduction
        
**Why you might want to use it:**

✅ Save on boilerplate <br>
Easily add new models, datasets, tasks, experiments, and train on different accelerators, like multi-GPU, TPU or SLURM clusters.
+ 减少样板代码
  - 样板代码通常是一些重复性或标准化的代码块，用于初始化、配置或处理常见任务。通过减少样板代码，开发者可以更快地启动项目并专注于核心功能的开发。
  - 轻松添加新模型、数据集、任务、实验 `（灵活性）`
  - 在不同的加速器上进行训练，如多GPU、TPU或SLURM集群 `兼容性`


✅ Education <br>
Thoroughly commented. You can use this repo as a learning resource.
+ 详细注释，可以将此存储库用作学习资源 `教育性`

✅ Reusability <br>
Collection of useful MLOps tools, configs, and code snippets. You can use this repo as a reference for various utilities.
+ 一组有用的MLOps工具、配置和代码片段
  - "MLOps tools"：指的是与机器学习运维相关的工具，这些工具用于管理和自动化机器学习模型的开发、部署和监控。
  - "configs"（配置）：这是配置文件或设置，用于配置和调整MLOps工具的行为。
  - "code snippets"（代码片段）：这是一小段可重用的代码，通常用于执行特定任务或实现特定功能。
+ 可以将此存储库用作各种工具的参考。

**Why you might not want to use it:**
+ 限制和注意事项

❌ Things break from time to time <br>
Lightning and Hydra are still evolving and integrate many libraries, which means sometimes things break. For the list of currently known problems visit [this page](https://github.com/ashleve/lightning-hydra-template/labels/bug).

+ 项目或模板中的某些功能偶尔可能会出现问题或`不稳定`
  - 由于"Lightning" 和 "Hydra"两个库仍在不断发展和演进，因此可能会有新功能添加或现有功能进行更改
  - 两个工具整合了许多其他库和组件，以增强其功能。
  - 这意味着，有时候，这些库和组件之间的兼容性可能会出现问题，导致项目或模板中的某些功能偶尔可能会出现问题或不稳定。由于不断的发展和整合，有时某些功能可能会出现故障或不兼容，导致问题出现。

❌ Not adjusted for data engineering <br>
Template is not really adjusted for building data pipelines that depend on each other. It's more efficient to use it for model prototyping on ready-to-use data.

+ 项目或模板并没有特别针对`数据工程方面`的需求进行优化或调整
  - 该模板`不太适用于构建相互依赖的数据管道`。数据工程通常涉及到处理、转换和连接数据，而该模板更适用于用于模型原型设计的场景，其中`使用现成的数据`更为高效。
  - 该模板更适合用于在准备好的数据上进行模型原型设计，而不是构建复杂的数据工程管道。
  - `230925记：不适用于数据预处理？？？`

❌ Overfitted to simple use case <br>
The configuration setup is built with simple lightning training in mind. You might need to put some effort to adjust it for different use cases, e.g. lightning fabric.

+ 该项目或模板在设计时主要考虑了简单的使用情况，导致可能对更复杂的用例不够灵活。
  - 配置设置是根据简单的 Lightning 训练场景而构建的，这意味着项目的设计更适用于较为基本的训练需求。
  - 如果你的使用情况较复杂，例如需要使用 `Lightning Fabric（可能是一种更高级的用例）`，那么你可能需要花一些额外的精力来对项目进行调整以满足你的需求。

❌ Might not support your workflow <br>
For example, you can't resume hydra-based multirun or hyperparameter search.

+ 该项目或模板可能无法完全支持你的工作流程，可能存在一些局限性。
  - 你可能无法在项目中恢复基于 Hydra 的多次运行（multirun）或超参数搜索，这可能限制了某些高级工作流程的支持。

> **Note**: _Keep in mind this is unofficial community project._
+ 是一个非官方社区项目
<br>

## Main Technologies
+ 主要使用的技术和库的信息

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.
-  PyTorch Lightning 是一个用于高性能人工智能研究的轻量级 PyTorch 包装器。PyTorch Lightning 旨在简化 PyTorch 代码的组织和管理，以便更轻松地进行深度学习研究。

[Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.
- Hydra 具有动态创建分层配置的能力，可以通过组合和使用配置文件以及命令行进行覆盖。这使得 Hydra 成为配置复杂应用程序的有力工具。

+ `PyTorch Lightning 用于组织 PyTorch 代码，Hydra 用于配置复杂应用程序。`这些库的使用有助于项目的开发和管理，提高了项目的可维护性和可配置性。
<br>

## Main Ideas
+ 主要理念: 高效且有组织的机器学习项目框架。这些理念有助于提高项目的可维护性、可扩展性和开发效率。
- [**Rapid Experimentation**](#your-superpowers): thanks to hydra command line superpowers
  + （快速实验）： Hydra 命令行的强大功能使得快速实验变得更加容易。
- [**Minimal Boilerplate**](#how-it-works): thanks to automating pipelines with config instantiation
  + （最少样板代码）：自动化管道与配置实例化
- [**Main Configs**](#main-config): allow you to specify default training configuration
  + （主配置）：允许指定默认的训练配置，以确保项目的一致性
- [**Experiment Configs**](#experiment-config): allow you to override chosen hyperparameters and version control experiments
  + （实验配置）：允许在不同的实验中覆盖选定的超参数，从而对实验进行版本控制和管理。
- [**Workflow**](#workflow): comes down to 4 simple steps
  + （工作流程）：将其简化为四个简单的步骤，以提高开发效率。
- [**Experiment Tracking**](#experiment-tracking): Tensorboard, W&B, Neptune, Comet, MLFlow and CSVLogger
  + `（实验跟踪）`：多种实验跟踪工具，包括 Tensorboard、W&B、Neptune、Comet、MLFlow 和 CSVLogger。
- [**Logs**](#logs): all logs (checkpoints, configs, etc.) are stored in a dynamically generated folder structure
  + `（日志）`：所有的日志，包括检查点、配置等，都存储在一个动态生成的文件夹结构中。
- [**Hyperparameter Search**](#hyperparameter-search): simple search is effortless with Hydra plugins like Optuna Sweeper
  + `（超参数搜索）`：使用 Hydra 插件如 Optuna Sweeper 等来简化搜索过程。
- [**Tests**](#tests): generic, easy-to-adapt smoke tests for speeding up the development
  + （测试）：是通用的、易于调整的冒烟测试[**冒烟测试:焦点是验证软件的核心或基本功能**]，可以加速开发过程。
- [**Continuous Integration**](#continuous-integration): automatically test and lint your repo with Github Actions
  + `（持续集成）`：  使用 Github Actions 自动测试和检查代码库，以确保代码的质量。
- [**Best Practices**](#best-practices): a couple of recommended tools, practices and standards
  + （最佳实践）：一些推荐的工具、实践和标准，以帮助开发者在项目中采用最佳实践。

<br>

## Project Structure

The directory structure of new project looks like this:
+ 代码、配置、数据、测试等
```
├── .github                   <- Github Actions workflows : Github Actions 的工作流配置文件。Github Actions 是一种自动化工具，用于在 Github 仓库中执行各种操作，例如自动构建、测试和部署。
│                            
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs 回调配置。(回调函数通常用于在模型训练过程中执行特定的操作或记录特定的事件。)
│   ├── data                     <- Data configs 数据配置 （数据处理和加载相关的配置文件。这些配置可能包括数据路径、数据预处理选项等。）
│   ├── debug                    <- Debugging configs 调试配置 （配置调试选项的文件。它们可能包括启用或禁用调试模式、记录调试信息等）
│   ├── experiment               <- Experiment configs 实验配置 （用于配置不同实验的参数和设置。每个实验可能有不同的配置。）
│   ├── extras                   <- Extra utilities configs （额外工具配置）：用于配置项目中使用的其他实用工具或组件。
│   ├── hparams_search           <- Hyperparameter search configs （超参数搜索配置）：用于配置超参数搜索的文件。超参数搜索是一种用于寻找模型最佳超参数组合的技术。
│   ├── hydra                    <- Hydra configs（Hydra 配置）：可能包含 Hydra 框架本身的配置文件，用于配置 Hydra 的行为和选项。
│   ├── local                    <- Local configs （本地配置）：包含了与本地开发和测试相关的配置文件。这些配置可能包括本地环境的设置和选项。
│   ├── logger                   <- Logger configs（日志配置）：包含了用于配置日志记录的文件。日志通常用于记录项目的运行信息、事件和结果。
│   ├── model                    <- Model configs 模型配置）：包含了用于配置机器学习模型的文件。这些配置可能包括模型结构、超参数等。
│   ├── paths                    <- Project paths configs （项目路径配置）：包含了用于配置项目中各种路径的文件，如数据路径、模型保存路径等。
│   ├── trainer                  <- Trainer configs （训练器配置）：包含了用于配置训练过程的文件。训练器配置可能包括训练的批量大小、学习率等设置。
│   │
│   ├── eval.yaml             <- Main config for evaluation （评估主配置）：用于评估（evaluation）的主配置文件。它包含了评估过程的主要参数和设置。
│   └── train.yaml            <- Main config for training （训练主配置）：用于训练（training）的主配置文件。它包含了训练过程的主要参数和设置。
│
├── data                   <- Project data :存储项目所需的数据
│
├── logs                   <- Logs generated by hydra and lightning loggers : 由 Hydra 和 Lightning Loggers 生成的日志文件。这些日志记录了项目的运行信息、结果和事件。
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.: 每个笔记本文件的命名采用一定的约定，包括序号、创建者的缩写和简短的描述，以方便组织和查找。
│
├── scripts                <- Shell scripts :Shell脚本文件，这些脚本可用于自动执行特定的任务或操作
│
├── src                    <- Source code 项目的源代码目录
│   ├── data                     <- Data scripts 数据处理脚本
│   ├── models                   <- Model scripts 模型脚本
│   ├── utils                    <- Utility scripts 工具脚本
│   │
│   ├── eval.py                  <- Run evaluation 运行评估
│   └── train.py                 <- Run training 运行训练
│
├── tests                  <- Tests of any kind 项目的测试文件，用于验证代码的正确性和功能性。
│
├── .env.example              <- Example of file for storing private environment variables :示例文件，用于存储私有环境变量的配置信息。通常，开发者可以使用这个示例文件创建自己的私有环境变量文件
├── .gitignore                <- List of files ignored by git :列出了 Git 忽略的文件和目录，防止它们被添加到版本控制中。
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting :配置 pre-commit 钩子的配置文件，用于自动化代码格式化。
├── .project-root             <- File for inferring the position of project root directory:用于标识项目根目录的位置，通常由工具自动生成。
├── environment.yaml          <- File for installing conda environment
├── Makefile                  <- Makefile with commands like `make train` or `make test`: 包含了一些命令，如 make train 或 make test，用于执行常见的项目任务
├── pyproject.toml            <- Configuration options for testing and linting: 包含了有关测试和代码检查的配置选项，通常与 Python 工具（如 poetry）一起使用。
├── requirements.txt          <- File for installing python dependencies
├── setup.py                  <- File for installing project as a package: 用于将项目安装为 Python 包
└── README.md
```
  
<br>

## 🚀  Quickstart

```bash
# clone project
git clone https://github.com/ashleve/lightning-hydra-template
cd lightning-hydra-template

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Template contains example with MNIST classification.<br>
When running `python src/train.py` you should see something like this:
+ MNIST 分类的示例
<div align="center">

![](https://github.com/ashleve/lightning-hydra-template/blob/resources/terminal.png)

</div>

## ⚡  Your Superpowers

<details>
<summary><b>Override any config parameter from command line从命令行覆盖任何配置参数</b></summary>

```bash
python train.py trainer.max_epochs=20 model.optimizer.lr=1e-4
```

> **Note**: You can also add new parameters with `+` sign.可使用`+` 符号来添加新参数

```bash
python train.py +model.new_param="owo"
```

</details>

<details>
<summary><b>Train on CPU, GPU, multi-GPU and TPU</b></summary>

```bash
# train on CPU
python train.py trainer=cpu

# train on 1 GPU
python train.py trainer=gpu

# train on TPU
python train.py +trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (4 GPUs)
# （使用 DDP（分布式数据并行）进行训练（4个GPU））
python train.py trainer=ddp trainer.devices=4

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
# （8个GPU，2个节点）
python train.py trainer=ddp trainer.devices=4 trainer.num_nodes=2

# simulate DDP on CPU processes （在CPU进程上模拟DDP训练）
python train.py trainer=ddp_sim trainer.devices=2

# accelerate training on mac（在Mac上加速训练）
python train.py trainer=mps
```

> **Warning**: Currently there are problems with DDP mode, read [this issue](https://github.com/ashleve/lightning-hydra-template/issues/393) to learn more.（目前 DDP 模式存在问题）：这句话说明在项目中使用 DDP 模式进行训练时，可能会遇到一些问题或 bug。

</details>

<details>
<summary><b>Train with mixed precision（使用混合精度进行训练）</b></summary>
+ 混合精度是一种训练技术，可以在保持训练效果的同时减少模型训练所需的显存
```bash
# train with pytorch native automatic mixed precision (AMP)
# （使用 PyTorch 原生的自动混合精度进行训练）
python train.py trainer=gpu +trainer.precision=16
```
+ 混合精度训练是一种用于减少深度学习模型训练所需显存的技术，通过使用较低位数的浮点数表示权重和梯度，可以在减少内存占用的同时保持训练的效果。在示例中，用户可以参考这个命令来了解如何配置项目以启用混合精度训练，以便在硬件资源有限的情况下提高模型的训练效率。
 
</details>

<!-- deepspeed support still in beta
# （DeepSpeed 支持仍处于测试阶段）
<details>
<summary><b>Optimize large scale models on multiple GPUs with Deepspeed</b></summary>
+ 使用 DeepSpeed 来优化在多个 GPU 上进行训练的大规模模型。
```bash
python train.py +trainer.
```

</details>
 -->

<details>
<summary><b>Train model with any logger available in PyTorch Lightning, like W&B or Tensorboard（使用PyTorch Lightning中的任何记录器（logger）来训练模型：记录和监视模型的训练过程）</b></summary>

```yaml
# set project and entity names in `configs/logger/wandb`
# 在configs/logger/wandb中设置项目和实体名称
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
# train model with Weights&Biases (link to wandb dashboard should appear in the terminal)（使用 Weights&Biases 记录器来训练模型，终端中应出现指向 W&B 仪表板的链接）
python train.py logger=wandb
```

> **Note**: Lightning provides convenient integrations with most popular logging frameworks. Learn more [here](#experiment-tracking).
（注意：Lightning 提供了与大多数流行的记录框架方便集成。了解更多信息在此处。）
> **Note**: Using wandb requires you to [setup account](https://www.wandb.com/) first. After that just complete the config as below.
（注意：使用 wandb 需要你先设置帐户。之后，只需按照下面的配置完成即可。）
> **Note**: Click [here](https://wandb.ai/hobglob/template-dashboard/) to see example wandb dashboard generated with this template.
（注意：点击这里查看使用此模板生成的示例 wandb 仪表板。）
</details>

<details>
<summary><b>Train model with chosen experiment config（使用已选择的实验配置来训练模型）</b></summary>

```bash
python train.py experiment=example
```

> **Note**: Experiment configs are placed in [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Attach some callbacks to run</b></summary>

```bash
python train.py callbacks=default
```

> **Note**: Callbacks can be used for things such as as model checkpointing, early stopping and [many more](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).

> **Note**: Callbacks configs are placed in [configs/callbacks/](configs/callbacks/).

</details>

<details>
<summary><b>Use different tricks available in Pytorch Lightning</b></summary>

```yaml
# gradient clipping may be enabled to avoid exploding gradients
python train.py +trainer.gradient_clip_val=0.5

# run validation loop 4 times during a training epoch
python train.py +trainer.val_check_interval=0.25

# accumulate gradients
python train.py +trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python train.py +trainer.max_time="00:12:00:00"
```

> **Note**: PyTorch Lightning provides about [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

</details>

<details>
<summary><b>Easily debug</b></summary>

```bash
# runs 1 epoch in default debugging mode
# changes logging directory to `logs/debugs/...`
# sets level of all command line loggers to 'DEBUG'
# enforces debug-friendly configuration
python train.py debug=default

# run 1 train, val and test loop, using only 1 batch
python train.py debug=fdr

# print execution time profiling
python train.py debug=profiler

# try overfitting to 1 batch
python train.py debug=overfit

# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python train.py +trainer.detect_anomaly=true

# use only 20% of the data
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2
```

> **Note**: Visit [configs/debug/](configs/debug/) for different debugging configs.

</details>

<details>
<summary><b>Resume training from checkpoint</b></summary>

```yaml
python train.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

> **Note**: Currently loading ckpt doesn't resume logger experiment, but it will be supported in future Lightning release.

</details>

<details>
<summary><b>Evaluate checkpoint on test dataset</b></summary>

```yaml
python eval.py ckpt_path="/path/to/ckpt/name.ckpt"
```

> **Note**: Checkpoint can be either path or URL.

</details>

<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m data.batch_size=32,64,128 model.lr=0.001,0.0005
```

> **Note**: Hydra composes configs lazily at job launch time. If you change code or configs after launching a job/sweep, the final composed configs might be impacted.

</details>

<details>
<summary><b>Create a sweep over hyperparameters with Optuna</b></summary>

```bash
# this will run hyperparameter search defined in `configs/hparams_search/mnist_optuna.yaml`
# over chosen experiment config
python train.py -m hparams_search=mnist_optuna experiment=example
```

> **Note**: Using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) doesn't require you to add any boilerplate to your code, everything is defined in a [single config file](configs/hparams_search/mnist_optuna.yaml).

> **Warning**: Optuna sweeps are not failure-resistant (if one job crashes then the whole sweep crashes).

</details>

<details>
<summary><b>Execute all experiments from folder</b></summary>

```bash
python train.py -m 'experiment=glob(*)'
```

> **Note**: Hydra provides special syntax for controlling behavior of multiruns. Learn more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run). The command above executes all experiments from [configs/experiment/](configs/experiment/).

</details>

<details>
<summary><b>Execute run for multiple different seeds</b></summary>

```bash
python train.py -m seed=1,2,3,4,5 trainer.deterministic=True logger=csv tags=["benchmark"]
```

> **Note**: `trainer.deterministic=True` makes pytorch more deterministic but impacts the performance.

</details>

<details>
<summary><b>Execute sweep on a remote AWS cluster</b></summary>

> **Note**: This should be achievable with simple config using [Ray AWS launcher for Hydra](https://hydra.cc/docs/next/plugins/ray_launcher). Example is not implemented in this template.

</details>

<!-- <details>
<summary><b>Execute sweep on a SLURM cluster</b></summary>

> This should be achievable with either [the right lightning trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html?highlight=SLURM#slurm-managed-cluster) or simple config using [Submitit launcher for Hydra](https://hydra.cc/docs/plugins/submitit_launcher). Example is not yet implemented in this template.

</details> -->

<details>
<summary><b>Use Hydra tab completion</b></summary>

> **Note**: Hydra allows you to autocomplete config argument overrides in shell as you write them, by pressing `tab` key. Read the [docs](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion).

</details>

<details>
<summary><b>Apply pre-commit hooks</b></summary>

```bash
pre-commit run -a
```

> **Note**: Apply pre-commit hooks to do things like auto-formatting code and configs, performing code analysis or removing output from jupyter notebooks. See [# Best Practices](#best-practices) for more.

Update pre-commit hook versions in `.pre-commit-config.yaml` with:

```bash
pre-commit autoupdate
```

</details>

<details>
<summary><b>Run tests</b></summary>

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```

</details>

<details>
<summary><b>Use tags</b></summary>

Each experiment should be tagged in order to easily filter them across files or in logger UI:

```bash
python train.py tags=["mnist","experiment_X"]
```

> **Note**: You might need to escape the bracket characters in your shell with `python train.py tags=\["mnist","experiment_X"\]`.

If no tags are provided, you will be asked to input them from command line:

```bash
>>> python train.py tags=[]
[2022-07-11 15:40:09,358][src.utils.utils][INFO] - Enforcing tags! <cfg.extras.enforce_tags=True>
[2022-07-11 15:40:09,359][src.utils.rich_utils][WARNING] - No tags provided in config. Prompting user to input tags...
Enter a list of comma separated tags (dev):
```

If no tags are provided for multirun, an error will be raised:

```bash
>>> python train.py -m +x=1,2,3 tags=[]
ValueError: Specify tags before launching a multirun!
```

> **Note**: Appending lists from command line is currently not supported in hydra :(

</details>

<br>

## ❤️  Contributions

This project exists thanks to all the people who contribute.

![Contributors](https://readme-contributors.now.sh/ashleve/lightning-hydra-template?extension=jpg&width=400&aspectRatio=1)

Have a question? Found a bug? Missing a specific feature? Feel free to file a new issue, discussion or PR with respective title and description.

Before making an issue, please verify that:

- The problem still exists on the current `main` branch.
- Your python dependencies are updated to recent versions.

Suggestions for improvements are always welcome!

<br>

## How It Works

All PyTorch Lightning modules are dynamically instantiated from module paths specified in config. Example model config:

```yaml
_target_: src.models.mnist_model.MNISTLitModule
lr: 0.001
net:
  _target_: src.models.components.simple_dense_net.SimpleDenseNet
  input_size: 784
  lin1_size: 256
  lin2_size: 256
  lin3_size: 256
  output_size: 10
```

Using this config we can instantiate the object with the following line:

```python
model = hydra.utils.instantiate(config.model)
```

This allows you to easily iterate over new models! Every time you create a new one, just specify its module path and parameters in appropriate config file. <br>

Switch between models and datamodules with command line arguments:

```bash
python train.py model=mnist
```

Example pipeline managing the instantiation logic: [src/train.py](src/train.py).

<br>

## Main Config

Location: [configs/train.yaml](configs/train.yaml) <br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python train.py`.<br>

<details>
<summary><b>Show main project config</b></summary>

```yaml
# order of defaults determines the order in which configs override each other
defaults:
  - _self_
  - data: mnist.yaml
  - model: mnist.yaml
  - callbacks: default.yaml
  - logger: null # set logger here or use command line (e.g. `python train.py logger=csv`)
  - trainer: default.yaml
  - paths: default.yaml
  - extras: default.yaml
  - hydra: default.yaml

  # experiment configs allow for version control of specific hyperparameters
  # e.g. best hyperparameters for given model and datamodule
  - experiment: null

  # config for hyperparameter optimization
  - hparams_search: null

  # optional local config for machine/user specific settings
  # it's optional since it doesn't need to exist and is excluded from version control
  - optional local: default.yaml

  # debugging config (enable through command line, e.g. `python train.py debug=default)
  - debug: null

# task name, determines output directory path
task_name: "train"

# tags to help you identify your experiments
# you can overwrite this in experiment configs
# overwrite from command line with `python train.py tags="[first_tag, second_tag]"`
# appending lists from command line is currently not supported :(
# https://github.com/facebookresearch/hydra/issues/1547
tags: ["dev"]

# set False to skip model training
train: True

# evaluate on test set, using best model weights achieved during training
# lightning chooses best weights based on the metric specified in checkpoint callback
test: True

# simply provide checkpoint path to resume training
ckpt_path: null

# seed for random number generators in pytorch, numpy and python.random
seed: null
```

</details>

<br>

## Experiment Config

Location: [configs/experiment](configs/experiment)<br>
Experiment configs allow you to overwrite parameters from main config.<br>
For example, you can use them to version control best hyperparameters for each combination of model and dataset.

<details>
<summary><b>Show example experiment config</b></summary>

```yaml
# @package _global_

# to execute this experiment run:
# python train.py experiment=example

defaults:
  - override /data: mnist.yaml
  - override /model: mnist.yaml
  - override /callbacks: default.yaml
  - override /trainer: default.yaml

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

tags: ["mnist", "simple_dense_net"]

seed: 12345

trainer:
  min_epochs: 10
  max_epochs: 10
  gradient_clip_val: 0.5

model:
  optimizer:
    lr: 0.002
  net:
    lin1_size: 128
    lin2_size: 256
    lin3_size: 64

data:
  batch_size: 64

logger:
  wandb:
    tags: ${tags}
    group: "mnist"
```

</details>

<br>

## Workflow

**Basic workflow**

1. Write your PyTorch Lightning module (see [models/mnist_module.py](src/models/mnist_module.py) for example)
2. Write your PyTorch Lightning datamodule (see [data/mnist_datamodule.py](src/data/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to model and datamodule
4. Run training with chosen experiment config:
   ```bash
   python src/train.py experiment=experiment_name.yaml
   ```

**Experiment design**

_Say you want to execute many runs to plot how accuracy changes in respect to batch size._

1. Execute the runs with some config parameter that allows you to identify them easily, like tags:

   ```bash
   python train.py -m logger=csv data.batch_size=16,32,64,128 tags=["batch_size_exp"]
   ```

2. Write a script or notebook that searches over the `logs/` folder and retrieves csv logs from runs containing given tags in config. Plot the results.

<br>

## Logs

Hydra creates new output directory for every executed run.

Default logging structure:

```
├── logs
│   ├── task_name
│   │   ├── runs                        # Logs generated by single runs
│   │   │   ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the run
│   │   │   │   ├── .hydra                  # Hydra logs
│   │   │   │   ├── csv                     # Csv logs
│   │   │   │   ├── wandb                   # Weights&Biases logs
│   │   │   │   ├── checkpoints             # Training checkpoints
│   │   │   │   └── ...                     # Any other thing saved during training
│   │   │   └── ...
│   │   │
│   │   └── multiruns                   # Logs generated by multiruns
│   │       ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the multirun
│   │       │   ├──1                        # Multirun job number
│   │       │   ├──2
│   │       │   └── ...
│   │       └── ...
│   │
│   └── debugs                          # Logs generated when debugging config is attached
│       └── ...
```

</details>

You can change this structure by modifying paths in [hydra configuration](configs/hydra).

<br>

## Experiment Tracking

PyTorch Lightning supports many popular logging frameworks: [Weights&Biases](https://www.wandb.com/), [Neptune](https://neptune.ai/), [Comet](https://www.comet.ml/), [MLFlow](https://mlflow.org), [Tensorboard](https://www.tensorflow.org/tensorboard/).

These tools help you keep track of hyperparameters and output metrics and allow you to compare and visualize results. To use one of them simply complete its configuration in [configs/logger](configs/logger) and run:

```bash
python train.py logger=logger_name
```

You can use many of them at once (see [configs/logger/many_loggers.yaml](configs/logger/many_loggers.yaml) for example).

You can also write your own logger.

Lightning provides convenient method for logging custom metrics from inside LightningModule. Read the [docs](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#automatic-logging) or take a look at [MNIST example](src/models/mnist_module.py).

<br>

## Tests

Template comes with generic tests implemented with `pytest`.

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/test_train.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```

Most of the implemented tests don't check for any specific output - they exist to simply verify that executing some commands doesn't end up in throwing exceptions. You can execute them once in a while to speed up the development.

Currently, the tests cover cases like:

- running 1 train, val and test step
- running 1 epoch on 1% of data, saving ckpt and resuming for the second epoch
- running 2 epochs on 1% of data, with DDP simulated on CPU

And many others. You should be able to modify them easily for your use case.

There is also `@RunIf` decorator implemented, that allows you to run tests only if certain conditions are met, e.g. GPU is available or system is not windows. See the [examples](tests/test_train.py).

<br>

## Hyperparameter Search

You can define hyperparameter search by adding new config file to [configs/hparams_search](configs/hparams_search).

<details>
<summary><b>Show example hyperparameter search config</b></summary>

```yaml
# @package _global_

defaults:
  - override /hydra/sweeper: optuna

# choose metric which will be optimized by Optuna
# make sure this is the correct name of some metric logged in lightning module!
optimized_metric: "val/acc_best"

# here we define Optuna hyperparameter search
# it optimizes for value returned from function with @hydra.main decorator
hydra:
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper

    # 'minimize' or 'maximize' the objective
    direction: maximize

    # total number of runs that will be executed
    n_trials: 20

    # choose Optuna hyperparameter sampler
    # docs: https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 1234
      n_startup_trials: 10 # number of random sampling runs before optimization starts

    # define hyperparameter search space
    params:
      model.optimizer.lr: interval(0.0001, 0.1)
      data.batch_size: choice(32, 64, 128, 256)
      model.net.lin1_size: choice(64, 128, 256)
      model.net.lin2_size: choice(64, 128, 256)
      model.net.lin3_size: choice(32, 64, 128, 256)
```

</details>

Next, execute it with: `python train.py -m hparams_search=mnist_optuna`

Using this approach doesn't require adding any boilerplate to code, everything is defined in a single config file. The only necessary thing is to return the optimized metric value from the launch file.

You can use different optimization frameworks integrated with Hydra, like [Optuna, Ax or Nevergrad](https://hydra.cc/docs/plugins/optuna_sweeper/).

The `optimization_results.yaml` will be available under `logs/task_name/multirun` folder.

This approach doesn't support resuming interrupted search and advanced techniques like prunning - for more sophisticated search and workflows, you should probably write a dedicated optimization task (without multirun feature).

<br>

## Continuous Integration

Template comes with CI workflows implemented in Github Actions:

- `.github/workflows/test.yaml`: running all tests with pytest
- `.github/workflows/code-quality-main.yaml`: running pre-commits on main branch for all files
- `.github/workflows/code-quality-pr.yaml`: running pre-commits on pull requests for modified files only

<br>

## Distributed Training

Lightning supports multiple ways of doing distributed training. The most common one is DDP, which spawns separate process for each GPU and averages gradients between them. To learn about other approaches read the [lightning docs](https://lightning.ai/docs/pytorch/latest/advanced/speed.html).

You can run DDP on mnist example with 4 GPUs like this:

```bash
python train.py trainer=ddp
```

> **Note**: When using DDP you have to be careful how you write your models - read the [docs](https://lightning.ai/docs/pytorch/latest/advanced/speed.html).

<br>

## Accessing Datamodule Attributes In Model

The simplest way is to pass datamodule attribute directly to model on initialization:

```python
# ./src/train.py
datamodule = hydra.utils.instantiate(config.data)
model = hydra.utils.instantiate(config.model, some_param=datamodule.some_param)
```

> **Note**: Not a very robust solution, since it assumes all your datamodules have `some_param` attribute available.

Similarly, you can pass a whole datamodule config as an init parameter:

```python
# ./src/train.py
model = hydra.utils.instantiate(config.model, dm_conf=config.data, _recursive_=False)
```

You can also pass a datamodule config parameter to your model through variable interpolation:

```yaml
# ./configs/model/my_model.yaml
_target_: src.models.my_module.MyLitModule
lr: 0.01
some_param: ${data.some_param}
```

Another approach is to access datamodule in LightningModule directly through Trainer:

```python
# ./src/models/mnist_module.py
def on_train_start(self):
  self.some_param = self.trainer.datamodule.some_param
```

> **Note**: This only works after the training starts since otherwise trainer won't be yet available in LightningModule.

<br>

## Best Practices

<details>
<summary><b>Use Miniconda</b></summary>

It's usually unnecessary to install full anaconda environment, miniconda should be enough (weights around 80MB).

Big advantage of conda is that it allows for installing packages without requiring certain compilers or libraries to be available in the system (since it installs precompiled binaries), so it often makes it easier to install some dependencies e.g. cudatoolkit for GPU support.

It also allows you to access your environments globally which might be more convenient than creating new local environment for every project.

Example installation:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Update conda:

```bash
conda update -n base -c defaults conda
```

Create new conda environment:

```bash
conda create -n myenv python=3.10
conda activate myenv
```

</details>

<details>
<summary><b>Use automatic code formatting</b></summary>

Use pre-commit hooks to standardize code formatting of your project and save mental energy.<br>
Simply install pre-commit package with:

```bash
pip install pre-commit
```

Next, install hooks from [.pre-commit-config.yaml](.pre-commit-config.yaml):

```bash
pre-commit install
```

After that your code will be automatically reformatted on every new commit.

To reformat all files in the project use command:

```bash
pre-commit run -a
```

To update hook versions in [.pre-commit-config.yaml](.pre-commit-config.yaml) use:

```bash
pre-commit autoupdate
```

</details>

<details>
<summary><b>Set private environment variables in .env file</b></summary>

System specific variables (e.g. absolute paths to datasets) should not be under version control or it will result in conflict between different users. Your private keys also shouldn't be versioned since you don't want them to be leaked.<br>

Template contains `.env.example` file, which serves as an example. Create a new file called `.env` (this name is excluded from version control in .gitignore).
You should use it for storing environment variables like this:

```
MY_VAR=/home/user/my_system_path
```

All variables from `.env` are loaded in `train.py` automatically.

Hydra allows you to reference any env variable in `.yaml` configs like this:

```yaml
path_to_data: ${oc.env:MY_VAR}
```

</details>

<details>
<summary><b>Name metrics using '/' character</b></summary>

Depending on which logger you're using, it's often useful to define metric name with `/` character:

```python
self.log("train/loss", loss)
```

This way loggers will treat your metrics as belonging to different sections, which helps to get them organised in UI.

</details>

<details>
<summary><b>Use torchmetrics</b></summary>

Use official [torchmetrics](https://github.com/PytorchLightning/metrics) library to ensure proper calculation of metrics. This is especially important for multi-GPU training!

For example, instead of calculating accuracy by yourself, you should use the provided `Accuracy` class like this:

```python
from torchmetrics.classification.accuracy import Accuracy


class LitModel(LightningModule):
    def __init__(self)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        ...
        acc = self.train_acc(predictions, targets)
        self.log("train/acc", acc)
        ...

    def validation_step(self, batch, batch_idx):
        ...
        acc = self.val_acc(predictions, targets)
        self.log("val/acc", acc)
        ...
```

Make sure to use different metric instance for each step to ensure proper value reduction over all GPU processes.

Torchmetrics provides metrics for most use cases, like F1 score or confusion matrix. Read [documentation](https://torchmetrics.readthedocs.io/en/latest/#more-reading) for more.

</details>

<details>
<summary><b>Follow PyTorch Lightning style guide</b></summary>

The style guide is available [here](https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html).<br>

1. Be explicit in your init. Try to define all the relevant defaults so that the user doesn’t have to guess. Provide type hints. This way your module is reusable across projects!

   ```python
   class LitModel(LightningModule):
       def __init__(self, layer_size: int = 256, lr: float = 0.001):
   ```

2. Preserve the recommended method order.

   ```python
   class LitModel(LightningModule):

       def __init__():
           ...

       def forward():
           ...

       def training_step():
           ...

       def training_step_end():
           ...

       def on_train_epoch_end():
           ...

       def validation_step():
           ...

       def validation_step_end():
           ...

       def on_validation_epoch_end():
           ...

       def test_step():
           ...

       def test_step_end():
           ...

       def on_test_epoch_end():
           ...

       def configure_optimizers():
           ...

       def any_extra_hook():
           ...
   ```

</details>

<details>
<summary><b>Version control your data and models with DVC</b></summary>

Use [DVC](https://dvc.org) to version control big files, like your data or trained ML models.<br>
To initialize the dvc repository:

```bash
dvc init
```

To start tracking a file or directory, use `dvc add`:

```bash
dvc add data/MNIST
```

DVC stores information about the added file (or a directory) in a special .dvc file named data/MNIST.dvc, a small text file with a human-readable format. This file can be easily versioned like source code with Git, as a placeholder for the original data:

```bash
git add data/MNIST.dvc data/.gitignore
git commit -m "Add raw data"
```

</details>

<details>
<summary><b>Support installing project as a package</b></summary>

It allows other people to easily use your modules in their own projects.
Change name of the `src` folder to your project name and complete the `setup.py` file.

Now your project can be installed from local files:

```bash
pip install -e .
```

Or directly from git repository:

```bash
pip install git+git://github.com/YourGithubName/your-repo-name.git --upgrade
```

So any file can be easily imported into any other file like so:

```python
from project_name.models.mnist_module import MNISTLitModule
from project_name.data.mnist_datamodule import MNISTDataModule
```

</details>

<details>
<summary><b>Keep local configs out of code versioning</b></summary>

Some configurations are user/machine/installation specific (e.g. configuration of local cluster, or harddrive paths on a specific machine). For such scenarios, a file [configs/local/default.yaml](configs/local/) can be created which is automatically loaded but not tracked by Git.

For example, you can use it for a SLURM cluster config:

```yaml
# @package _global_

defaults:
  - override /hydra/launcher@_here_: submitit_slurm

data_dir: /mnt/scratch/data/

hydra:
  launcher:
    timeout_min: 1440
    gpus_per_task: 1
    gres: gpu:1
  job:
    env_set:
      MY_VAR: /home/user/my/system/path
      MY_KEY: asdgjhawi8y23ihsghsueity23ihwd
```

</details>

<br>

## Resources

+ 与当前项目相关的资源和灵感来源
This template was inspired by:

- [PyTorchLightning/deep-learning-project-template](https://github.com/PyTorchLightning/deep-learning-project-template)
  -  PyTorch Lightning 团队创建的，用于深度学习项目的模板。
- [drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science)
  -  DrivenData 团队创建的，用于数据科学项目的模板。
- [lucmos/nn-template](https://github.com/lucmos/nn-template)
  - 用于深度学习项目的模板。
Other useful repositories:

- [jxpress/lightning-hydra-template-vertex-ai](https://github.com/jxpress/lightning-hydra-template-vertex-ai) - lightning-hydra-template integration with Vertex AI hyperparameter tuning and custom training job
  -集成了 Vertex AI 的超参数调整和自定义训练作业功能，与当前的 "lightning-hydra-template" 进行了整合。
</details>

<br>

## License
+ 许可证
Lightning-Hydra-Template is licensed under the MIT License.

```
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<br>
<br>
<br>
<br>

**DELETE EVERYTHING ABOVE FOR YOUR PROJECT**
为你的项目删除上面所有内容
______________________________________________________________________

<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
