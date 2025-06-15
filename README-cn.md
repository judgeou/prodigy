# Prodigy: 一个快速自适应的无参数学习器
[![Downloads](https://static.pepy.tech/badge/prodigyopt)](https://pepy.tech/project/prodigyopt) [![Downloads](https://static.pepy.tech/badge/prodigyopt/month)](https://pepy.tech/project/prodigyopt)

这是提出 Prodigy 优化器论文的官方仓库，用于运行论文中的实验。该优化器在 PyTorch 中实现。
在 [Optax 中也有 Prodigy 的 JAX 版本](https://optax.readthedocs.io/en/latest/api/contrib.html#prodigy)，目前还没有 `slice_p` 参数。

**Prodigy: 一个快速自适应的无参数学习器**  
*K. Mishchenko, A. Defazio*  
论文链接: https://arxiv.org/pdf/2306.06101.pdf

## 安装
要安装此包，只需运行：
```pip install prodigyopt```

## 使用方法
设 `net` 为您要训练的神经网络。然后，您可以按以下方式使用该方法：
```
from prodigyopt import Prodigy
# 根据您的问题选择权重衰减值，默认为 0
# 如果内存有限，请将 slice_p 设置为 11，默认为 1
opt = Prodigy(net.parameters(), lr=1., weight_decay=weight_decay, slice_p=slice_p)
```

### 使用余弦衰减的高级用法
Prodigy 现在支持内置的余弦学习率衰减，可以自动检测峰值学习率并应用衰减：
```
from prodigyopt import Prodigy
opt = Prodigy(
    net.parameters(), 
    lr=1., 
    weight_decay=weight_decay, 
    slice_p=slice_p,
    cosine_decay=True,                    # 启用自动余弦衰减
    peak_patience=50,                     # 等待 50 步无 d 增长来检测峰值
    total_training_steps=10000            # 总训练步数，用于正确的衰减调度
)
```

**新参数：**
- `cosine_decay` (bool): 在检测到峰值学习率后启用自动余弦衰减（默认：False）
- `peak_patience` (int): 考虑达到峰值的连续无 d 增长步数（默认：50）
- `total_training_steps` (int): 总训练步数。当检测到峰值时，余弦衰减将自动使用剩余步数。如果为 None，则从峰值开始默认 1000 步（默认：None）

默认情况下，Prodigy 使用类似 AdamW 的权重衰减。
如果您希望使用标准的 $\ell_2$ 正则化（如 Adam），请使用选项 `decouple=False`。
我们建议对所有网络使用 `lr=1.`（默认值）。如果您想强制该方法估计更小或更大的学习率，
最好更改 `d_coef` 的值（默认为 1.0）。`d_coef` 值大于 1，如 2 或 10，
将强制估计更大的学习率；如果您想要更小的学习率，请将其设置为 0.5 甚至 0.1。
可尝试的标准 `weight_decay` 值有 0（Prodigy 中的默认值）、0.001、0.01（AdamW 中的默认值）和 0.1。
使用大于 1 的 `slice_p` 值来减少内存消耗。`slice_p=11` 应该在估计学习率的准确性和内存效率之间提供良好的权衡。

## 调度器
根据经验，我们建议不使用调度器、使用内置余弦衰减功能或使用余弦退火方法：

### 内置余弦衰减（推荐）
优化器现在包含自动余弦衰减，可以检测峰值学习率并应用衰减：
```
opt = Prodigy(
    net.parameters(),
    cosine_decay=True,
    peak_patience=50,              # 根据您的训练动态调整
    total_training_steps=total_steps
)
```

### 外部调度器
或者，您可以使用外部调度器：
```
# n_epoch 是训练网络的总轮数
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
```
我们不建议在余弦退火中使用重启，因此建议设置 `T_max=total_steps`，其中
`total_steps` 应该是调用 `scheduler.step()` 的次数。如果您确实使用重启，我们强烈
建议设置 `safeguard_warmup=True`。

**注意：** 当使用内置的 `cosine_decay=True` 时，避免使用外部学习率调度器，因为它们可能干扰自动衰减机制。

如果您在开始时使用线性预热，需要特别注意：
该方法会由于初始基础学习率较小而看到缓慢的进展，
因此可能会高估 `d`。
为避免预热问题，请使用选项 `safeguard_warmup=True`。

## 扩散模型
根据与一些用户的交互，我们建议在训练扩散模型时设置 `safeguard_warmup=True`、
`use_bias_correction=True` 和 `weight_decay=0.01`。
有时，[设置 `betas=(0.9, 0.99)` 是有帮助的](https://github.com/konstmish/prodigy/issues/8)。
如果模型没有训练，请尝试跟踪 `d`，如果它保持太小，[可能值得将 `d0` 增加](https://github.com/konstmish/prodigy/issues/27)到 1e-5 甚至 1e-4。
话虽如此，在我们的其他实验中，优化器对 `d0` 大多不敏感。

对于扩散模型，内置余弦衰减可能特别有用：
```
opt = Prodigy(
    net.parameters(),
    safeguard_warmup=True,
    use_bias_correction=True,
    weight_decay=0.01,
    cosine_decay=True,
    peak_patience=100,  # 扩散模型可能需要更高的耐心值
    total_training_steps=total_steps
)
```

## 使用 Prodigy 的示例

请参阅[此 Colab 笔记本](https://colab.research.google.com/drive/1TrhEfI3stJ-yNp7_ZxUAtfWjj-Qe_Hym?usp=sharing)
了解如何使用 Prodigy 在 Cifar10 上训练 ResNet-18 的简单示例（20 轮后测试准确率 80%）。
如果您有兴趣分享您的经验，请考虑创建一个 Colab 笔记本并在 issues 中分享。

## 如何引用
如果您发现我们的工作有用，请考虑引用我们的论文。
```
@inproceedings{mishchenko2024prodigy,
    title={Prodigy: An Expeditiously Adaptive Parameter-Free Learner},
    author={Mishchenko, Konstantin and Defazio, Aaron},
    booktitle={Forty-first International Conference on Machine Learning},
    year={2024},
    url={https://openreview.net/forum?id=JJpOssn0uP}
}
``` 