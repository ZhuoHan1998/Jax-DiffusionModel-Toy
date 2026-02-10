# Jax Diffusion Toy

一个模块化的Diffusion Model实现，用于生成3D Swiss Roll等toy数据集。

## 特点

- **模块化设计**: 每个组件都是即插即用的
- **多种采样方法**: DDPM和DDIM采样器
- **Classifier-Free Guidance**: 支持条件生成
- **灵活的网络架构**: 可配置的UNet模型
- **完整的训练框架**: 包括数据加载、优化、检查点等
- **可视化工具**: 内置的2D和3D可视化函数

## 项目结构

```
Jax-Diffusion-Toy/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── toy_datasets.py      # Swiss Roll, Gaussian, Sinusoid数据集
│   ├── diffusion/
│   │   ├── __init__.py
│   │   └── diffusion.py         # 高斯扩散过程
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # 时间和条件编码
│   │   └── unet.py              # UNet架构
│   ├── samplers/
│   │   ├── __init__.py
│   │   ├── base_sampler.py      # 采样器基类
│   │   ├── ddpm_sampler.py      # DDPM采样器
│   │   └── ddim_sampler.py      # DDIM采样器（加速）
│   ├── train.py                 # 训练框架
│   └── inference.py             # 推理和可视化工具
├── notebooks/
│   └── tutorial.ipynb           # 教程notebook
├── config.py                    # 配置文件
├── train_swiss_roll.py          # 训练脚本示例
├── requirements.txt             # 依赖列表
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
python train_swiss_roll.py
```

### 3. 或使用Jupyter Notebook

```bash
jupyter notebook notebooks/tutorial.ipynb
```

## 核心模块说明

### 数据集 (src/data/)

- **SwissRoll**: 3D Swiss Roll数据集
- **Gaussian**: 高斯分布数据
- **Sinusoid**: 正弦波形数据

```python
from src.data import SwissRoll

dataset = SwissRoll(n_samples=5000, height=21, noise=0.1)
```

### 扩散过程 (src/diffusion/)

高斯扩散过程，支持多种beta调度:

```python
from src.diffusion import GaussianDiffusion

diffusion = GaussianDiffusion(
    timesteps=1000,
    beta_schedule='linear',  # 或 'quadratic', 'cosine'
    beta_start=0.0001,
    beta_end=0.02
)
```

### 网络模型 (src/models/)

SimpleUNet - 用于toy数据的轻量级UNet:

```python
from src.models import SimpleUNet

model = SimpleUNet(
    data_dim=3,
    time_emb_dim=128,
    cond_emb_dim=128,
    num_classes=None,  # 无条件生成
    hidden_dims=[64, 128, 256, 128, 64]
)
```

### 采样器 (src/samplers/)

支持多种采样方法：

```python
from src.samplers import DDPMSampler, DDIMSampler

# DDPM - 标准采样（1000步）
ddpm_sampler = DDPMSampler(diffusion, model, device='cuda')
samples = ddpm_sampler.sample(batch_size=100, data_dim=3)

# DDIM - 加速采样（50步）
ddim_sampler = DDIMSampler(diffusion, model, device='cuda', num_steps=50, eta=0.0)
samples = ddim_sampler.sample(batch_size=100, data_dim=3)
```

### Classifier-Free Guidance

支持条件生成和无分类器指导：

```python
# 需要在模型中指定num_classes
model = SimpleUNet(
    data_dim=3,
    num_classes=10,  # 10个类别
    ...
)

# 采样时使用指导
class_labels = torch.tensor([0, 1, 2])
samples = sampler.sample(
    batch_size=100,
    data_dim=3,
    class_labels=class_labels,
    guidance_scale=7.5  # 指导强度（1.0=无指导）
)
```

### 训练 (src/train.py)

```python
from src.train import DiffusionTrainer

trainer = DiffusionTrainer(model, diffusion, device='cuda')
trainer.setup_optimizer(learning_rate=1e-3)
trainer.setup_scheduler(mode='cosine', num_epochs=100)
trainer.train(train_loader, num_epochs=100, save_every=10)
```

### 推理和可视化 (src/inference.py)

```python
from src.inference import DiffusionInference

inference = DiffusionInference(sampler, device='cuda')

# 生成样本
samples = inference.generate(batch_size=1000, data_dim=3)

# 可视化
inference.plot_samples_3d(samples, real_data=real_data, save_path='result.png')

# 评估
metrics = inference.evaluate_fid_like(samples, real_data)
```

## 配置示例

在 `config.py` 中提供了三个预设配置：

- `SWISS_ROLL_CONFIG` - 3D Swiss Roll
- `GAUSSIAN_CONFIG` - 2D高斯分布
- `SINUSOID_CONFIG` - 2D正弦波形

## 高级用法

### 自定义数据集

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, ...):
        ...
    
    def __len__(self):
        ...
    
    def __getitem__(self, idx):
        ...
```

### 自定义网络架构

继承 `SimpleUNet` 或 `UNet` 官实现自己的架构：

```python
import torch.nn as nn
from src.models import SimpleUNet

class CustomUNet(SimpleUNet):
    def __init__(self, ...):
        super().__init__(...)
        # 添加自定义层
    
    def forward(self, x, t, class_labels=None):
        # 自定义前向传播
        return super().forward(x, t, class_labels)
```

### 自定义采样器

继承基采样器：

```python
from src.samplers import BaseSampler

class CustomSampler(BaseSampler):
    def sample(self, batch_size, data_dim, class_labels=None, guidance_scale=1.0):
        # 实现自定义采样逻辑
        pass
```

## 性能优化建议

1. **使用DDIM采样** - 将采样步数从1000减少到50，速度提升20倍
2. **调整网络大小** - 根据数据维度调整 `hidden_dims`
3. **批处理** - 增加 `batch_size` 来提高GPU利用率
4. **混合精度训练** - 使用 `torch.cuda.amp` 加速训练

## 故障排查

### CUDA内存不足

- 减少 `batch_size`
- 减少 `hidden_dims`
- 减少 `timesteps`

### 生成质量差

- 增加 `num_epochs`
- 降低 `learning_rate`
- 使用更好的 `beta_schedule`（如 'cosine'）
- 增加 `hidden_dims`

## 参考论文

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (DDIM)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

## 许可证

MIT

## 作者

Created with PyTorch
