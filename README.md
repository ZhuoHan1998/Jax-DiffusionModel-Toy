# Jax Diffusion Toy

A modularized Diffusion Model implementation for generating toy datasets like 3D Swiss Roll.

## Features

- **Modular Design**: Each component is plug-and-play
- **Multiple Sampling Methods**: DDPM and DDIM samplers
- **Classifier-Free Guidance**: Support for conditional generation
- **Flexible Network Architecture**: Configurable UNet model
- **Complete Training Framework**: Including data loading, optimization, checkpoints, etc.
- **Visualization Tools**: Built-in 2D and 3D visualization functions

## Project Structure

```
Jax-Diffusion-Toy/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── toy_datasets.py      # Swiss Roll, Gaussian, Sinusoid datasets
│   ├── diffusion/
│   │   ├── __init__.py
│   │   └── diffusion.py         # Gaussian diffusion process
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embeddings.py        # Time and condition encoding
│   │   └── unet.py              # UNet architecture
│   ├── samplers/
│   │   ├── __init__.py
│   │   ├── base_sampler.py      # Sampler base class
│   │   ├── ddpm_sampler.py      # DDPM sampler
│   │   └── ddim_sampler.py      # DDIM sampler (accelerated)
│   ├── train.py                 # Training framework
│   └── inference.py             # Inference and visualization tools
├── notebooks/
│   └── tutorial.ipynb           # Tutorial notebook
├── config.py                    # Configuration file
├── train_swiss_roll.py          # Training script example
├── requirements.txt             # Dependency list
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model

```bash
python train_swiss_roll.py
```

### 3. Or Use Jupyter Notebook

```bash
jupyter notebook notebooks/tutorial.ipynb
```

## Core Module Documentation

### Datasets (src/data/)

- **SwissRoll**: 3D Swiss Roll dataset
- **Gaussian**: Gaussian distribution data
- **Sinusoid**: Sinusoidal waveform data

```python
from src.data import SwissRoll

dataset = SwissRoll(n_samples=5000, height=21, noise=0.1)
```

### Diffusion Process (src/diffusion/)

Gaussian diffusion process, supporting multiple beta schedules:

```python
from src.diffusion import GaussianDiffusion

diffusion = GaussianDiffusion(
    timesteps=1000,
    beta_schedule='linear',  # or 'quadratic', 'cosine'
    beta_start=0.0001,
    beta_end=0.02
)
```

### Network Model (src/models/)

SimpleUNet - Lightweight UNet for toy data:

```python
from src.models import SimpleUNet

model = SimpleUNet(
    data_dim=3,
    time_emb_dim=128,
    cond_emb_dim=128,
    num_classes=None,  # unconditional generation
    hidden_dims=[64, 128, 256, 128, 64]
)
```

### Samplers (src/samplers/)

Support for multiple sampling methods:

```python
from src.samplers import DDPMSampler, DDIMSampler

# DDPM - standard sampling (1000 steps)
ddpm_sampler = DDPMSampler(diffusion, model, device='cuda')
samples = ddpm_sampler.sample(batch_size=100, data_dim=3)

# DDIM - accelerated sampling (50 steps)
ddim_sampler = DDIMSampler(diffusion, model, device='cuda', num_steps=50, eta=0.0)
samples = ddim_sampler.sample(batch_size=100, data_dim=3)
```

### Classifier-Free Guidance

Support for conditional generation and classifier-free guidance:

```python
# Need to specify num_classes in the model
model = SimpleUNet(
    data_dim=3,
    num_classes=10,  # 10 classes
    ...
)

# Use guidance during sampling
class_labels = torch.tensor([0, 1, 2])
samples = sampler.sample(
    batch_size=100,
    data_dim=3,
    class_labels=class_labels,
    guidance_scale=7.5  # guidance strength (1.0=no guidance)
)
```

### Training (src/train.py)

```python
from src.train import DiffusionTrainer

trainer = DiffusionTrainer(model, diffusion, device='cuda')
trainer.setup_optimizer(learning_rate=1e-3)
trainer.setup_scheduler(mode='cosine', num_epochs=100)
trainer.train(train_loader, num_epochs=100, save_every=10)
```

### Inference and Visualization (src/inference.py)

```python
from src.inference import DiffusionInference

inference = DiffusionInference(sampler, device='cuda')

# Generate samples
samples = inference.generate(batch_size=1000, data_dim=3)

# Visualization
inference.plot_samples_3d(samples, real_data=real_data, save_path='result.png')

# Evaluation
metrics = inference.evaluate_fid_like(samples, real_data)
```

## Configuration Examples

Three preset configurations are provided in `config.py`:

- `SWISS_ROLL_CONFIG` - 3D Swiss Roll
- `GAUSSIAN_CONFIG` - 2D Gaussian distribution
- `SINUSOID_CONFIG` - 2D Sinusoidal waveform

## Advanced Usage

### Custom Datasets

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

### Custom Network Architecture

Inherit from `SimpleUNet` or `UNet` to implement your own architecture:

```python
import torch.nn as nn
from src.models import SimpleUNet

class CustomUNet(SimpleUNet):
    def __init__(self, ...):
        super().__init__(...)
        # Add custom layers
    
    def forward(self, x, t, class_labels=None):
        # Custom forward pass
        return super().forward(x, t, class_labels)
```

### Custom Sampler

Inherit from base sampler:

```python
from src.samplers import BaseSampler

class CustomSampler(BaseSampler):
    def sample(self, batch_size, data_dim, class_labels=None, guidance_scale=1.0):
        # Implement custom sampling logic
        pass
```

## Performance Optimization Tips

1. **Use DDIM Sampling** - Reduce sampling steps from 1000 to 50, 20x speed improvement
2. **Adjust Network Size** - Adjust `hidden_dims` based on data dimension
3. **Batch Processing** - Increase `batch_size` to improve GPU utilization
4. **Mixed Precision Training** - Use `torch.cuda.amp` to accelerate training

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size`
- Reduce `hidden_dims`
- Reduce `timesteps`

### Poor Generation Quality

- Increase `num_epochs`
- Lower `learning_rate`
- Use better `beta_schedule` (e.g., 'cosine')
- Increase `hidden_dims`

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (DDPM)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (DDIM)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

## License

MIT

## Author

Created with PyTorch
