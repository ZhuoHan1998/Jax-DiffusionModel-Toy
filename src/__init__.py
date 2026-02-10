"""
JAX Diffusion Toy - Modular Diffusion Model Framework

Core modules:
- data: Datasets and data loading
- diffusion: Gaussian diffusion process
- models: Network architectures (UNet)
- samplers: Sampling methods (DDPM, DDIM, etc.)
- train: Training framework
- inference: Inference and visualization tools
"""

from .data import SwissRoll, Gaussian, Sinusoid
from .diffusion import GaussianDiffusion
from .models import SimpleUNet, UNet
from .samplers import DDPMSampler, DDIMSampler
from .train import DiffusionTrainer
from .inference import DiffusionInference

__version__ = '0.1.0'
__all__ = [
    'SwissRoll', 'Gaussian', 'Sinusoid',
    'GaussianDiffusion',
    'SimpleUNet', 'UNet',
    'DDPMSampler', 'DDIMSampler',
    'DiffusionTrainer',
    'DiffusionInference'
]
