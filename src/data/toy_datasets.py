import numpy as np
import torch
from torch.utils.data import Dataset


class SwissRoll(Dataset):
    """Generate 3D Swiss Roll dataset"""
    
    def __init__(self, n_samples=5000, height=21, noise=0.0, seed=None):
        """
        Args:
            n_samples: Number of samples
            height: Height of the roll
            noise: Standard deviation of Gaussian noise
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_samples = n_samples
        self.data = self._generate_swiss_roll(n_samples, height, noise)
        self.labels = None  # Can be used for classifier-free guidance conditions
        
    def _generate_swiss_roll(self, n_samples, height, noise):
        """Generate Swiss Roll data"""
        t = 3 * np.pi * np.random.rand(n_samples)
        h = height * np.random.rand(n_samples)
        
        x = t * np.cos(t)
        y = h
        z = t * np.sin(t)
        
        if noise > 0:
            x += np.random.randn(n_samples) * noise
            y += np.random.randn(n_samples) * noise
            z += np.random.randn(n_samples) * noise
        
        data = np.column_stack([x, y, z]).astype(np.float32)
        # Standardize to mean=0, std=1 (proper normalization for diffusion)
        data = (data - data.mean(axis=0)) / data.std(axis=0)
        return data
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


class Gaussian(Dataset):
    """Generate Gaussian distribution dataset"""
    
    def __init__(self, n_samples=5000, dim=2, mean=None, cov=None, seed=None):
        """
        Args:
            n_samples: Number of samples
            dim: Dimension
            mean: Mean vector
            cov: Covariance matrix
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_samples = n_samples
        self.dim = dim
        
        if mean is None:
            mean = np.zeros(dim)
        if cov is None:
            cov = np.eye(dim)
        
        self.data = np.random.multivariate_normal(mean, cov, n_samples).astype(np.float32)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


class Sinusoid(Dataset):
    """Generate sinusoidal wave dataset"""
    
    def __init__(self, n_samples=5000, freq=2.0, phase=0.0, noise=0.0, seed=None):
        """
        Args:
            n_samples: Number of samples
            freq: Frequency
            phase: Phase
            noise: Standard deviation of Gaussian noise
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.n_samples = n_samples
        x = np.linspace(0, 4 * np.pi, n_samples)
        y = np.sin(freq * x + phase)
        
        if noise > 0:
            y += np.random.randn(n_samples) * noise
        
        self.data = np.column_stack([x, y]).astype(np.float32)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])
