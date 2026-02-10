import torch
import torch.nn as nn
import numpy as np


class GaussianDiffusion:
    """
    Gaussian diffusion process
    Supports multiple variance scheduling strategies
    """
    
    def __init__(self, timesteps=1000, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
        """
        Args:
            timesteps: Total number of time steps
            beta_schedule: Variance schedule method ('linear', 'quadratic', 'cosine')
            beta_start: Initial beta value
            beta_end: Final beta value
        """
        self.timesteps = timesteps
        self.beta_schedule = beta_schedule
        
        # Calculate beta sequence
        if beta_schedule == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'quadratic':
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2
        elif beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
        
        # Calculate key quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])
        
        # Constants required for forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Constants required for reverse process
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    @staticmethod
    def _cosine_beta_schedule(timesteps, s=0.008):
        """Cosine beta schedule"""
        steps = torch.arange(timesteps + 1, dtype=torch.float32) / float(timesteps)
        alphas_cumprod = torch.cos((steps + s) / (1 + s) * torch.tensor(np.pi) * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def to(self, device):
        """Move all tensors to the specified device"""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward process: q(x_t|x_0)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_0: Original data (batch_size, dim)
            t: Time steps (batch_size,)
            noise: Gaussian noise, generated automatically if None
            
        Returns:
            x_t: Noisy data
            noise: Noise used
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        batch_size = x_0.shape[0]
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Adjust shape for broadcasting
        sqrt_alpha_bar = sqrt_alpha_bar.view(batch_size, *([1] * (len(x_0.shape) - 1)))
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(batch_size, *([1] * (len(x_0.shape) - 1)))
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
    
    def q_posterior_mean_variance(self, x_0, x_t, t):
        """
        Compute posterior distribution q(x_{t-1}|x_t, x_0) mean and variance
        
        Args:
            x_0: Original data
            x_t: Data at time t
            t: Time steps
            
        Returns:
            mean: Posterior mean
            variance: Posterior variance
            log_variance: Log of posterior variance
        """
        batch_size = x_0.shape[0]
        
        coef1 = self.posterior_mean_coef1[t].view(batch_size, *([1] * (len(x_0.shape) - 1)))
        coef2 = self.posterior_mean_coef2[t].view(batch_size, *([1] * (len(x_0.shape) - 1)))
        
        mean = coef1 * x_0 + coef2 * x_t
        variance = self.posterior_variance[t].view(batch_size, *([1] * (len(x_0.shape) - 1)))
        log_variance = self.posterior_log_variance_clipped[t].view(batch_size, *([1] * (len(x_0.shape) - 1)))
        
        return mean, variance, log_variance
