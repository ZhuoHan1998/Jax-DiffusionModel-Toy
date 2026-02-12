import torch
from .base_sampler import BaseSampler


class DDPMSampler(BaseSampler):
    """
    DDPM sampler
    Reference: Denoising Diffusion Probabilistic Models
    """
    
    def sample(self, batch_size, data_dim, class_labels=None, guidance_scale=1.0):
        """
        Sample using DDPM method
        
        Args:
            batch_size: Batch size
            data_dim: Data dimension
            class_labels: Class labels (batch_size,) or None
            guidance_scale: Classifier-Free Guidance strength
            
        Returns:
            Sampled data (batch_size, data_dim)
        """
        # Start from Gaussian noise
        x_t = torch.randn(batch_size, data_dim, device=self.device)
        
        # Reverse diffusion process
        for t in reversed(range(self.diffusion.timesteps)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                # Predict noise
                if class_labels is not None and guidance_scale > 1.0:
                    # Classifier-Free Guidance
                    noise_pred = self.model(x_t, t_batch, class_labels)
                    
                    # Unconditional prediction
                    noise_pred_uncond = self.model(x_t, t_batch, None)
                    
                    # Apply guidance
                    noise_pred = self._classifier_free_guidance(
                        noise_pred, noise_pred_uncond, guidance_scale
                    )
                else:
                    noise_pred = self.model(x_t, t_batch, class_labels)
                
                # Get coefficients directly from tensors (avoid .item() for numerical stability)
                posterior_mean_coef1 = self.diffusion.posterior_mean_coef1[t]
                posterior_mean_coef2 = self.diffusion.posterior_mean_coef2[t]
                posterior_variance = self.diffusion.posterior_variance[t]
                
                # Calculate mean using correct DDPM formula: x_0 prediction
                # x_0 = (x_t - sqrt(1 - alpha_bar_t) * noise) / sqrt(alpha_bar_t)
                alpha_bar_t = self.diffusion.alphas_cumprod[t]
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
                
                # Predict x_0 from noise prediction
                x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
                
                # Calculate posterior mean: q(x_{t-1}|x_t, x_0)
                if t > 0:
                    alpha_bar_t_prev = self.diffusion.alphas_cumprod[t-1]
                    sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
                    sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)
                    
                    # Posterior mean formula
                    mean = (sqrt_alpha_bar_t_prev * self.diffusion.betas[t] / (1 - alpha_bar_t)) * x_0_pred + \
                           ((1 - alpha_bar_t_prev) * torch.sqrt(self.diffusion.alphas[t]) / (1 - alpha_bar_t)) * x_t
                    
                    # Add noise
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.sqrt(posterior_variance) * noise
                else:
                    x_t = x_0_pred
        
        return x_t
