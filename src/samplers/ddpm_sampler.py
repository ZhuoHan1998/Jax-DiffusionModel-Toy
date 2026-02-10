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
                
                # Calculate mean
                alpha = self.diffusion.alphas[t].item()
                alpha_bar = self.diffusion.alphas_cumprod[t].item()
                alpha_bar_prev = self.diffusion.alphas_cumprod_prev[t].item()
                
                posterior_mean_coef1 = self.diffusion.posterior_mean_coef1[t].item()
                posterior_mean_coef2 = self.diffusion.posterior_mean_coef2[t].item()
                
                # Calculate mean
                mean = posterior_mean_coef1 * noise_pred + posterior_mean_coef2 * x_t
                
                # Calculate variance
                posterior_variance = self.diffusion.posterior_variance[t].item()
                
                # Add noise if not the last step
                if t > 0:
                    noise = torch.randn_like(x_t)
                    x_t = mean + torch.sqrt(torch.tensor(posterior_variance, device=self.device)) * noise
                else:
                    x_t = mean
        
        return x_t
