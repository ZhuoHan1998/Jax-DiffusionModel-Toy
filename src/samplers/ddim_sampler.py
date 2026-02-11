import torch
from .base_sampler import BaseSampler


class DDIMSampler(BaseSampler):
    """
    DDIM sampler
    Reference: Denoising Diffusion Implicit Models
    Supports accelerated sampling
    """
    
    def __init__(self, diffusion_model, denoising_model, device='cpu', num_steps=50, eta=0.0):
        """
        Args:
            diffusion_model: GaussianDiffusion instance
            denoising_model: Denoising network model
            device: Computation device
            num_steps: Number of sampling steps (less than total timesteps)
            eta: Stochasticity parameter (0.0 for deterministic, 1.0 for high stochasticity)
        """
        super().__init__(diffusion_model, denoising_model, device)
        
        self.num_steps = num_steps
        self.eta = eta
        
        # Calculate sampling timesteps
        self.timestep_indices = self._get_timestep_indices()
    
    def _get_timestep_indices(self):
        """Get uniformly distributed timestep indices"""
        indices = torch.linspace(0, self.diffusion.timesteps - 1, self.num_steps, dtype=torch.long)
        return indices.tolist()[::-1]  # Reverse order
    
    def sample(self, batch_size, data_dim, class_labels=None, guidance_scale=1.0):
        """
        Sample using DDIM method
        
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
        for step_idx, t in enumerate(self.timestep_indices):
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                # Predict noise
                if class_labels is not None and guidance_scale != 1.0:
                    # Classifier-Free Guidance
                    noise_pred = self.model(x_t, t_batch, class_labels)
                    noise_pred_uncond = self.model(x_t, t_batch, None)
                    noise_pred = self._classifier_free_guidance(
                        noise_pred, noise_pred_uncond, guidance_scale
                    )
                else:
                    noise_pred = self.model(x_t, t_batch, class_labels)
                
                # DDIM sampling formula
                alpha_bar_t = self.diffusion.alphas_cumprod[t].item()
                
                # Get previous timestep
                if step_idx < len(self.timestep_indices) - 1:
                    t_prev = self.timestep_indices[step_idx + 1]
                    alpha_bar_t_prev = self.diffusion.alphas_cumprod[t_prev].item()
                else:
                    alpha_bar_t_prev = 1.0
                
                # Get x_0 from noise prediction
                x_0_pred = (x_t - torch.sqrt(torch.tensor(1 - alpha_bar_t, device=self.device)) * noise_pred) / torch.sqrt(torch.tensor(alpha_bar_t, device=self.device))
                
                # Calculate sigma (control stochasticity)
                sigma = self.eta * torch.sqrt(
                    torch.tensor((1 - alpha_bar_t_prev) / (1 - alpha_bar_t), device=self.device) *
                    torch.tensor((1 - alpha_bar_t / alpha_bar_t_prev), device=self.device)
                )
                
                # DDIM sampling
                c1 = torch.sqrt(torch.tensor(alpha_bar_t_prev, device=self.device))
                c2 = torch.sqrt(torch.tensor(1 - alpha_bar_t_prev - sigma ** 2, device=self.device))
                
                x_t = c1 * x_0_pred + c2 * noise_pred
                
                # Add random noise
                if step_idx < len(self.timestep_indices) - 1:
                    x_t = x_t + sigma * torch.randn_like(x_t)
        
        return x_t
