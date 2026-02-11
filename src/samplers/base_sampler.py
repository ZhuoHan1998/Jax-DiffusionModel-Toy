import torch
import abc


class BaseSampler(abc.ABC):
    """Base sampler class"""
    
    def __init__(self, diffusion_model, denoising_model, device='cpu'):
        """
        Args:
            diffusion_model: GaussianDiffusion instance
            denoising_model: Denoising network model
            device: Computation device
        """
        self.diffusion = diffusion_model
        self.model = denoising_model
        self.device = device
        
        # Move diffusion model to device
        self.diffusion.to(device)
        self.model.to(device)
        self.model.eval()
    
    @abc.abstractmethod
    def sample(self, batch_size, data_dim, class_labels=None, guidance_scale=1.0):
        """
        Sample
        
        Args:
            batch_size: Batch size
            data_dim: Data dimension
            class_labels: Class labels (batch_size,) or None
            guidance_scale: Classifier-Free Guidance strength (1.0 means no guidance)
            
        Returns:
            Sampled data (batch_size, data_dim)
        """
        pass
    
    def _classifier_free_guidance(self, noise_pred, noise_pred_uncond, guidance_scale):
        """
        Apply Classifier-Free Guidance
        
        Args:
            noise_pred: Conditional prediction (batch_size, data_dim)
            noise_pred_uncond: Unconditional prediction (batch_size, data_dim)
            guidance_scale: Guidance strength
            
        Returns:
            Adjusted noise prediction
        """
        return noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
