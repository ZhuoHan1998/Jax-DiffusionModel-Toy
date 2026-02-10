import torch
import matplotlib.pyplot as plt
import numpy as np


class DiffusionInference:
    """Inference utility class"""
    
    def __init__(self, sampler, device='cpu'):
        """
        Args:
            sampler: Sampler instance (DDPMSampler or DDIMSampler)
            device: Computation device
        """
        self.sampler = sampler
        self.device = device
    
    def generate(self, batch_size, data_dim, class_labels=None, guidance_scale=1.0):
        """
        Generate samples
        
        Args:
            batch_size: Batch size
            data_dim: Data dimension
            class_labels: Class labels (batch_size,) or None
            guidance_scale: Classifier-Free Guidance strength
            
        Returns:
            Generated data (batch_size, data_dim)
        """
        samples = self.sampler.sample(
            batch_size=batch_size,
            data_dim=data_dim,
            class_labels=class_labels,
            guidance_scale=guidance_scale
        )
        return samples.detach().cpu().numpy()
    
    def plot_samples_2d(self, samples, real_data=None, title='Generated Samples', save_path=None):
        """
        Plot 2D samples
        
        Args:
            samples: Generated samples (n_samples, 2)
            real_data: Real data (n_samples, 2), optional
            title: Chart title
            save_path: Save path, None means not to save
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot generated samples
        axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
        axes[0].set_title(f'{title} (Generated)')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].grid(True, alpha=0.3)
        
        # If real data available, plot comparison
        if real_data is not None:
            axes[1].scatter(real_data[:, 0], real_data[:, 1], alpha=0.5, s=10)
            axes[1].set_title(f'{title} (Real)')
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Chart saved: {save_path}")
        else:
            plt.show()
    
    def plot_samples_3d(self, samples, real_data=None, title='Generated Samples', save_path=None):
        """
        Plot 3D samples
        
        Args:
            samples: Generated samples (n_samples, 3)
            real_data: Real data (n_samples, 3), optional
            title: Chart title
            save_path: Save path
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 5))
        
        # Generated samples
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(samples[:, 0], samples[:, 1], samples[:, 2], alpha=0.5, s=3)
        ax1.set_title(f'{title} (Generated)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Real data comparison
        if real_data is not None:
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(real_data[:, 0], real_data[:, 1], real_data[:, 2], alpha=0.5, s=3)
            ax2.set_title(f'{title} (Real)')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Chart saved: {save_path}")
        else:
            plt.show()
    
    def evaluate_fid_like(self, generated_samples, real_samples):
        """
        Calculate FID-like metrics (simplified version)
        
        Args:
            generated_samples: Generated samples (n, dim)
            real_samples: Real samples (n, dim)
            
        Returns:
            Metrics dictionary
        """
        gen_mean = generated_samples.mean(axis=0)
        gen_cov = np.cov(generated_samples.T)
        
        real_mean = real_samples.mean(axis=0)
        real_cov = np.cov(real_samples.T)
        
        # Calculate mean distance
        mean_distance = np.linalg.norm(gen_mean - real_mean)
        
        # Calculate covariance distance
        cov_distance = np.linalg.norm(gen_cov - real_cov, 'fro')
        
        return {
            'mean_distance': float(mean_distance),
            'cov_distance': float(cov_distance),
            'generated_mean': gen_mean.tolist(),
            'real_mean': real_mean.tolist(),
        }
