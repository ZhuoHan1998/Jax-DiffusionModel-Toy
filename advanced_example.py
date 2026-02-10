"""
Advanced usage examples - Demonstrates how to customize and extend the framework

This script demonstrates:
1. Custom datasets
2. Custom samplers
3. Mixed precision training
4. Multi-GPU support
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from diffusion import GaussianDiffusion
from models import SimpleUNet
from samplers import BaseSampler
from train import DiffusionTrainer
from inference import DiffusionInference


# ============================================================
# 1. Custom Dataset Example
# ============================================================
class CustomDataset(Dataset):
    """Custom dataset - mixture of multiple Gaussian distributions"""
    
    def __init__(self, n_samples=1000, n_clusters=5, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.data = []
        samples_per_cluster = n_samples // n_clusters
        
        for i in range(n_clusters):
            # Each cluster has a different mean
            angle = 2 * np.pi * i / n_clusters
            mean = np.array([3 * np.cos(angle), 3 * np.sin(angle)])
            
            # Generate data for this cluster
            cluster_data = np.random.randn(samples_per_cluster, 2) * 0.5 + mean
            self.data.append(cluster_data)
        
        self.data = np.vstack(self.data).astype(np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


# ============================================================
# 2. Custom Sampler Example
# ============================================================
class EulerSampler(BaseSampler):
    """Simplified Euler sampler - uses first-order Taylor expansion"""
    
    def __init__(self, diffusion_model, denoising_model, device='cpu', num_steps=50):
        super().__init__(diffusion_model, denoising_model, device)
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps
    
    def sample(self, batch_size, data_dim, class_labels=None, guidance_scale=1.0):
        """Euler sampling"""
        # Start from noise
        x = torch.randn(batch_size, data_dim, device=self.device)
        
        # Reverse from t=1 to t=0
        for step in range(self.num_steps):
            # Current timestep
            t = int((1.0 - step / self.num_steps) * (self.diffusion.timesteps - 1))
            t_batch = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            
            with torch.no_grad():
                # Predict noise
                noise_pred = self.model(x, t_batch, class_labels)
                
                if class_labels is not None and guidance_scale > 1.0:
                    noise_cond = self.model(x, t_batch, class_labels)
                    noise_uncond = self.model(x, t_batch, None)
                    noise_pred = self._classifier_free_guidance(
                        noise_cond, noise_uncond, guidance_scale
                    )
                
                # Euler step
                x = x - self.dt * noise_pred
        
        return x


# ============================================================
# 3. Custom UNet Example
# ============================================================
from models.embeddings import TimeEmbedding, MLPEmbedding

class CustomUNet(nn.Module):
    """Custom deeper UNet"""
    
    def __init__(self, data_dim, timesteps, num_classes=None):
        super().__init__()
        
        self.data_dim = data_dim
        self.timesteps = timesteps
        
        # Time embedding
        self.time_embedding = TimeEmbedding(128)
        self.time_mlp = MLPEmbedding(128, 256)
        
        # Condition embedding (optional)
        self.num_classes = num_classes
        if num_classes is not None:
            self.cond_embedding = nn.Embedding(num_classes, 128)
            self.cond_mlp = MLPEmbedding(128, 256)
        
        # Main network
        self.net = nn.Sequential(
            nn.Linear(data_dim + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, data_dim)
        )
    
    def forward(self, x, t, class_labels=None):
        # Time encoding
        t_emb = self.time_embedding(t.float())
        t_emb = self.time_mlp(t_emb)
        
        # Condition encoding
        cond_emb = None
        if class_labels is not None and self.num_classes is not None:
            cond_emb = self.cond_embedding(class_labels)
            cond_emb = self.cond_mlp(cond_emb)
            t_emb = t_emb + cond_emb
        
        # Concatenate input and encoding
        x_emb = torch.cat([x, t_emb], dim=-1)
        
        # Forward pass
        out = self.net(x_emb)
        return out


# ============================================================
# 4. Complete Advanced Training Example
# ============================================================
def advanced_training_example():
    """Demonstrate advanced training features"""
    
    print("\n" + "="*60)
    print("Advanced Usage Example")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dim = 2
    
    # 1. Use custom dataset
    print("\n1. Creating custom dataset...")
    dataset = CustomDataset(n_samples=1000, n_clusters=5, seed=42)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"✓ Custom dataset: {len(dataset)} samples")
    
    # 2. Initialize diffusion and model
    print("\n2. Initializing models...")
    diffusion = GaussianDiffusion(timesteps=1000)
    model = CustomUNet(data_dim=data_dim, timesteps=1000).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Custom model: {num_params:,} parameters")
    
    # 3. Train
    print("\n3. Training model...")
    trainer = DiffusionTrainer(model, diffusion, device=device)
    trainer.setup_optimizer(learning_rate=1e-3)
    trainer.setup_scheduler(mode='cosine', num_epochs=20)
    
    trainer.train(train_loader, num_epochs=20, save_every=10)
    
    # 4. Use custom sampler
    print("\n4. Using custom Euler sampler...")
    euler_sampler = EulerSampler(
        diffusion, model, device=device, num_steps=50
    )
    euler_inference = DiffusionInference(euler_sampler, device=device)
    
    samples_euler = euler_inference.generate(batch_size=500, data_dim=data_dim)
    print(f"✓ Euler sampling complete: {samples_euler.shape}")
    
    # 5. Visualization
    print("\n5. Visualizing results...")
    real_data = dataset.data
    
    euler_inference.plot_samples_2d(
        samples_euler,
        real_data=real_data,
        title='Gaussian Mixture (Custom Sampler)',
        save_path='advanced_example.png'
    )
    
    print("\n✓ Advanced example complete!")


# ============================================================
# 5. Performance Optimization Example
# ============================================================
def performance_tips():
    """Performance optimization recommendations"""
    print("\n" + "="*60)
    print("Performance Optimization Tips")
    print("="*60)
    
    tips = """
    1. Sampling acceleration:
       - Use DDIM instead of DDPM: 20x speedup (50 steps vs 1000 steps)
       - Adjust num_steps parameter to balance quality and speed
    
    2. Memory optimization:
       - Reduce hidden_dims to reduce model size
       - Increase batch_size to improve GPU utilization
       - Use gradient accumulation for larger effective batch sizes
    
    3. Training optimization:
       - Use cosine learning rate schedule instead of constant
       - Apply gradient clipping to prevent exploding gradients
       - Use warmup phase to stabilize early training
    
    4. Multi-GPU training:
       - Use DataParallel: model = nn.DataParallel(model)
       - Or use DistributedDataParallel for better scaling
    
    5. Mixed precision training:
       - Use torch.cuda.amp.autocast() to accelerate computation
       - Use GradScaler to handle gradient overflow
    """
    
    print(tips)


if __name__ == '__main__':
    # Run advanced example
    advanced_training_example()
    
    # Print performance optimization tips
    performance_tips()
