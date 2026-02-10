import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os

# Add source code path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data import SwissRoll, Gaussian
from diffusion import GaussianDiffusion
from models import SimpleUNet
from samplers import DDPMSampler, DDIMSampler
from train import DiffusionTrainer
from inference import DiffusionInference


def main():
    """Main training script"""
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dim = 3  # Swiss Roll is 3D
    batch_size = 64
    learning_rate = 1e-3
    num_epochs = 100
    
    print(f"Device: {device}")
    print(f"Data dimension: {data_dim}")
    
    # 1. Create dataset
    print("\n1. Creating dataset...")
    dataset = SwissRoll(n_samples=5000, height=21, noise=0.1, seed=42)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)}")
    
    # 2. Initialize diffusion model
    print("\n2. Initializing diffusion model...")
    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02
    )
    
    # 3. Create network model
    print("\n3. Creating network model...")
    model = SimpleUNet(
        data_dim=data_dim,
        time_emb_dim=128,
        cond_emb_dim=128,
        num_classes=None,  # For unconditional generation
        hidden_dims=[64, 128, 256, 128, 64]
    )
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # 4. Initialize trainer
    print("\n4. Initializing trainer...")
    trainer = DiffusionTrainer(model, diffusion, device=device)
    trainer.setup_optimizer(learning_rate=learning_rate)
    trainer.setup_scheduler(mode='cosine', num_epochs=num_epochs)
    
    # 5. Train model
    print("\n5. Starting training...")
    trainer.train(
        train_loader,
        num_epochs=num_epochs,
        save_every=10
    )
    
    # 6. Save loss curve
    trainer.plot_loss(save_path='loss_curve.png')
    
    # 7. Generate samples
    print("\n6. Generating samples...")
    
    # Use DDPM sampler
    ddpm_sampler = DDPMSampler(diffusion, model, device=device)
    ddpm_inference = DiffusionInference(ddpm_sampler, device=device)
    
    samples_ddpm = ddpm_inference.generate(batch_size=1000, data_dim=data_dim)
    print(f"DDPM generated samples shape: {samples_ddpm.shape}")
    
    # Use DDIM sampler (faster)
    ddim_sampler = DDIMSampler(diffusion, model, device=device, num_steps=50, eta=0.0)
    ddim_inference = DiffusionInference(ddim_sampler, device=device)
    
    samples_ddim = ddim_inference.generate(batch_size=1000, data_dim=data_dim)
    print(f"DDIM generated samples shape: {samples_ddim.shape}")
    
    # 8. Visualize results
    print("\n7. Visualizing results...")
    real_data = dataset.data
    
    ddpm_inference.plot_samples_3d(
        samples_ddpm,
        real_data=real_data,
        title='Swiss Roll',
        save_path='swiss_roll_ddpm.png'
    )
    
    ddim_inference.plot_samples_3d(
        samples_ddim,
        real_data=real_data,
        title='Swiss Roll (DDIM)',
        save_path='swiss_roll_ddim.png'
    )
    
    # 9. Evaluate
    print("\n8. Evaluating generation quality...")
    metrics_ddpm = ddpm_inference.evaluate_fid_like(samples_ddpm, real_data)
    metrics_ddim = ddim_inference.evaluate_fid_like(samples_ddim, real_data)
    
    print("\nDDPM Metrics:")
    print(f"  Mean distance: {metrics_ddpm['mean_distance']:.4f}")
    print(f"  Covariance distance: {metrics_ddpm['cov_distance']:.4f}")
    
    print("\nDDIM Metrics:")
    print(f"  Mean distance: {metrics_ddim['mean_distance']:.4f}")
    print(f"  Covariance distance: {metrics_ddim['cov_distance']:.4f}")


if __name__ == '__main__':
    main()
