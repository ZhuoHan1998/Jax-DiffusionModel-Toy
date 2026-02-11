"""
Complete Diffusion Model tutorial and demonstration script
Demonstrates how to use all core features: data loading, training, and sampling
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data import SwissRoll, Gaussian, Sinusoid
from diffusion import GaussianDiffusion
from models import SimpleUNet
from samplers import DDPMSampler, DDIMSampler
from train import DiffusionTrainer
from inference import DiffusionInference


def visualize_forward_diffusion():
    """Visualize forward diffusion process"""
    print("\n" + "="*60)
    print("1. Visualize Forward Diffusion Process")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create data and diffusion model
    swiss_roll = SwissRoll(n_samples=500, height=21, noise=0.1, seed=42)
    diffusion = GaussianDiffusion(timesteps=1000, beta_schedule='linear')
    diffusion.to(device)
    
    # Get samples
    x_0 = torch.tensor(swiss_roll.data, dtype=torch.float32).to(device)
    
    # Add noise at different timesteps
    timesteps_to_show = [0, 250, 500, 750, 999]
    fig = plt.figure(figsize=(15, 3))
    
    for idx, t in enumerate(timesteps_to_show):
        t_batch = torch.full((len(x_0),), t, dtype=torch.long, device=device)
        x_t, noise = diffusion.q_sample(x_0, t_batch)
        
        ax = fig.add_subplot(1, 5, idx + 1, projection='3d')
        x_t_np = x_t.cpu().numpy()
        scatter = ax.scatter(
            x_t_np[:, 0], x_t_np[:, 1], x_t_np[:, 2],
            c=np.arange(len(x_t_np)), cmap='viridis', s=3, alpha=0.6
        )
        ax.set_title(f't={t}')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)
    
    plt.suptitle('Forward Diffusion Process - Swiss Roll', fontsize=14)
    plt.tight_layout()
    plt.savefig('pics/forward_diffusion.png', dpi=100)
    print("✓ Forward diffusion image saved: pics/forward_diffusion.png")
    plt.close()


def train_model_example():
    """Complete example of training a model"""
    print("\n" + "="*60)
    print("2. Training Diffusion Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Data preparation
    print("\nPreparing data...")
    dataset = SwissRoll(n_samples=3000, height=21, noise=0.1, seed=42)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    print(f"✓ Dataset loaded: {len(dataset)} samples")
    
    # Initialize diffusion and model
    print("\nInitializing diffusion process...")
    diffusion = GaussianDiffusion(
        timesteps=1000,
        beta_schedule='linear',
        beta_start=0.0001,
        beta_end=0.02
    )
    
    print("Creating denoising network...")
    model = SimpleUNet(
        data_dim=3,
        time_emb_dim=128,
        cond_emb_dim=128,
        num_classes=None,
        hidden_dims=[64, 128, 256, 128, 64]
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters")
    
    # Training
    print("\nStarting training...")
    trainer = DiffusionTrainer(model, diffusion, device=device)
    trainer.setup_optimizer(learning_rate=1e-3)
    trainer.setup_scheduler(mode='cosine', num_epochs=30)
    
    trainer.train(
        train_loader,
        num_epochs=30,
        save_every=10
    )
    
    # Save loss curve
    trainer.plot_loss(save_path='training_loss.png')
    print("✓ Training completed, loss curve saved: training_loss.png")
    
    return model, diffusion


def sample_and_visualize(model, diffusion):
    """Sample and visualize using trained model"""
    print("\n" + "="*60)
    print("3. Sampling and Generation Results")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reload original data for comparison
    swiss_roll = SwissRoll(n_samples=3000, height=21, noise=0.1, seed=42)
    real_data = swiss_roll.data
    
    # DDPM sampling
    print("\nUsing DDPM sampling (1000 steps)...")
    ddpm_sampler = DDPMSampler(diffusion, model, device=device)
    ddpm_inference = DiffusionInference(ddpm_sampler, device=device)
    
    samples_ddpm = ddpm_inference.generate(batch_size=1000, data_dim=3)
    print(f"✓ DDPM generation completed: {samples_ddpm.shape}")
    
    # DDIM sampling (accelerated)
    print("\nUsing DDIM sampling (50 steps)...")
    ddim_sampler = DDIMSampler(diffusion, model, device=device, num_steps=50, eta=0.0)
    ddim_inference = DiffusionInference(ddim_sampler, device=device)
    
    samples_ddim = ddim_inference.generate(batch_size=1000, data_dim=3)
    print(f"✓ DDIM generation completed: {samples_ddim.shape}")
    
    # Visualization comparison
    print("\nGenerating visualizations...")
    
    fig = plt.figure(figsize=(15, 5))
    
    # Real data
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(real_data[:, 0], real_data[:, 1], real_data[:, 2], s=3, alpha=0.6)
    ax1.set_title('Real Data')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # DDPM generated
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(samples_ddpm[:, 0], samples_ddpm[:, 1], samples_ddpm[:, 2], s=3, alpha=0.6)
    ax2.set_title('DDPM Generated (1000 steps)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # DDIM generated
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(samples_ddim[:, 0], samples_ddim[:, 1], samples_ddim[:, 2], s=3, alpha=0.6)
    ax3.set_title('DDIM Generated (50 steps)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('pics/sampling_comparison.png', dpi=100)
    print("✓ Sampling comparison image saved: pics/sampling_comparison.png")
    plt.close()
    
    # Evaluation
    print("\nEvaluating generation quality...")
    metrics_ddpm = ddpm_inference.evaluate_fid_like(samples_ddpm, real_data)
    metrics_ddim = ddim_inference.evaluate_fid_like(samples_ddim, real_data)
    
    print("\nDDPM Metrics:")
    print(f"  Mean distance: {metrics_ddpm['mean_distance']:.4f}")
    print(f"  Covariance distance: {metrics_ddpm['cov_distance']:.4f}")
    
    print("\nDDIM Metrics:")
    print(f"  Mean distance: {metrics_ddim['mean_distance']:.4f}")
    print(f"  Covariance distance: {metrics_ddim['cov_distance']:.4f}")


def demo_different_datasets():
    """Demonstrate generation on different datasets"""
    print("\n" + "="*60)
    print("4. Generation Demo on Different Datasets")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create simple config for quick training demo
    configs = [
        {
            'name': 'Gaussian 2D',
            'dataset': Gaussian(n_samples=2000, dim=2, seed=42),
            'data_dim': 2,
            'hidden_dims': [32, 64, 128, 64, 32],
            'num_epochs': 20
        },
        {
            'name': 'Sinusoid',
            'dataset': Sinusoid(n_samples=2000, freq=2.0, noise=0.1, seed=42),
            'data_dim': 2,
            'hidden_dims': [32, 64, 128, 64, 32],
            'num_epochs': 20
        }
    ]
    
    for config in configs:
        print(f"\nProcessing dataset: {config['name']}")
        
        # Data preparation
        train_loader = DataLoader(config['dataset'], batch_size=32, shuffle=True)
        
        # Create model
        diffusion = GaussianDiffusion(timesteps=1000)
        model = SimpleUNet(
            data_dim=config['data_dim'],
            time_emb_dim=128,
            cond_emb_dim=128,
            num_classes=None,
            hidden_dims=config['hidden_dims']
        ).to(device)
        
        # Train
        print(f"  Training ({config['num_epochs']} epochs)...")
        trainer = DiffusionTrainer(model, diffusion, device=device)
        trainer.setup_optimizer(learning_rate=1e-3)
        trainer.setup_scheduler(mode='cosine', num_epochs=config['num_epochs'])
        trainer.train(train_loader, num_epochs=config['num_epochs'], save_every=10)
        
        # Sample
        print(f"  Generating samples...")
        ddim_sampler = DDIMSampler(diffusion, model, device=device, num_steps=30, eta=0.0)
        inference = DiffusionInference(ddim_sampler, device=device)
        
        samples = inference.generate(batch_size=500, data_dim=config['data_dim'])
        
        # Visualization
        real_data = config['dataset'].data
        if config['data_dim'] == 2:
            inference.plot_samples_2d(
                samples, 
                real_data=real_data,
                title=config['name'],
                save_path=f'{config["name"].replace(" ", "_").lower()}.png'
            )


def main():
    """Main program"""
    print("\n" + "="*60)
    print("Complete Diffusion Model Demo")
    print("="*60)
    
    # 1. Visualize forward diffusion
    visualize_forward_diffusion()
    
    # 2. Train model
    model, diffusion = train_model_example()
    
    # 3. Sample and visualize
    sample_and_visualize(model, diffusion)
    
    # 4. (Optional) Demo on different datasets
    # demo_different_datasets()
    
    print("\n" + "="*60)
    print("Demo Completed!")
    print("="*60)
    print("\nGenerated files:")
    print("  - forward_diffusion.png: Forward diffusion process")
    print("  - training_loss.png: Training loss curve")
    print("  - sampling_comparison.png: Sampling comparison")
    print("\nCheckpoints saved in: checkpoints/")


if __name__ == '__main__':
    main()
