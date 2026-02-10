"""
Example of conditional generation with Classifier-Free Guidance
Demonstrates how to use class labels for conditional generation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from diffusion import GaussianDiffusion
from models import SimpleUNet
from samplers import DDIMSampler
from inference import DiffusionInference


class ConditionalGaussianDataset(Dataset):
    """Gaussian dataset with class labels"""
    
    def __init__(self, n_samples=2000, num_classes=3, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        self.num_classes = num_classes
        self.data = []
        self.labels = []
        
        # Generate Gaussian distributed data for each class
        samples_per_class = n_samples // num_classes
        for class_id in range(num_classes):
            # Each class has a different mean
            mean = torch.tensor([
                torch.cos(2 * 3.14159 * class_id / num_classes),
                torch.sin(2 * 3.14159 * class_id / num_classes)
            ]) * 2.0
            
            # Generate data for this class
            samples = torch.randn(samples_per_class, 2) * 0.5 + mean
            self.data.append(samples)
            self.labels.extend([class_id] * samples_per_class)
        
        self.data = torch.cat(self.data)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def train_conditional_model(device='cpu'):
    """Train conditional diffusion model"""
    
    from train import DiffusionTrainer
    
    print("=" * 50)
    print("Conditional Diffusion Model Training")
    print("=" * 50)
    
    # Parameters
    num_classes = 3
    batch_size = 32
    num_epochs = 50
    
    # Create dataset
    print("\n1. Creating conditional dataset...")
    dataset = ConditionalGaussianDataset(n_samples=1000, num_classes=num_classes, seed=42)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)}, Number of classes: {num_classes}")
    
    # Initialize diffusion and model
    print("\n2. Initializing diffusion model...")
    diffusion = GaussianDiffusion(timesteps=1000)
    
    print("3. Creating conditional network...")
    model = SimpleUNet(
        data_dim=2,
        time_emb_dim=128,
        cond_emb_dim=128,
        num_classes=num_classes,  # Important: specify number of classes
        hidden_dims=[32, 64, 128, 64, 32]
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Training
    print("\n4. Starting training...")
    trainer = DiffusionTrainer(model, diffusion, device=device)
    trainer.setup_optimizer(learning_rate=1e-3)
    trainer.setup_scheduler(mode='cosine', num_epochs=num_epochs)
    trainer.train(train_loader, num_epochs=num_epochs, save_every=10)
    
    # Save checkpoint
    checkpoint_path = 'checkpoints/conditional_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_classes': num_classes,
    }, checkpoint_path)
    print(f"\nModel saved: {checkpoint_path}")
    
    return model, diffusion, num_classes, dataset


def generate_conditional_samples(model, diffusion, num_classes, device='cpu'):
    """Generate samples using conditional generation"""
    
    print("\n" + "=" * 50)
    print("Conditional Generation Sampling")
    print("=" * 50)
    
    # Create sampler
    sampler = DDIMSampler(diffusion, model, device=device, num_steps=50, eta=0.0)
    inference = DiffusionInference(sampler, device=device)
    
    # Generate samples for each class
    batch_size = 100
    samples_by_class = {}
    
    for class_id in range(num_classes):
        print(f"\nGenerating samples for class {class_id} (no guidance)...")
        
        # Create class labels
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        # Generate samples
        samples = inference.generate(
            batch_size=batch_size,
            data_dim=2,
            class_labels=class_labels,
            guidance_scale=1.0  # No guidance
        )
        samples_by_class[f'class_{class_id}_no_guidance'] = samples
        
        # Regenerate with guidance
        print(f"Generating samples for class {class_id} (with guidance)...")
        samples_guided = inference.generate(
            batch_size=batch_size,
            data_dim=2,
            class_labels=class_labels,
            guidance_scale=7.5  # Strong guidance
        )
        samples_by_class[f'class_{class_id}_with_guidance'] = samples_guided
    
    return samples_by_class, inference


def visualize_conditional_generation(samples_by_class, inference, dataset):
    """Visualize conditional generation results"""
    
    import matplotlib.pyplot as plt
    
    print("\n" + "=" * 50)
    print("Visualization Results")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Conditional Diffusion Generation - Gaussian Mixture', fontsize=16)
    
    num_classes = 3
    
    for class_id in range(num_classes):
        # No guidance
        ax_no = axes[0, class_id]
        samples_no = samples_by_class[f'class_{class_id}_no_guidance']
        ax_no.scatter(samples_no[:, 0], samples_no[:, 1], alpha=0.5, s=20, label=f'Generated (No Guidance)')
        
        # Add real data
        mask = dataset.labels == class_id
        ax_no.scatter(dataset.data[mask, 0], dataset.data[mask, 1], alpha=0.5, s=20, 
                     label='Real Data', marker='x')
        ax_no.set_title(f'Class {class_id} (No Guidance)')
        ax_no.legend()
        ax_no.grid(True, alpha=0.3)
        ax_no.set_xlim(-3, 3)
        ax_no.set_ylim(-3, 3)
        
        # With guidance
        ax_with = axes[1, class_id]
        samples_with = samples_by_class[f'class_{class_id}_with_guidance']
        ax_with.scatter(samples_with[:, 0], samples_with[:, 1], alpha=0.5, s=20, label=f'Generated (With Guidance)')
        
        # Add real data
        ax_with.scatter(dataset.data[mask, 0], dataset.data[mask, 1], alpha=0.5, s=20,
                       label='Real Data', marker='x')
        ax_with.set_title(f'Class {class_id} (With Guidance, scale=7.5)')
        ax_with.legend()
        ax_with.grid(True, alpha=0.3)
        ax_with.set_xlim(-3, 3)
        ax_with.set_ylim(-3, 3)
    
    plt.tight_layout()
    plt.savefig('conditional_generation.png', dpi=150)
    print("Visualization results saved: conditional_generation.png")
    plt.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train conditional model
    model, diffusion, num_classes, dataset = train_conditional_model(device=device)
    
    # Generate conditional samples
    samples_by_class, inference = generate_conditional_samples(
        model, diffusion, num_classes, device=device
    )
    
    # Visualization
    visualize_conditional_generation(samples_by_class, inference, dataset)


if __name__ == '__main__':
    main()
