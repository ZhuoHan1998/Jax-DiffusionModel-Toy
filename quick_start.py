"""
Quick Start Example - Minimal code demonstrating all core features
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torch.utils.data import DataLoader

# ============================================================
# 1. Load and prepare data
# ============================================================
from data import SwissRoll

dataset = SwissRoll(n_samples=1000, height=21, noise=0.1, seed=42)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
print(f"✓ Dataset: {len(dataset)} samples")


# ============================================================
# 2. Initialize diffusion process and model
# ============================================================
from diffusion import GaussianDiffusion
from models import SimpleUNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

diffusion = GaussianDiffusion(timesteps=1000)
model = SimpleUNet(
    data_dim=3,
    time_emb_dim=128,
    cond_emb_dim=128,
    hidden_dims=[64, 128, 256, 128, 64]
).to(device)

print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# 3. Train model
# ============================================================
from train import DiffusionTrainer

trainer = DiffusionTrainer(model, diffusion, device=device)
trainer.setup_optimizer(learning_rate=1e-3)

# Train only 5 epochs as quick demonstration
print("\nTraining model (5 epochs example)...")
trainer.train(train_loader, num_epochs=5, save_every=5)


# ============================================================
# 4. Generate samples
# ============================================================
from samplers import DDIMSampler
from inference import DiffusionInference

# Use DDIM sampling (faster)
sampler = DDIMSampler(diffusion, model, device=device, num_steps=50, eta=0.0)
inference = DiffusionInference(sampler, device=device)

# Generate samples
samples = inference.generate(batch_size=500, data_dim=3)
print(f"✓ Generation complete! Sample shape: {samples.shape}")


# ============================================================
# 5. Visualize results
# ============================================================
real_data = dataset.data

# Visualize if matplotlib is available
try:
    inference.plot_samples_3d(
        samples,
        real_data=real_data,
        title='Swiss Roll',
        save_path='quick_start_result.png'
    )
    print("✓ Results saved: quick_start_result.png")
except Exception as e:
    print(f"Visualization failed: {e}")

print("\n✓ Quick start complete!")
