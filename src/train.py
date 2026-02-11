import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from datetime import datetime
import json


class DiffusionTrainer:
    """
    Diffusion Model Training class
    """
    
    def __init__(self, model, diffusion, device='cpu', checkpoint_dir='checkpoints'):
        """
        Args:
            model: Denoising network
            diffusion: GaussianDiffusion instance
            device: Computation device
            checkpoint_dir: Checkpoint save directory
        """
        self.model = model.to(device)
        self.diffusion = diffusion.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.optimizer = None
        self.scheduler = None
        self.loss_history = []
    
    def setup_optimizer(self, learning_rate=1e-3, weight_decay=0.0):
        """
        Setup optimizer
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay
        """
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    def setup_scheduler(self, mode='cosine', num_epochs=100):
        """
        Setup learning rate scheduler
        
        Args:
            mode: Scheduler type ('cosine', 'linear')
            num_epochs: Total number of training epochs
        """
        if mode == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs
            )
        elif mode == 'linear':
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, total_iters=num_epochs
            )
    
    def _loss_fn(self, x_0, t, class_labels=None):
        """
        Calculate loss function
        Supports two forms of loss:
        1. Predict noise (epsilon prediction)
        2. Predict x_0 (x_0 prediction)
        
        Args:
            x_0: Original data (batch_size, data_dim)
            t: Time step (batch_size,)
            class_labels: Class labels
            
        Returns:
            Loss value
        """
        # Generate random noise
        noise = torch.randn_like(x_0)
        
        # Forward process: add noise
        x_t, _ = self.diffusion.q_sample(x_0, t, noise)
        
        # Network prediction
        noise_pred = self.model(x_t, t, class_labels)
        
        # Calculate loss (predict noise)
        loss = nn.MSELoss()(noise_pred, noise)
        
        return loss
    
    def train_epoch(self, train_loader, num_classes=None):
        """
        Train one epoch
        
        Args:
            train_loader: Data loader
            num_classes: Number of classes (for conditional generation)
            
        Returns:
            Average loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x_0 = batch[0].to(self.device)
                class_labels = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                x_0 = batch.to(self.device)
                class_labels = None
            
            batch_size = x_0.shape[0]
            
            # Randomly sample timesteps
            t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device)
            
            # Calculate loss
            loss = self._loss_fn(x_0, t, class_labels)
            
            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, num_epochs, num_classes=None, save_every=10):
        """
        Train model
        
        Args:
            train_loader: Training data loader
            num_epochs: Number of training epochs
            num_classes: Number of classes
            save_every: Save checkpoint every N epochs
        """
        if self.optimizer is None:
            self.setup_optimizer()
        
        if self.scheduler is not None:
            self.scheduler = None
        self.setup_scheduler(num_epochs=num_epochs)
        
        print(f"Starting training... (device={self.device})")
        
        for epoch in range(num_epochs):
            loss = self.train_epoch(train_loader, num_classes)
            self.loss_history.append(loss)
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
                self.save_checkpoint(epoch + 1)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.6f}")
        
        print("Training completed!")
    
    def save_checkpoint(self, epoch):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'loss_history': self.loss_history,
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_history = checkpoint.get('loss_history', [])
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def plot_loss(self, save_path=None):
        """Plot loss curve"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Loss curve saved: {save_path}")
            plt.close()
        else:
            # In headless mode, still save to a default path
            default_path = 'pics/loss_curve.png'
            plt.savefig(default_path)
            print(f"Loss curve saved: {default_path}")
            plt.close()
