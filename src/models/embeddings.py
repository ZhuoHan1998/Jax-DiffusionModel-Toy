import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for time steps"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer(
            'inv_freq',
            1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        )
    
    def forward(self, t):
        """
        Args:
            t: Time step tensor (batch_size,)
            
        Returns:
            Time encoding (batch_size, dim)
        """
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        return emb


class ConditionEmbedding(nn.Module):
    """Condition encoding (for classifier-free guidance)"""
    
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, dim)
    
    def forward(self, class_labels):
        """
        Args:
            class_labels: Class labels tensor (batch_size,)
            
        Returns:
            Condition encoding (batch_size, dim)
        """
        return self.embedding(class_labels)


class MLPEmbedding(nn.Module):
    """Project encoding to higher dimensional representation"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 4),
            nn.SiLU(),
            nn.Linear(out_dim * 4, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)
