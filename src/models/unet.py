import torch
import torch.nn as nn
from .embeddings import TimeEmbedding, MLPEmbedding


class ResidualBlock(nn.Module):
    """Residual block with time and condition modulation"""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim=None):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        
        self.cond_mlp = None
        if cond_emb_dim is not None:
            self.cond_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_emb_dim, out_channels * 2)
            )
        
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels) if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 3, padding=1)
        )
        
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )
        
        self.res_conv = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)
    
    def forward(self, x, time_emb, cond_emb=None):
        h = self.block1(x)
        
        # Apply time embedding modulation
        time_scale = self.mlp(time_emb)
        scale, shift = time_scale.chunk(2, dim=1)
        
        if len(x.shape) == 2:  # (batch, features)
            scale = scale.unsqueeze(2)
            shift = shift.unsqueeze(2)
        else:  # (batch, channels, length)
            scale = scale.unsqueeze(2)
            shift = shift.unsqueeze(2)
        
        h = h * (1 + scale) + shift
        
        # Apply condition embedding
        if cond_emb is not None and self.cond_mlp is not None:
            cond_scale = self.cond_mlp(cond_emb)
            cond_scale, cond_shift = cond_scale.chunk(2, dim=1)
            if len(h.shape) == 3:
                cond_scale = cond_scale.unsqueeze(2)
                cond_shift = cond_shift.unsqueeze(2)
            h = h * (1 + cond_scale) + cond_shift
        
        h = self.block2(h)
        return h + self.res_conv(x)


class SimpleUNet(nn.Module):
    """
    Simplified UNet architecture suitable for low-dimensional toy data.
    Supports time embedding and condition embedding (classifier-free guidance).
    """
    
    def __init__(self, data_dim, time_emb_dim=128, cond_emb_dim=128, num_classes=None, hidden_dims=None):
        """
        Args:
            data_dim: Input data dimension
            time_emb_dim: Time embedding dimension
            cond_emb_dim: Condition embedding dimension
            num_classes: Number of classes (for conditional generation), None for unconditional
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        self.data_dim = data_dim
        self.time_emb_dim = time_emb_dim
        self.cond_emb_dim = cond_emb_dim
        self.num_classes = num_classes
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 256]
        
        self.hidden_dims = hidden_dims
        
        # Time encoding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = MLPEmbedding(time_emb_dim, time_emb_dim)
        
        # Condition encoding
        self.cond_embedding = None
        self.cond_mlp = None
        if num_classes is not None:
            self.cond_embedding = nn.Embedding(num_classes, cond_emb_dim)
            self.cond_mlp = MLPEmbedding(cond_emb_dim, cond_emb_dim)
        
        # Main network
        self.input_projection = nn.Linear(data_dim, hidden_dims[0])
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            out_dim = hidden_dims[i + 1]
            self.encoder_blocks.append(
                ResidualBlock(in_dim, out_dim, time_emb_dim, cond_emb_dim)
            )
        
        # Middle block
        self.middle_block = ResidualBlock(
            hidden_dims[-1], hidden_dims[-1], time_emb_dim, cond_emb_dim
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1, 0, -1):
            in_dim = hidden_dims[i] + hidden_dims[i - 1]  # Add skip connection
            out_dim = hidden_dims[i - 1]
            self.decoder_blocks.append(
                ResidualBlock(in_dim, out_dim, time_emb_dim, cond_emb_dim)
            )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dims[0], data_dim)
    
    def forward(self, x, t, class_labels=None):
        """
        Args:
            x: Input data (batch_size, data_dim)
            t: Time step (batch_size,) value range [0, 1000)
            class_labels: Class labels (batch_size,) or None
            
        Returns:
            Predicted noise (batch_size, data_dim)
        """
        # Time encoding
        t_emb = self.time_embedding(t.float())
        t_emb = self.time_mlp(t_emb)
        
        # Condition encoding
        cond_emb = None
        if class_labels is not None and self.cond_embedding is not None:
            cond_emb = self.cond_embedding(class_labels)
            cond_emb = self.cond_mlp(cond_emb)
        
        # Project input to hidden layer
        h = self.input_projection(x)
        
        # Encoding
        skip_connections = [h]
        for block in self.encoder_blocks:
            h = block(h, t_emb, cond_emb)
            skip_connections.append(h)
        
        # Middle
        h = self.middle_block(h, t_emb, cond_emb)
        
        # Decoding
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_connections[-(i + 2)]
            h = torch.cat([h, skip], dim=-1)
            h = block(h, t_emb, cond_emb)
        
        # Output projection
        out = self.output_projection(h)
        
        return out


class UNet(nn.Module):
    """
    Standard UNet architecture for more complex data.
    """
    
    def __init__(self, data_dim, time_emb_dim=128, cond_emb_dim=128, 
                 num_classes=None, hidden_dims=None, num_layers=4):
        """
        Args:
            data_dim: Input data dimension
            time_emb_dim: Time embedding dimension
            cond_emb_dim: Condition embedding dimension
            num_classes: Number of classes
            hidden_dims: Hidden layer dimensions
            num_layers: Number of network layers
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 256, 512, 256, 128]
        
        # Use SimpleUNet as base
        self.net = SimpleUNet(
            data_dim=data_dim,
            time_emb_dim=time_emb_dim,
            cond_emb_dim=cond_emb_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims
        )
    
    def forward(self, x, t, class_labels=None):
        return self.net(x, t, class_labels)
