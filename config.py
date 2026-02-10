"""
Example configuration file demonstrating how to use different configurations
"""

# Swiss Roll Configuration
SWISS_ROLL_CONFIG = {
    'dataset': {
        'type': 'SwissRoll',
        'n_samples': 5000,
        'height': 21,
        'noise': 0.1,
        'seed': 42
    },
    'diffusion': {
        'timesteps': 1000,
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02
    },
    'model': {
        'data_dim': 3,
        'time_emb_dim': 128,
        'cond_emb_dim': 128,
        'hidden_dims': [64, 128, 256, 128, 64]
    },
    'training': {
        'batch_size': 64,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'weight_decay': 0.0,
        'scheduler': 'cosine'
    },
    'sampling': {
        'ddpm_steps': 1000,
        'ddim_steps': 50,
        'ddim_eta': 0.0,
        'guidance_scale': 1.0
    }
}

# Gaussian Configuration (for quick testing)
GAUSSIAN_CONFIG = {
    'dataset': {
        'type': 'Gaussian',
        'n_samples': 2000,
        'dim': 2,
        'seed': 42
    },
    'diffusion': {
        'timesteps': 1000,
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02
    },
    'model': {
        'data_dim': 2,
        'time_emb_dim': 128,
        'cond_emb_dim': 128,
        'hidden_dims': [32, 64, 128, 64, 32]
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'weight_decay': 0.0,
        'scheduler': 'cosine'
    },
    'sampling': {
        'ddpm_steps': 1000,
        'ddim_steps': 30,
        'ddim_eta': 0.0,
        'guidance_scale': 1.0
    }
}

# Sinusoid Configuration
SINUSOID_CONFIG = {
    'dataset': {
        'type': 'Sinusoid',
        'n_samples': 2000,
        'freq': 2.0,
        'phase': 0.0,
        'noise': 0.1,
        'seed': 42
    },
    'diffusion': {
        'timesteps': 1000,
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.02
    },
    'model': {
        'data_dim': 2,
        'time_emb_dim': 128,
        'cond_emb_dim': 128,
        'hidden_dims': [32, 64, 128, 64, 32]
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_epochs': 50,
        'weight_decay': 0.0,
        'scheduler': 'cosine'
    },
    'sampling': {
        'ddpm_steps': 1000,
        'ddim_steps': 30,
        'ddim_eta': 0.0,
        'guidance_scale': 1.0
    }
}
