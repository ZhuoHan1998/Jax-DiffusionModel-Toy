"""
Project validation script - Check if all core features are working properly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from torch.utils.data import DataLoader


def test_imports():
    """Test importing all modules"""
    print("\n" + "="*60)
    print("Test 1: Import Modules")
    print("="*60)
    
    try:
        from data import SwissRoll, Gaussian, Sinusoid
        print("✓ Data module imported successfully")
        
        from diffusion import GaussianDiffusion
        print("✓ Diffusion module imported successfully")
        
        from models import SimpleUNet, UNet
        print("✓ Model module imported successfully")
        
        from samplers import DDPMSampler, DDIMSampler
        print("✓ Sampler module imported successfully")
        
        from train import DiffusionTrainer
        print("✓ Training module imported successfully")
        
        from inference import DiffusionInference
        print("✓ Inference module imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_datasets():
    """Test dataset creation"""
    print("\n" + "="*60)
    print("Test 2: Datasets")
    print("="*60)
    
    try:
        from data import SwissRoll, Gaussian, Sinusoid
        
        # Swiss Roll
        sr = SwissRoll(n_samples=100, seed=42)
        assert sr.data.shape == (100, 3), f"Swiss Roll shape error: {sr.data.shape}"
        assert len(sr) == 100, "Swiss Roll length error"
        print(f"✓ SwissRoll: {sr.data.shape}")
        
        # Gaussian
        g = Gaussian(n_samples=100, dim=2, seed=42)
        assert g.data.shape == (100, 2), f"Gaussian shape error: {g.data.shape}"
        print(f"✓ Gaussian: {g.data.shape}")
        
        # Sinusoid
        s = Sinusoid(n_samples=100, seed=42)
        assert s.data.shape == (100, 2), f"Sinusoid shape error: {s.data.shape}"
        print(f"✓ Sinusoid: {s.data.shape}")
        
        # DataLoader
        loader = DataLoader(sr, batch_size=32)
        batch = next(iter(loader))
        assert batch.shape[0] == 32, "Batch size error"
        assert batch.shape[1] == 3, "Batch dimension error"
        print(f"✓ DataLoader: {batch.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diffusion():
    """Test diffusion process"""
    print("\n" + "="*60)
    print("Test 3: Diffusion Process")
    print("="*60)
    
    try:
        from diffusion import GaussianDiffusion
        from data import SwissRoll
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create diffusion process
        diffusion = GaussianDiffusion(timesteps=1000, beta_schedule='linear')
        diffusion.to(device)
        print("✓ GaussianDiffusion created successfully")
        
        # Test forward process
        dataset = SwissRoll(n_samples=10, seed=42)
        x_0 = torch.from_numpy(dataset.data).to(device)
        t = torch.randint(0, 1000, (10,), device=device)
        
        x_t, noise = diffusion.q_sample(x_0, t)
        assert x_t.shape == x_0.shape, "Shape of x_t error"
        assert noise.shape == x_0.shape, "Shape of noise error"
        print(f"✓ q_sample: {x_t.shape}")
        
        # Test posterior
        mean, var, log_var = diffusion.q_posterior_mean_variance(x_0, x_t, t)
        assert mean.shape == x_0.shape, "Shape of mean error"
        print(f"✓ q_posterior_mean_variance: {mean.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Diffusion process test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_models():
    """Test models"""
    print("\n" + "="*60)
    print("Test 4: Models")
    print("="*60)
    
    try:
        from models import SimpleUNet, UNet
        from data import SwissRoll
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # SimpleUNet
        model = SimpleUNet(
            data_dim=3,
            time_emb_dim=128,
            cond_emb_dim=128,
            num_classes=None,
            hidden_dims=[32, 64, 32]
        ).to(device)
        
        # Test forward pass
        dataset = SwissRoll(n_samples=4, seed=42)
        x = torch.from_numpy(dataset.data).to(device)
        t = torch.randint(0, 1000, (4,), device=device)
        
        with torch.no_grad():
            output = model(x, t, class_labels=None)
        
        assert output.shape == x.shape, f"Output shape error: {output.shape} vs {x.shape}"
        print(f"✓ SimpleUNet: {output.shape}")
        
        # Model parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model parameters: {num_params:,}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_samplers():
    """Test samplers"""
    print("\n" + "="*60)
    print("Test 5: Samplers")
    print("="*60)
    
    try:
        from samplers import DDPMSampler, DDIMSampler
        from diffusion import GaussianDiffusion
        from models import SimpleUNet
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create minimal model (quick test)
        diffusion = GaussianDiffusion(timesteps=100)  # Reduce timesteps for faster test
        model = SimpleUNet(
            data_dim=2,
            time_emb_dim=64,
            hidden_dims=[32, 64, 32]
        ).to(device)
        
        # DDIM sampler (faster)
        ddim_sampler = DDIMSampler(
            diffusion, model, 
            device=device, 
            num_steps=10,  # Reduce steps for faster test
            eta=0.0
        )
        
        with torch.no_grad():
            samples = ddim_sampler.sample(
                batch_size=4,
                data_dim=2,
                class_labels=None,
                guidance_scale=1.0
            )
        
        assert samples.shape == (4, 2), f"Sampler shape error: {samples.shape}"
        print(f"✓ DDIMSampler: {samples.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Sampler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training():
    """Test training"""
    print("\n" + "="*60)
    print("Test 6: Training")
    print("="*60)
    
    try:
        from train import DiffusionTrainer
        from diffusion import GaussianDiffusion
        from models import SimpleUNet
        from data import Gaussian
        from torch.utils.data import DataLoader
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create minimal dataset
        dataset = Gaussian(n_samples=50, dim=2, seed=42)
        train_loader = DataLoader(dataset, batch_size=10)
        
        # Create minimal model
        diffusion = GaussianDiffusion(timesteps=100)
        model = SimpleUNet(
            data_dim=2,
            time_emb_dim=64,
            hidden_dims=[32, 64, 32]
        ).to(device)
        
        # Train
        trainer = DiffusionTrainer(model, diffusion, device=device)
        trainer.setup_optimizer(learning_rate=1e-3)
        
        # Train 1 epoch
        loss = trainer.train_epoch(train_loader)
        assert loss > 0, "Loss value anomaly"
        print(f"✓ Training complete, loss: {loss:.4f}")
        
        # Checkpoint saving
        trainer.save_checkpoint(1)
        assert os.path.exists('checkpoints/checkpoint_epoch_1.pt'), "Checkpoint save failed"
        print("✓ Checkpoint saved successfully")
        
        return True
    except Exception as e:
        print(f"✗ Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test inference"""
    print("\n" + "="*60)
    print("Test 7: Inference")
    print("="*60)
    
    try:
        from inference import DiffusionInference
        from samplers import DDIMSampler
        from diffusion import GaussianDiffusion
        from models import SimpleUNet
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create minimal model
        diffusion = GaussianDiffusion(timesteps=100)
        model = SimpleUNet(
            data_dim=2,
            time_emb_dim=64,
            hidden_dims=[32, 64, 32]
        ).to(device)
        
        sampler = DDIMSampler(diffusion, model, device=device, num_steps=10)
        inference = DiffusionInference(sampler, device=device)
        
        # Generate samples
        samples = inference.generate(batch_size=10, data_dim=2)
        assert samples.shape == (10, 2), f"Generation shape error: {samples.shape}"
        print(f"✓ Generated samples: {samples.shape}")
        
        # Evaluate
        real_data = np.random.randn(10, 2)
        metrics = inference.evaluate_fid_like(samples, real_data)
        assert 'mean_distance' in metrics, "Evaluation metrics missing"
        print(f"✓ Evaluation complete: mean_distance={metrics['mean_distance']:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Jax-Diffusion-Toy Project Validation")
    print("="*60)
    
    tests = [
        ("Module Import", test_imports),
        ("Datasets", test_datasets),
        ("Diffusion Process", test_diffusion),
        ("Models", test_models),
        ("Samplers", test_samplers),
        ("Training", test_training),
        ("Inference", test_inference),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ {name} test exception: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for name, passed in results.items():
        status = "✓ Passed" if passed else "✗ Failed"
        print(f"{status}: {name}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Project is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
