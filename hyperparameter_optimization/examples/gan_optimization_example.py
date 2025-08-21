#!/usr/bin/env python3
"""
GAN Hyperparameter Optimization Example

This example demonstrates how to use the GAN hyperparameter optimization framework
to optimize Generator and Discriminator models for synthetic data generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from gan_hyperopt import GANHyperparameterOptimizer
from visualizer import HyperparameterVisualizationSuite

def create_sample_dataset(n_samples=1000, image_size=64, n_channels=3):
    """
    Create a sample dataset for GAN training
    
    Args:
        n_samples: Number of samples to generate
        image_size: Size of square images
        n_channels: Number of channels (RGB=3, Grayscale=1)
    
    Returns:
        X: Sample images tensor
        y: Sample labels tensor
    """
    # Generate synthetic image data
    X = torch.randn(n_samples, n_channels, image_size, image_size)
    
    # Add some structure to make it more realistic
    for i in range(n_samples):
        # Add some patterns
        if i % 3 == 0:
            # Horizontal stripes
            X[i, :, ::4, :] *= 2
        elif i % 3 == 1:
            # Vertical stripes
            X[i, :, :, ::4] *= 2
        else:
            # Checkerboard pattern
            X[i, :, ::2, ::2] *= 1.5
            X[i, :, 1::2, 1::2] *= 1.5
    
    # Normalize to [-1, 1] range (typical for GANs)
    X = torch.tanh(X)
    
    # Generate binary labels (for conditional GAN if needed)
    y = torch.randint(0, 2, (n_samples,))
    
    return X, y

def main():
    """
    Main function demonstrating GAN hyperparameter optimization
    """
    print("=" * 60)
    print("GAN Hyperparameter Optimization Example")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create results directory
    results_dir = Path("results/gan_example")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Create sample dataset
    print("\n1. Creating sample dataset...")
    X, y = create_sample_dataset(n_samples=500, image_size=32, n_channels=1)  # Smaller for faster training
    print(f"   Dataset shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   Data range: [{X.min():.3f}, {X.max():.3f}]")
    
    # Step 2: Initialize GAN optimizer
    print("\n2. Initializing GAN hyperparameter optimizer...")
    config_path = "config/gan_config.yaml"
    
    gan_optimizer = GANHyperparameterOptimizer(
        config_path=config_path,
        results_dir=str(results_dir),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"   Using device: {gan_optimizer.device}")
    print(f"   Config loaded from: {config_path}")
    
    # Step 3: Run hyperparameter optimization
    print("\n3. Running hyperparameter optimization...")
    print("   This may take several minutes depending on your hardware.")
    
    study = gan_optimizer.optimize(
        X=X,
        y=y,
        n_trials=20,  # Reduced for example - increase for better results
        timeout=1800,  # 30 minutes timeout
        show_progress_bar=True
    )
    
    # Step 4: Display optimization results
    print("\n4. Optimization Results:")
    print(f"   Number of trials: {len(study.trials)}")
    print(f"   Best value: {study.best_value:.6f}")
    print("   Best parameters:")
    for param, value in study.best_params.items():
        print(f"     {param}: {value}")
    
    # Step 5: Create optimized models
    print("\n5. Creating optimized models...")
    best_generator, best_discriminator = gan_optimizer.create_optimized_models(
        study.best_params
    )
    
    print(f"   Generator parameters: {sum(p.numel() for p in best_generator.parameters()):,}")
    print(f"   Discriminator parameters: {sum(p.numel() for p in best_discriminator.parameters()):,}")
    
    # Step 6: Generate sample images with best model
    print("\n6. Generating sample images with optimized model...")
    best_generator.eval()
    with torch.no_grad():
        # Generate random noise
        latent_dim = study.best_params.get('generator_latent_dim', 100)
        noise = torch.randn(16, latent_dim, device=gan_optimizer.device)
        
        # Generate fake images
        fake_images = best_generator(noise)
        
        # Move to CPU for visualization
        fake_images = fake_images.cpu()
        
        # Create visualization
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.suptitle('Generated Images (Optimized GAN)', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if fake_images.shape[1] == 1:  # Grayscale
                ax.imshow(fake_images[i, 0], cmap='gray')
            else:  # RGB
                # Denormalize from [-1, 1] to [0, 1]
                img = (fake_images[i].permute(1, 2, 0) + 1) / 2
                ax.imshow(img.clamp(0, 1))
            ax.axis('off')
        
        plt.tight_layout()
        sample_path = results_dir / 'generated_samples.png'
        plt.savefig(sample_path, dpi=150, bbox_inches='tight')
        print(f"   Sample images saved to: {sample_path}")
        plt.close()
    
    # Step 7: Create visualizations
    print("\n7. Creating optimization visualizations...")
    visualizer = HyperparameterVisualizationSuite(str(results_dir))
    
    # Optimization history
    hist_fig = visualizer.plot_optimization_history(
        study, 
        save_path='optimization_history.html'
    )
    print("   ✓ Optimization history plot created")
    
    # Parameter importance
    imp_fig = visualizer.plot_parameter_importance(
        study, 
        save_path='parameter_importance.html'
    )
    if imp_fig:
        print("   ✓ Parameter importance plot created")
    
    # Parallel coordinates
    par_fig = visualizer.plot_parallel_coordinate(
        study, 
        save_path='parallel_coordinates.html'
    )
    if par_fig:
        print("   ✓ Parallel coordinates plot created")
    
    # Create comprehensive dashboard
    dashboard_path = visualizer.create_optimization_dashboard(
        study, 
        save_path='gan_optimization_dashboard.html'
    )
    print(f"   ✓ Comprehensive dashboard created: {dashboard_path}")
    
    # Step 8: Save optimization study
    print("\n8. Saving optimization study...")
    study_path = results_dir / 'gan_optimization_study.pkl'
    
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    
    print(f"   Study saved to: {study_path}")
    
    # Step 9: Generate final report
    print("\n9. Generating optimization report...")
    report = visualizer.generate_optimization_report(study)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Results directory: {results_dir}")
    print(f"Best objective value: {study.best_value:.6f}")
    print(f"Total trials completed: {len(study.trials)}")
    print("\nFiles created:")
    print(f"  - Generated samples: generated_samples.png")
    print(f"  - Optimization study: gan_optimization_study.pkl")
    print(f"  - Visualization dashboard: gan_optimization_dashboard.html")
    print(f"  - Optimization report: optimization_report.json")
    print("\nNext steps:")
    print("  1. Review the optimization dashboard for insights")
    print("  2. Use the best parameters for full-scale training")
    print("  3. Consider running more trials for better optimization")
    print("  4. Experiment with different architectures in the config")

def load_and_analyze_study(study_path: str):
    """
    Load a previously saved study and analyze results
    
    Args:
        study_path: Path to the saved study pickle file
    """
    import pickle
    
    print(f"Loading study from: {study_path}")
    
    with open(study_path, 'rb') as f:
        study = pickle.load(f)
    
    print(f"Study loaded with {len(study.trials)} trials")
    print(f"Best value: {study.best_value:.6f}")
    print("Best parameters:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    # Analyze trial distribution
    values = [trial.value for trial in study.trials if trial.value is not None]
    print(f"\nTrial statistics:")
    print(f"  Mean: {np.mean(values):.6f}")
    print(f"  Std: {np.std(values):.6f}")
    print(f"  Min: {np.min(values):.6f}")
    print(f"  Max: {np.max(values):.6f}")
    
    return study

if __name__ == "__main__":
    # Check if we should load an existing study
    if len(sys.argv) > 1 and sys.argv[1] == "--load":
        if len(sys.argv) > 2:
            study = load_and_analyze_study(sys.argv[2])
        else:
            print("Please provide path to study file: python gan_optimization_example.py --load path/to/study.pkl")
    else:
        # Run the main optimization example
        main()