import torch
import torch.nn as nn
import optuna
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple

class OptimizedGenerator(nn.Module):
    def __init__(self, latent_dim: int, img_shape: Tuple[int, int, int], 
                 hidden_dims: list, activation: str = 'leaky_relu'):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        
        # Activation function selection
        if activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        else:
            act_fn = nn.LeakyReLU(0.2)
            
        # Calculate initial size for reshape
        c, h, w = img_shape
        init_h, init_w = h // 8, w // 8  # 3 upsampling layers (2^3 = 8)
        
        layers = []
        layers.append(nn.Linear(latent_dim, hidden_dims[0] * init_h * init_w))
        layers.append(act_fn)
        layers.append(nn.Unflatten(1, (hidden_dims[0], init_h, init_w)))
        
        # Convolutional transpose layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1], 
                                 kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                act_fn
            ])
        
        # Final layer
        layers.extend([
            nn.ConvTranspose2d(hidden_dims[-1], c, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.model(z)

class OptimizedDiscriminator(nn.Module):
    def __init__(self, img_shape: Tuple[int, int, int], hidden_dims: list, 
                 activation: str = 'leaky_relu', dropout_rate: float = 0.3):
        super().__init__()
        
        # Activation function selection
        if activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.2)
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        else:
            act_fn = nn.LeakyReLU(0.2)
            
        c, h, w = img_shape
        layers = []
        
        # First layer
        layers.extend([
            nn.Conv2d(c, hidden_dims[0], 4, stride=2, padding=1),
            act_fn,
            nn.Dropout2d(dropout_rate)
        ])
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Conv2d(hidden_dims[i], hidden_dims[i+1], 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                act_fn,
                nn.Dropout2d(dropout_rate)
            ])
        
        # Final layers
        layers.extend([
            nn.Conv2d(hidden_dims[-1], 1, 4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        ])
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, img):
        return self.model(img)

class GANHyperparameterOptimizer:
    def __init__(self, dataset_path: str, img_shape: Tuple[int, int, int] = (3, 256, 128)):
        self.dataset_path = dataset_path
        self.img_shape = img_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_params = None
        self.best_score = float('inf')
        
    def create_dataset(self, batch_size: int):
        """Create dataset and dataloader"""
        from torch.utils.data import Dataset
        
        class BoreholeDataset(Dataset):
            def __init__(self, root_dir, transform=None):
                self.img_paths = [os.path.join(root_dir, fname) 
                                for fname in os.listdir(root_dir) 
                                if fname.lower().endswith('.png')]
                self.transform = transform

            def __len__(self):
                return len(self.img_paths)

            def __getitem__(self, idx):
                img = Image.open(self.img_paths[idx]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img
        
        transform = transforms.Compose([
            transforms.Resize((self.img_shape[1], self.img_shape[2])),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        dataset = BoreholeDataset(self.dataset_path, transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader
    
    def objective(self, trial):
        """Optuna objective function for hyperparameter optimization"""
        # Suggest hyperparameters
        params = {
            'latent_dim': trial.suggest_int('latent_dim', 64, 256, step=32),
            'g_hidden_dims': [
                trial.suggest_int('g_dim1', 128, 512, step=64),
                trial.suggest_int('g_dim2', 64, 256, step=32),
                trial.suggest_int('g_dim3', 32, 128, step=16)
            ],
            'd_hidden_dims': [
                trial.suggest_int('d_dim1', 32, 128, step=16),
                trial.suggest_int('d_dim2', 64, 256, step=32),
                trial.suggest_int('d_dim3', 128, 512, step=64)
            ],
            'lr_g': trial.suggest_float('lr_g', 1e-5, 1e-2, log=True),
            'lr_d': trial.suggest_float('lr_d', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'activation': trial.suggest_categorical('activation', ['leaky_relu', 'relu', 'elu']),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'beta1': trial.suggest_float('beta1', 0.5, 0.9),
            'beta2': trial.suggest_float('beta2', 0.9, 0.999)
        }
        
        # Create models
        generator = OptimizedGenerator(
            latent_dim=params['latent_dim'],
            img_shape=self.img_shape,
            hidden_dims=params['g_hidden_dims'],
            activation=params['activation']
        ).to(self.device)
        
        discriminator = OptimizedDiscriminator(
            img_shape=self.img_shape,
            hidden_dims=params['d_hidden_dims'],
            activation=params['activation'],
            dropout_rate=params['dropout_rate']
        ).to(self.device)
        
        # Create optimizers
        optimizer_G = torch.optim.Adam(
            generator.parameters(), 
            lr=params['lr_g'], 
            betas=(params['beta1'], params['beta2'])
        )
        optimizer_D = torch.optim.Adam(
            discriminator.parameters(), 
            lr=params['lr_d'], 
            betas=(params['beta1'], params['beta2'])
        )
        
        # Create dataloader
        dataloader = self.create_dataset(params['batch_size'])
        
        # Training loop (shortened for optimization)
        adversarial_loss = nn.BCELoss()
        epochs = 20  # Reduced for faster optimization
        
        generator.train()
        discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        num_batches = 0
        
        for epoch in range(epochs):
            for imgs in dataloader:
                if len(imgs) < params['batch_size']:  # Skip incomplete batches
                    continue
                    
                real_imgs = imgs.to(self.device)
                batch_size = real_imgs.size(0)
                
                # Labels
                valid = torch.ones((batch_size, 1), device=self.device)
                fake = torch.zeros((batch_size, 1), device=self.device)
                
                # Train Generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size, params['latent_dim'], device=self.device)
                gen_imgs = generator(z)
                validity = discriminator(gen_imgs)
                g_loss = adversarial_loss(validity, valid)
                g_loss.backward()
                optimizer_G.step()
                
                # Train Discriminator
                optimizer_D.zero_grad()
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                
                total_g_loss += g_loss.item()
                total_d_loss += d_loss.item()
                num_batches += 1
        
        # Calculate average losses
        avg_g_loss = total_g_loss / num_batches if num_batches > 0 else float('inf')
        avg_d_loss = total_d_loss / num_batches if num_batches > 0 else float('inf')
        
        # Combined loss as optimization target (balance between G and D)
        combined_loss = avg_g_loss + avg_d_loss
        
        return combined_loss
    
    def optimize(self, n_trials: int = 50, timeout: int = 3600):
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        # Save results
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study_stats': {
                'n_trials': len(study.trials),
                'best_trial': study.best_trial.number,
                'optimization_time': str(datetime.now())
            }
        }
        
        os.makedirs('hyperparameter_optimization/results', exist_ok=True)
        with open('hyperparameter_optimization/results/gan_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return study
    
    def create_optimized_models(self):
        """Create models with optimized hyperparameters"""
        if self.best_params is None:
            raise ValueError("No optimization results found. Run optimize() first.")
        
        generator = OptimizedGenerator(
            latent_dim=self.best_params['latent_dim'],
            img_shape=self.img_shape,
            hidden_dims=[
                self.best_params['g_dim1'],
                self.best_params['g_dim2'],
                self.best_params['g_dim3']
            ],
            activation=self.best_params['activation']
        ).to(self.device)
        
        discriminator = OptimizedDiscriminator(
            img_shape=self.img_shape,
            hidden_dims=[
                self.best_params['d_dim1'],
                self.best_params['d_dim2'],
                self.best_params['d_dim3']
            ],
            activation=self.best_params['activation'],
            dropout_rate=self.best_params['dropout_rate']
        ).to(self.device)
        
        return generator, discriminator

if __name__ == "__main__":
    # Example usage
    optimizer = GANHyperparameterOptimizer("synthetic_borehole_dataset")
    study = optimizer.optimize(n_trials=30, timeout=1800)  # 30 minutes
    
    print(f"Best parameters: {optimizer.best_params}")
    print(f"Best score: {optimizer.best_score}")
    
    # Create optimized models
    gen, disc = optimizer.create_optimized_models()
    print("Optimized models created successfully!")