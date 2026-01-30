"""
Autoencoder Models for Market Analysis
Implements: Standard, Variational (VAE), Denoising, and Contractive Autoencoders
For regime detection, anomaly detection, and feature extraction
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseForecaster
from src.utils.errors import PredictionError


class StandardAutoencoder(nn.Module):
    """
    Standard Autoencoder for dimensionality reduction and feature extraction.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: list = None,
        activation: str = 'relu',
        dropout: float = 0.2
    ):
        super(StandardAutoencoder, self).__init__()
        
        if encoding_dims is None:
            encoding_dims = [64, 32, 16]
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.latent_dim = encoding_dims[-1]
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(self._get_activation(activation))
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (mirror of encoder)
        decoder_layers = []
        for i in range(len(encoding_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(encoding_dims[i], encoding_dims[i-1]))
            decoder_layers.append(nn.BatchNorm1d(encoding_dims[i-1]))
            decoder_layers.append(self._get_activation(activation))
            decoder_layers.append(nn.Dropout(dropout))
        
        # Final layer
        decoder_layers.append(nn.Linear(encoding_dims[0], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder (VAE) for probabilistic encoding and generation.
    Good for anomaly detection and regime clustering.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: list = None,
        activation: str = 'relu',
        dropout: float = 0.2
    ):
        super(VariationalAutoencoder, self).__init__()
        
        if encoding_dims is None:
            encoding_dims = [64, 32, 16]
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.latent_dim = encoding_dims[-1]
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims[:-1]:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(self._get_activation(activation))
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space: mean and log-variance
        self.fc_mu = nn.Linear(prev_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self.latent_dim)
        
        # Decoder
        decoder_layers = []
        decoder_layers.append(nn.Linear(self.latent_dim, encoding_dims[-2]))
        decoder_layers.append(nn.BatchNorm1d(encoding_dims[-2]))
        decoder_layers.append(self._get_activation(activation))
        decoder_layers.append(nn.Dropout(dropout))
        
        for i in range(len(encoding_dims) - 2, 0, -1):
            decoder_layers.append(nn.Linear(encoding_dims[i], encoding_dims[i-1]))
            decoder_layers.append(nn.BatchNorm1d(encoding_dims[i-1]))
            decoder_layers.append(self._get_activation(activation))
            decoder_layers.append(nn.Dropout(dropout))
        
        decoder_layers.append(nn.Linear(encoding_dims[0], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def encode(self, x):
        """Encode to latent distribution parameters."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass with reparameterization."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder that learns robust features by reconstructing from corrupted input.
    Good for noisy market data.
    """
    
    def __init__(
        self,
        input_dim: int,
        encoding_dims: list = None,
        activation: str = 'relu',
        dropout: float = 0.2,
        noise_factor: float = 0.1
    ):
        super(DenoisingAutoencoder, self).__init__()
        
        if encoding_dims is None:
            encoding_dims = [64, 32, 16]
        
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        self.latent_dim = encoding_dims[-1]
        self.noise_factor = noise_factor
        
        # Same architecture as StandardAutoencoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.append(nn.Linear(prev_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(self._get_activation(activation))
            encoder_layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = []
        for i in range(len(encoding_dims) - 1, 0, -1):
            decoder_layers.append(nn.Linear(encoding_dims[i], encoding_dims[i-1]))
            decoder_layers.append(nn.BatchNorm1d(encoding_dims[i-1]))
            decoder_layers.append(self._get_activation(activation))
            decoder_layers.append(nn.Dropout(dropout))
        
        decoder_layers.append(nn.Linear(encoding_dims[0], input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def _get_activation(self, name: str):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'elu': nn.ELU()
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def add_noise(self, x):
        """Add Gaussian noise to input."""
        if self.training:
            noise = torch.randn_like(x) * self.noise_factor
            return x + noise
        return x
    
    def encode(self, x):
        """Encode (possibly noisy) input."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass with noise injection during training."""
        x_noisy = self.add_noise(x)
        z = self.encode(x_noisy)
        x_recon = self.decode(z)
        return x_recon, z


class AutoencoderTrainer:
    """
    Unified trainer for all autoencoder types.
    Handles training, evaluation, and encoding/decoding.
    """
    
    def __init__(
        self,
        model_type: str = 'standard',
        input_dim: int = 10,
        encoding_dims: list = None,
        config: Dict[str, Any] = None
    ):
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims or [64, 32, 16]
        self.config = config or {}
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        if self.model_type == 'standard':
            self.model = StandardAutoencoder(
                input_dim=input_dim,
                encoding_dims=self.encoding_dims,
                activation=self.config.get('activation', 'relu'),
                dropout=self.config.get('dropout', 0.2)
            )
        elif self.model_type == 'vae':
            self.model = VariationalAutoencoder(
                input_dim=input_dim,
                encoding_dims=self.encoding_dims,
                activation=self.config.get('activation', 'relu'),
                dropout=self.config.get('dropout', 0.2)
            )
        elif self.model_type == 'denoising':
            self.model = DenoisingAutoencoder(
                input_dim=input_dim,
                encoding_dims=self.encoding_dims,
                activation=self.config.get('activation', 'relu'),
                dropout=self.config.get('dropout', 0.2),
                noise_factor=self.config.get('noise_factor', 0.1)
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model = self.model.to(self.device)
        self.history = {}
        self.scaler_mean = None
        self.scaler_std = None
    
    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict]:
        """
        Train the autoencoder.
        
        Args:
            X_train: Training data (N, input_dim)
            X_val: Validation data (optional)
            config: Training configuration
        
        Returns:
            (model, history) tuple
        """
        if config is None:
            config = self.config
        
        # Normalize data
        X_train_norm = self._normalize(X_train, fit=True)
        if X_val is not None:
            X_val_norm = self._normalize(X_val, fit=False)
        else:
            X_val_norm = None
        
        # Training parameters
        epochs = config.get('epochs', 100)
        batch_size = config.get('batch_size', 128)
        learning_rate = config.get('learning_rate', 0.001)
        
        # DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train_norm))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if X_val_norm is not None:
            val_dataset = TensorDataset(torch.FloatTensor(X_val_norm))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        else:
            val_loader = None
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'train_recon_loss': [], 'train_kl_loss': []}
        best_val_loss = float('inf')
        patience = config.get('patience', 15)
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            
            for (X_batch,) in train_loader:
                X_batch = X_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass depends on model type
                if self.model_type == 'vae':
                    x_recon, mu, logvar, z = self.model(X_batch)
                    recon_loss = F.mse_loss(x_recon, X_batch, reduction='mean')
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    beta = config.get('beta', 1.0)  # β-VAE parameter
                    loss = recon_loss + beta * kl_loss
                    train_recon_loss += recon_loss.item()
                    train_kl_loss += kl_loss.item()
                else:
                    x_recon, z = self.model(X_batch)
                    loss = F.mse_loss(x_recon, X_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            if self.model_type == 'vae':
                train_recon_loss /= len(train_loader)
                train_kl_loss /= len(train_loader)
                history['train_recon_loss'].append(train_recon_loss)
                history['train_kl_loss'].append(train_kl_loss)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for (X_batch,) in val_loader:
                        X_batch = X_batch.to(self.device)
                        
                        if self.model_type == 'vae':
                            x_recon, mu, logvar, z = self.model(X_batch)
                            recon_loss = F.mse_loss(x_recon, X_batch, reduction='mean')
                            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                            beta = config.get('beta', 1.0)
                            loss = recon_loss + beta * kl_loss
                        else:
                            x_recon, z = self.model(X_batch)
                            loss = F.mse_loss(x_recon, X_batch)
                        
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
        
        self.history = history
        return self.model, history
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode data to latent representation.
        
        Args:
            X: Input data (N, input_dim)
        
        Returns:
            Latent representations (N, latent_dim)
        """
        X_norm = self._normalize(X, fit=False)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(self.device)
            
            if self.model_type == 'vae':
                mu, logvar = self.model.encode(X_tensor)
                z = mu  # Use mean for deterministic encoding
            else:
                z = self.model.encode(X_tensor)
            
            return z.cpu().numpy()
    
    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct data through autoencoder.
        
        Args:
            X: Input data (N, input_dim)
        
        Returns:
            Reconstructed data (N, input_dim)
        """
        X_norm = self._normalize(X, fit=False)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_norm).to(self.device)
            
            if self.model_type == 'vae':
                x_recon, mu, logvar, z = self.model(X_tensor)
            else:
                x_recon, z = self.model(X_tensor)
            
            # Denormalize
            x_recon_np = x_recon.cpu().numpy()
            x_recon_denorm = x_recon_np * (self.scaler_std + 1e-8) + self.scaler_mean
            
            return x_recon_denorm
    
    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for anomaly detection.
        
        Args:
            X: Input data (N, input_dim)
        
        Returns:
            Reconstruction errors (N,)
        """
        X_recon = self.reconstruct(X)
        errors = np.mean((X - X_recon) ** 2, axis=1)
        return errors
    
    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize data using z-score normalization."""
        if fit:
            self.scaler_mean = X.mean(axis=0, keepdims=True)
            self.scaler_std = X.std(axis=0, keepdims=True)
        
        X_norm = (X - self.scaler_mean) / (self.scaler_std + 1e-8)
        return X_norm
    
    def get_hyperparams(self) -> Dict[str, Any]:
        """Return model configuration."""
        return {
            'model_type': self.model_type,
            'input_dim': self.input_dim,
            'encoding_dims': self.encoding_dims,
            'latent_dim': self.encoding_dims[-1],
            'config': self.config
        }


def get_autoencoder_model(model_type: str, input_dim: int, config: Dict = None) -> AutoencoderTrainer:
    """
    Factory function for autoencoder models.
    
    Args:
        model_type: Model type ('standard', 'vae', 'denoising')
        input_dim: Input dimension
        config: Configuration dictionary
    
    Returns:
        AutoencoderTrainer instance
    """
    return AutoencoderTrainer(
        model_type=model_type,
        input_dim=input_dim,
        config=config or {}
    )
