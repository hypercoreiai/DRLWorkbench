"""
PatchTST (Patched Time Series Transformer) model and trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class PatchTSTConfig:
    """Configuration for PatchTST model."""

    seq_len: int = 504
    pred_len: int = 30
    d_model: int = 256
    n_heads: int = 8
    n_encoder_layers: int = 6
    d_ff: int = 512
    dropout: float = 0.1
    patch_len: int = 16
    stride: int = 8
    n_features: int = 35
    activation: str = "gelu"


class PatchEmbedding(nn.Module):
    """Patch embedding layer for PatchTST."""

    def __init__(self, patch_len: int, stride: int, d_model: int, n_features: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.linear = nn.Linear(patch_len * n_features, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch, n_patches, d_model)
        """
        if x.dim() != 3:
            raise ValueError("Input must be 3D: (batch, seq_len, n_features)")
        # (batch, n_patches, patch_len, n_features)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # Flatten patch
        patches = patches.contiguous().view(patches.size(0), patches.size(1), -1)
        return self.linear(patches)


class TransformerEncoder(nn.Module):
    """Transformer encoder block."""

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU() if config.activation == "gelu" else nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff))
        return x


class PatchTST(nn.Module):
    """Patch Time Series Transformer."""

    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config
        self.embedding = PatchEmbedding(
            config.patch_len,
            config.stride,
            config.d_model,
            config.n_features,
        )
        n_patches = 1 + (config.seq_len - config.patch_len) // config.stride
        self.pos_encoding = nn.Parameter(torch.randn(1, n_patches, config.d_model))
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoder(config) for _ in range(config.n_encoder_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)

        Returns:
            (batch, pred_len)
        """
        x = self.embedding(x)
        pos = self.pos_encoding[:, : x.size(1), :]
        x = x + pos
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.head(x)


class PatchTSTTrainer:
    """Handle model training and validation."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: PatchTSTConfig,
        device: torch.device | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.criterion = nn.MSELoss()
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")
        self.best_state: Dict[str, torch.Tensor] | None = None

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / max(len(self.train_loader), 1)

    def validate(self) -> float:
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                running_loss += loss.item()
        return running_loss / max(len(self.val_loader), 1)

    def train(self, epochs: int = 100, early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        patience_counter = 0
        for _ in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step(val_loss)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return {"train_loss": self.train_losses, "val_loss": self.val_losses}

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)


@torch.no_grad()
def predict_batches(model: nn.Module, loader: DataLoader, device: torch.device | None = None) -> np.ndarray:
    """Predict over a DataLoader and return numpy array."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    preds = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        preds.append(outputs.cpu().numpy())
    if not preds:
        return np.empty((0,))
    return np.concatenate(preds, axis=0)