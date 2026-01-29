# Ensemble methods: combine multiple forecasters/DRL agents (V3 — PROJECT_OUTLINE Section 3.4)
# V4 Update: GPU Acceleration via PyTorch

from typing import Any, Dict, List, Optional
import numpy as np
import torch

def get_device() -> torch.device:
    """Detect and return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available(): # Mac support
         return torch.device("mps")
    return torch.device("cpu")

def ensemble_predict(
    models: List[Any],
    X: Any,
    method: str = "mean",
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Combine predictions from multiple models using PyTorch for acceleration.

    Args:
        models: List of fitted models with .predict(X).
        X: Input features.
        method: 'mean', 'median', or 'weighted'.
        weights: Weights per model (for 'weighted').

    Returns:
        Combined prediction array (NumPy).
    """
    device = get_device()
    
    # 1. Gather predictions
    # Models might return numpy arrays or tensors or lists
    preds_list = []
    for m in models:
        p = m.predict(X)
        if isinstance(p, np.ndarray):
            p = torch.from_numpy(p)
        elif not isinstance(p, torch.Tensor):
            p = torch.tensor(p)
        preds_list.append(p.to(device, dtype=torch.float32))
        
    if not preds_list:
        return np.array([])
        
    # Stack: (n_models, n_samples, ...)
    # Stack inputs onto the GPU
    stack = torch.stack(preds_list)
    
    # 2. Compute Aggregate
    if method == "mean":
        result = torch.nanmean(stack, dim=0)
    elif method == "median":
        # torch.nanmedian not strictly available in all versions same way as numpy, 
        # but torch.median ignores NaNs if strictly tensor-based? No, usually not.
        # Fallback: if NaNs present, mask them. For now assume clean data or standard median.
        result = torch.median(stack, dim=0).values
    elif method == "weighted" and weights is not None:
        w = torch.tensor(weights, device=device, dtype=torch.float32)
        w = w / w.sum()
        # Reshape w to broadcast: (n_models, 1, 1...)
        # Assume 2D predictions (samples, targets) for now, or 1D
        shape = [len(models)] + [1] * (stack.ndim - 1)
        w_broad = w.view(*shape)
        result = (stack * w_broad).sum(dim=0)
    else:
        # Default to mean
        result = torch.nanmean(stack, dim=0)
        
    # 3. Return as NumPy (CPU)
    return result.cpu().numpy()
