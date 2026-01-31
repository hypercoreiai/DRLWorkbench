# Ensemble Autoencoder Pipeline

## Overview

The `run_pipeline_ensemble_auto.py` script implements a comprehensive autoencoder pipeline for market analysis, combining multiple autoencoder architectures (Standard, VAE, Denoising) for regime detection, anomaly detection, and feature extraction.

## Features

### Autoencoder Models
1. **Standard Autoencoder** - Dimensionality reduction and feature extraction
2. **Variational Autoencoder (VAE)** - Probabilistic encoding for robust clustering
3. **Denoising Autoencoder** - Robust features from noisy market data

### Applications
- **Regime Detection**: Identifies market states via clustering in latent space
- **Anomaly Detection**: Flags unusual market conditions via reconstruction error
- **Feature Extraction**: Reduces 195 features to 16-dimensional latent space
- **Market State Representation**: Compact encoding of complex market dynamics

### Comprehensive Analysis
- **Latent Space Visualization**: t-SNE and PCA projections
- **Clustering**: K-Means on latent representations (5 regimes detected)
- **Anomaly Scoring**: Reconstruction error-based detection
- **Regime Characterization**: Returns analysis by detected regime

## Usage

```bash
# Run the pipeline
python run/run_pipeline_ensemble_auto.py \
  --config configs/autoencoder_ensemble.yaml \
  --output outputs/autoencoder

# Runtime: ~20 seconds
```

## Example Results (Test Run)

### Models Trained
- ✅ Standard Autoencoder (loss: 0.590)
- ✅ VAE (loss: 0.878)
- ✅ Denoising Autoencoder (loss: 0.575)

### Dimensionality Reduction
- **Input**: 195 features (15 assets × 13 features/asset)
- **Output**: 16-dimensional latent space
- **Compression**: 92% reduction

### Regime Detection
- **Method**: K-Means clustering on latent space
- **Regimes Found**: 5 distinct market states
- **Distribution**: [53, 15, 5, 7, 7] samples per regime

### Anomaly Detection
- **Threshold**: 95th percentile of reconstruction error
- **Anomalies Detected**: 5 (5.7% of test samples)
- **Mean Recon Error**: 0.032 (Standard), 0.038 (VAE), 0.032 (Denoising)

### Outputs Generated
- 9-panel visualization (602KB PNG)
- Latent encodings for each model (CSV)
- Cluster assignments (CSV)
- Anomaly flags (CSV)
- Reconstruction errors (CSV)
- Checkpoint for resumability

## Configuration

```yaml
models:
  types: [standard, vae, denoising]
  encoding_dims: [64, 32, 16]  # Layer dimensions
  latent_dim: 16
  epochs: 100
  batch_size: 128
  learning_rate: 0.001

analysis:
  clustering:
    n_clusters: 5
  anomaly:
    threshold_percentile: 95
```

## Visualizations (9 Plots)

1. **Latent Space 2D (t-SNE)** - Colored by cluster
2. **Reconstruction Errors** - Time series
3. **Anomaly Detection** - With threshold
4. **Training Curves** - Loss over epochs
5. **Latent Space 3D (PCA)** - 3D scatter
6. **Regime Distribution** - Cluster sizes
7. **Error Distribution** - Histogram
8. **Explained Variance** - PCA components
9. **Returns by Regime** - Scatter plot

## Files Created

1. **`src/models/autoencoders.py`** (600+ lines)
   - StandardAutoencoder, VariationalAutoencoder, DenoisingAutoencoder
   - AutoencoderTrainer - Unified training interface
   
2. **`src/data/autoencoder_pipeline.py`** (200+ lines)
   - Comprehensive feature engineering
   - Multi-asset market data preparation
   
3. **`run/run_pipeline_ensemble_auto.py`** (350+ lines)
   - Complete 9-step pipeline
   
4. **`configs/autoencoder_ensemble.yaml`**
   - Full configuration

**Total: ~1,200 lines of production code**

## Applications

### Trading Strategies
- **Regime-Based Allocation**: Adjust positions based on detected regime
- **Anomaly Alerts**: Flag unusual market conditions for risk management
- **State Transitions**: Detect regime changes early

### Risk Management
- **Anomaly Monitoring**: Real-time unusual pattern detection
- **Regime Risk**: Different risk models for different states
- **Stress Detection**: Identify market stress periods

### Research
- **Market Structure**: Understand natural market states
- **Feature Selection**: Identify most important market factors
- **Dimensionality Reduction**: Simplify complex market data

## Performance

- **Training**: 3-6 seconds per model
- **Encoding**: <1 second for 87 samples
- **Clustering**: <2 seconds
- **Visualization**: 1-2 seconds
- **Total Runtime**: ~20 seconds for 15 assets

GPU acceleration available for training (3-5x speedup).

## License

MIT License - See project LICENSE file for details.
