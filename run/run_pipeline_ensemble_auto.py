"""
Ensemble Autoencoder Pipeline
Market regime detection, anomaly detection, and feature extraction
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logging import setup_logging
from src.utils.checkpoint import save_checkpoint
from src.utils import ConfigError
from src.data.autoencoder_pipeline import AutoencoderPipeline
from src.models.autoencoders import get_autoencoder_model


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def run_autoencoder_pipeline(
    config_path: str,
    output_dir: str,
    resume: Optional[str] = None
) -> None:
    config = load_config(config_path)
    run_id = config.get("display", {}).get("run_id", "autoencoder")
    
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(str(log_dir), run_id)
    logger.info("=" * 80)
    logger.info("Starting Ensemble Autoencoder Pipeline")
    logger.info("=" * 80)
    
    # STEP 1: DATA PIPELINE
    logger.info("STEP 1: Data Pipeline")
    pipeline = AutoencoderPipeline(config)
    data_bundle = pipeline.run()
    
    if data_bundle.features_train is None:
        logger.error("Data pipeline failed")
        return
    
    logger.info(f"Loaded {data_bundle.metadata['n_assets']} assets")
    logger.info(f"Features: {data_bundle.metadata['n_features']}")
    logger.info(f"Training samples: {len(data_bundle.features_train)}")
    logger.info(f"Testing samples: {len(data_bundle.features_test)}")
    
    # STEP 2: MODEL TRAINING
    logger.info("=" * 80)
    logger.info("STEP 2: Training Autoencoders")
    logger.info("=" * 80)
    
    model_types = config.get('models', {}).get('types', ['standard', 'vae'])
    input_dim = data_bundle.features_train.shape[1]
    
    trained_models = {}
    
    for model_type in model_types:
        logger.info(f"\nTraining {model_type.upper()} autoencoder...")
        
        try:
            model = get_autoencoder_model(model_type, input_dim, config.get('models', {}))
            model.fit(
                data_bundle.features_train.values,
                data_bundle.features_test.values,
                config.get('models', {})
            )
            trained_models[model_type] = model
            
            final_loss = model.history['train_loss'][-1]
            logger.info(f"  Training completed. Final loss: {final_loss:.6f}")
            logger.info(f"  Latent dim: {model.encoding_dims[-1]}")
        except Exception as e:
            logger.error(f"  Training failed: {e}")
    
    if not trained_models:
        logger.error("No models trained successfully")
        return
    
    logger.info(f"\nSuccessfully trained {len(trained_models)} models")
    
    # STEP 3: LATENT SPACE ENCODING
    logger.info("=" * 80)
    logger.info("STEP 3: Latent Space Encoding")
    logger.info("=" * 80)
    
    latent_encodings = {}
    reconstruction_errors = {}
    
    for name, model in trained_models.items():
        logger.info(f"\nEncoding with {name.upper()}...")
        
        # Encode test data
        latent_test = model.encode(data_bundle.features_test.values)
        latent_encodings[name] = latent_test
        
        # Compute reconstruction error
        recon_errors = model.compute_reconstruction_error(data_bundle.features_test.values)
        reconstruction_errors[name] = recon_errors
        
        logger.info(f"  Latent shape: {latent_test.shape}")
        logger.info(f"  Mean recon error: {recon_errors.mean():.6f}")
        logger.info(f"  Std recon error: {recon_errors.std():.6f}")
    
    # STEP 4: CLUSTERING & REGIME DETECTION
    logger.info("=" * 80)
    logger.info("STEP 4: Regime Detection")
    logger.info("=" * 80)
    
    # Use primary model for clustering
    primary_model = list(trained_models.keys())[0]
    latent_data = latent_encodings[primary_model]
    
    n_clusters = config.get('analysis', {}).get('clustering', {}).get('n_clusters', 5)
    
    logger.info(f"Clustering with K-Means (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(latent_data)
    
    logger.info(f"Cluster distribution: {np.bincount(clusters)}")
    
    # STEP 5: ANOMALY DETECTION
    logger.info("=" * 80)
    logger.info("STEP 5: Anomaly Detection")
    logger.info("=" * 80)
    
    anomalies = {}
    threshold_pct = config.get('analysis', {}).get('anomaly', {}).get('threshold_percentile', 95)
    
    for name, errors in reconstruction_errors.items():
        threshold = np.percentile(errors, threshold_pct)
        is_anomaly = errors > threshold
        anomalies[name] = is_anomaly
        
        n_anomalies = is_anomaly.sum()
        logger.info(f"{name.upper()}: {n_anomalies} anomalies ({n_anomalies/len(errors)*100:.1f}%)")
        logger.info(f"  Threshold: {threshold:.6f}")
    
    # STEP 6: DIMENSIONALITY REDUCTION FOR VISUALIZATION
    logger.info("=" * 80)
    logger.info("STEP 6: Visualization Preparation")
    logger.info("=" * 80)
    
    # t-SNE for 2D visualization
    logger.info("Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_data)-1))
    latent_2d = tsne.fit_transform(latent_data)
    
    # PCA for 3D
    logger.info("Computing PCA projection...")
    pca = PCA(n_components=min(3, latent_data.shape[1]))
    latent_3d = pca.fit_transform(latent_data)
    
    # STEP 7: VISUALIZATIONS
    logger.info("=" * 80)
    logger.info("STEP 7: Creating Visualizations")
    logger.info("=" * 80)
    
    fig = plt.figure(figsize=(20, 14))
    
    # Plot 1: Latent space 2D (t-SNE)
    ax1 = plt.subplot(3, 3, 1)
    scatter = ax1.scatter(latent_2d[:, 0], latent_2d[:, 1], c=clusters, cmap='viridis', alpha=0.6, s=30)
    ax1.set_title('Latent Space (t-SNE) - Colored by Cluster')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax1)
    
    # Plot 2: Reconstruction errors
    ax2 = plt.subplot(3, 3, 2)
    for name, errors in reconstruction_errors.items():
        ax2.plot(errors, label=name, alpha=0.7)
    ax2.set_title('Reconstruction Errors Over Time')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Anomaly detection
    ax3 = plt.subplot(3, 3, 3)
    primary_errors = reconstruction_errors[primary_model]
    threshold = np.percentile(primary_errors, threshold_pct)
    ax3.plot(primary_errors, label='Reconstruction Error', alpha=0.7)
    ax3.axhline(y=threshold, color='r', linestyle='--', label=f'{threshold_pct}% Threshold')
    ax3.fill_between(range(len(primary_errors)), 0, primary_errors, 
                     where=primary_errors>threshold, alpha=0.3, color='red', label='Anomalies')
    ax3.set_title(f'Anomaly Detection ({primary_model.upper()})')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Training curves
    ax4 = plt.subplot(3, 3, 4)
    for name, model in trained_models.items():
        ax4.plot(model.history['train_loss'], label=f'{name} train', alpha=0.7)
        if model.history.get('val_loss'):
            ax4.plot(model.history['val_loss'], label=f'{name} val', linestyle='--', alpha=0.7)
    ax4.set_title('Training Curves')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Latent space 3D (PCA)
    ax5 = plt.subplot(3, 3, 5, projection='3d')
    ax5.scatter(latent_3d[:, 0], latent_3d[:, 1], latent_3d[:, 2] if latent_3d.shape[1] > 2 else 0, 
               c=clusters, cmap='viridis', alpha=0.6, s=20)
    ax5.set_title('Latent Space 3D (PCA)')
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    if latent_3d.shape[1] > 2:
        ax5.set_zlabel('PC3')
    
    # Plot 6: Cluster distribution
    ax6 = plt.subplot(3, 3, 6)
    cluster_counts = np.bincount(clusters)
    ax6.bar(range(len(cluster_counts)), cluster_counts, alpha=0.7)
    ax6.set_title('Regime Distribution')
    ax6.set_xlabel('Regime/Cluster')
    ax6.set_ylabel('Count')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Plot 7: Reconstruction error distribution
    ax7 = plt.subplot(3, 3, 7)
    for name, errors in reconstruction_errors.items():
        ax7.hist(errors, bins=50, alpha=0.5, label=name, density=True)
    ax7.set_title('Reconstruction Error Distribution')
    ax6.set_xlabel('Error')
    ax7.set_ylabel('Density')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Latent dimensions variance (PCA)
    ax8 = plt.subplot(3, 3, 8)
    explained_var = pca.explained_variance_ratio_[:min(10, len(pca.explained_variance_ratio_))]
    ax8.bar(range(len(explained_var)), explained_var, alpha=0.7)
    ax8.set_title('Latent Space Explained Variance')
    ax8.set_xlabel('Principal Component')
    ax8.set_ylabel('Variance Ratio')
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Plot 9: Returns vs Regime
    ax9 = plt.subplot(3, 3, 9)
    returns_test = data_bundle.returns_test.mean(axis=1).values
    for c in range(n_clusters):
        mask = clusters == c
        ax9.scatter(np.where(mask)[0], returns_test[mask], label=f'Regime {c}', alpha=0.6, s=20)
    ax9.set_title('Returns by Detected Regime')
    ax9.set_xlabel('Time')
    ax9.set_ylabel('Avg Return')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = out / f"{run_id}_analysis.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization: {fig_path}")
    plt.close()
    
    # STEP 8: EXPORT RESULTS
    logger.info("=" * 80)
    logger.info("STEP 8: Export Results")
    logger.info("=" * 80)
    
    # Save latent encodings
    for name, latent in latent_encodings.items():
        latent_df = pd.DataFrame(latent, index=data_bundle.features_test.index)
        latent_path = out / f"{run_id}_{name}_latent.csv"
        latent_df.to_csv(latent_path)
        logger.info(f"Saved {name} latent encodings: {latent_path}")
    
    # Save clusters
    clusters_df = pd.DataFrame({
        'cluster': clusters,
        'returns': returns_test
    }, index=data_bundle.features_test.index)
    clusters_path = out / f"{run_id}_clusters.csv"
    clusters_df.to_csv(clusters_path)
    logger.info(f"Saved clusters: {clusters_path}")
    
    # Save anomalies
    anomalies_df = pd.DataFrame(anomalies, index=data_bundle.features_test.index)
    anomalies_path = out / f"{run_id}_anomalies.csv"
    anomalies_df.to_csv(anomalies_path)
    logger.info(f"Saved anomalies: {anomalies_path}")
    
    # Save reconstruction errors
    errors_df = pd.DataFrame(reconstruction_errors, index=data_bundle.features_test.index)
    errors_path = out / f"{run_id}_reconstruction_errors.csv"
    errors_df.to_csv(errors_path)
    logger.info(f"Saved reconstruction errors: {errors_path}")
    
    # STEP 9: CHECKPOINT
    state = {
        'config': config,
        'run_id': run_id,
        'n_clusters': n_clusters,
        'cluster_counts': cluster_counts.tolist(),
        'models': list(trained_models.keys())
    }
    
    checkpoint_path = out / f"{run_id}_checkpoint.pkl"
    save_checkpoint(state, str(checkpoint_path))
    logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # FINAL SUMMARY
    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Models Trained: {', '.join(trained_models.keys())}")
    logger.info(f"Latent Dimension: {trained_models[primary_model].encoding_dims[-1]}")
    logger.info(f"Regimes Detected: {n_clusters}")
    logger.info(f"Anomalies Detected: {anomalies[primary_model].sum()}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Ensemble Autoencoder Pipeline")
    parser.add_argument("--config", default="configs/autoencoder_ensemble.yaml")
    parser.add_argument("--output", default="outputs/autoencoder")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    
    try:
        run_autoencoder_pipeline(args.config, args.output, args.resume)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
