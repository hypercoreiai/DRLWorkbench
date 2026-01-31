"""
PatchTST Crypto Price & Returns Forecasting Pipeline
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from src.utils import ConfigError
from src.utils.logging import setup_logging
from src.analysis.selection import select_features_correlation
from src.analysis.forecast_metrics import (
    regression_metrics,
    directional_accuracy,
    coverage_rate,
    horizon_metrics,
    information_ratio,
)
from src.data.patchtst_pipeline import (
    PatchTSTDataPipeline,
    StandardScaler,
    create_multistep_sequences,
    split_time_series,
)
from src.display.forecast_plots import (
    plot_forecast_vs_actual,
    plot_prediction_intervals,
    plot_error_distribution,
    plot_horizon_metrics,
    plot_calibration_curve,
    plot_correlation_heatmap,
    plot_feature_correlation_bar,
)
from src.models.patchtst import PatchTST, PatchTSTConfig, PatchTSTTrainer, predict_batches


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML: {e}") from e


def _make_dataloaders(
    features: pd.DataFrame,
    target: pd.Series,
    seq_len: int,
    pred_len: int,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
) -> Tuple[Dict[str, torch.utils.data.DataLoader], StandardScaler, StandardScaler, Dict[str, Any]]:
    split = split_time_series(features, target, train_ratio, val_ratio)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    feature_scaler.fit(split["train"][0])
    target_scaler.fit(split["train"][1])

    loaders = {}
    meta: Dict[str, Any] = {}
    for key, (f_df, t_ser) in split.items():
        f_scaled = feature_scaler.transform(f_df)
        t_scaled = target_scaler.transform(t_ser)
        X, y = create_multistep_sequences(f_scaled, t_scaled, seq_len, pred_len)
        if X.size == 0:
            raise ValueError("Not enough data to create sequences")
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X), torch.FloatTensor(y)
        )
        loaders[key] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(key == "train"),
            num_workers=0,
        )
        meta[f"{key}_index"] = t_ser.index[seq_len : len(t_ser) - pred_len + 1]
        meta[f"{key}_target_raw"] = t_ser.values
    return loaders, feature_scaler, target_scaler, meta


def _collect_targets(loader: torch.utils.data.DataLoader) -> np.ndarray:
    ys = []
    for _, y_batch in loader:
        ys.append(y_batch.numpy())
    return np.concatenate(ys, axis=0) if ys else np.empty((0,))


def _naive_forecast(target_values: np.ndarray, seq_len: int, pred_len: int) -> np.ndarray:
    preds = []
    for i in range(len(target_values) - seq_len - pred_len + 1):
        last_val = target_values[i + seq_len - 1]
        preds.append(np.full(pred_len, last_val))
    return np.array(preds)


def run_forecast_pipeline(config_path: str, output_dir: str) -> None:
    config = load_config(config_path)
    run_id = config.get("display", {}).get("run_id", "patchtst_forecast")
    out = Path(output_dir)
    log_dir = out / "logs"
    plot_dir = out / "plots"
    metrics_dir = out / "metrics"
    model_dir = out / "models"
    for d in [log_dir, plot_dir, metrics_dir, model_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(str(log_dir), run_id)
    logger.info("Starting PatchTST Forecasting Pipeline")
    logger.info("Config: %s", config_path)

    data_pipe = PatchTSTDataPipeline(config)
    data_bundle = data_pipe.run()

    features = data_bundle.features
    target_price = data_bundle.target_price
    target_returns = data_bundle.target_returns
    benchmarks = data_bundle.benchmarks

    # Feature selection
    sel_cfg = config.get("data", {}).get("feature_selection", {})
    threshold = sel_cfg.get("threshold", 0.05)
    selected = select_features_correlation(features, target_returns, threshold=threshold)
    if selected:
        features = features[selected]
    logger.info("Selected %d features", features.shape[1])

    # Correlation plots
    try:
        corr = features.corr()
        ax = plot_correlation_heatmap(corr, "Feature Correlation")
        ax.figure.tight_layout()
        ax.figure.savefig(plot_dir / "feature_corr_heatmap.png", dpi=150)
        ax = plot_feature_correlation_bar(features.corrwith(target_returns), "Feature vs Returns Correlation")
        ax.figure.tight_layout()
        ax.figure.savefig(plot_dir / "feature_vs_returns_corr.png", dpi=150)
    except Exception as e:
        logger.warning("Correlation plotting failed: %s", e)

    # Model configs
    seq_len = int(config.get("model", {}).get("seq_len", 504))
    pred_len = int(config.get("model", {}).get("pred_len", 30))
    train_ratio = float(config.get("data", {}).get("train_ratio", 0.7))
    val_ratio = float(config.get("data", {}).get("val_ratio", 0.15))
    batch_size = int(config.get("model", {}).get("batch_size", 64))
    epochs = int(config.get("model", {}).get("epochs", 50))

    # Price model
    loaders_price, _, target_scaler_price, meta_price = _make_dataloaders(
        features, target_price, seq_len, pred_len, train_ratio, val_ratio, batch_size
    )
    price_config = PatchTSTConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        n_features=features.shape[1],
        d_model=int(config.get("model", {}).get("d_model", 256)),
        n_heads=int(config.get("model", {}).get("n_heads", 8)),
        n_encoder_layers=int(config.get("model", {}).get("n_encoder_layers", 6)),
        d_ff=int(config.get("model", {}).get("d_ff", 512)),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
        patch_len=int(config.get("model", {}).get("patch_len", 16)),
        stride=int(config.get("model", {}).get("stride", 8)),
    )
    price_model = PatchTST(price_config)
    price_trainer = PatchTSTTrainer(price_model, loaders_price["train"], loaders_price["val"], price_config)
    price_history = price_trainer.train(epochs=epochs, early_stopping_patience=15)
    price_trainer.save_model(str(model_dir / "patchtst_price.pt"))

    # Returns model
    loaders_ret, _, target_scaler_ret, meta_ret = _make_dataloaders(
        features, target_returns, seq_len, pred_len, train_ratio, val_ratio, batch_size
    )
    ret_config = PatchTSTConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        n_features=features.shape[1],
        d_model=int(config.get("model", {}).get("d_model", 256)),
        n_heads=int(config.get("model", {}).get("n_heads", 8)),
        n_encoder_layers=int(config.get("model", {}).get("n_encoder_layers", 6)),
        d_ff=int(config.get("model", {}).get("d_ff", 512)),
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
        patch_len=int(config.get("model", {}).get("patch_len", 16)),
        stride=int(config.get("model", {}).get("stride", 8)),
    )
    ret_model = PatchTST(ret_config)
    ret_trainer = PatchTSTTrainer(ret_model, loaders_ret["train"], loaders_ret["val"], ret_config)
    ret_history = ret_trainer.train(epochs=epochs, early_stopping_patience=15)
    ret_trainer.save_model(str(model_dir / "patchtst_returns.pt"))

    # Predictions
    price_val_pred = predict_batches(price_model, loaders_price["val"])
    price_test_pred = predict_batches(price_model, loaders_price["test"])
    price_val_true = _collect_targets(loaders_price["val"])
    price_test_true = _collect_targets(loaders_price["test"])

    ret_val_pred = predict_batches(ret_model, loaders_ret["val"])
    ret_test_pred = predict_batches(ret_model, loaders_ret["test"])
    ret_val_true = _collect_targets(loaders_ret["val"])
    ret_test_true = _collect_targets(loaders_ret["test"])

    # Inverse transform targets
    price_val_pred = target_scaler_price.inverse_transform(price_val_pred)
    price_test_pred = target_scaler_price.inverse_transform(price_test_pred)
    price_val_true = target_scaler_price.inverse_transform(price_val_true)
    price_test_true = target_scaler_price.inverse_transform(price_test_true)

    ret_val_pred = target_scaler_ret.inverse_transform(ret_val_pred)
    ret_test_pred = target_scaler_ret.inverse_transform(ret_test_pred)
    ret_val_true = target_scaler_ret.inverse_transform(ret_val_true)
    ret_test_true = target_scaler_ret.inverse_transform(ret_test_true)

    # Prediction intervals from validation residuals
    price_resid = price_val_true - price_val_pred
    ret_resid = ret_val_true - ret_val_pred
    price_std = price_resid.std(axis=0)
    ret_std = ret_resid.std(axis=0)
    z80, z95 = 1.2816, 1.96
    price_lower_80 = price_test_pred - z80 * price_std
    price_upper_80 = price_test_pred + z80 * price_std
    price_lower_95 = price_test_pred - z95 * price_std
    price_upper_95 = price_test_pred + z95 * price_std
    ret_lower_80 = ret_test_pred - z80 * ret_std
    ret_upper_80 = ret_test_pred + z80 * ret_std
    ret_lower_95 = ret_test_pred - z95 * ret_std
    ret_upper_95 = ret_test_pred + z95 * ret_std

    # Metrics
    metrics = []
    price_metrics = regression_metrics(price_test_true.flatten(), price_test_pred.flatten())
    price_dir = directional_accuracy(np.diff(price_test_true, axis=1), np.diff(price_test_pred, axis=1))
    price_cov_80 = coverage_rate(price_test_true, price_lower_80, price_upper_80)
    price_cov_95 = coverage_rate(price_test_true, price_lower_95, price_upper_95)
    price_horizon = horizon_metrics(price_test_true, price_test_pred)
    metrics.append({
        "model": "patchtst_price",
        **price_metrics,
        "directional_accuracy": price_dir,
        "coverage_80": price_cov_80,
        "coverage_95": price_cov_95,
    })

    ret_metrics = regression_metrics(ret_test_true.flatten(), ret_test_pred.flatten())
    ret_dir = directional_accuracy(ret_test_true.flatten(), ret_test_pred.flatten())
    ret_cov_80 = coverage_rate(ret_test_true, ret_lower_80, ret_upper_80)
    ret_cov_95 = coverage_rate(ret_test_true, ret_lower_95, ret_upper_95)
    ret_horizon = horizon_metrics(ret_test_true, ret_test_pred)
    metrics.append({
        "model": "patchtst_returns",
        **ret_metrics,
        "directional_accuracy": ret_dir,
        "coverage_80": ret_cov_80,
        "coverage_95": ret_cov_95,
    })

    # Naive baseline
    price_naive = _naive_forecast(meta_price["test_target_raw"], seq_len, pred_len)
    ret_naive = _naive_forecast(meta_ret["test_target_raw"], seq_len, pred_len)
    price_naive_metrics = regression_metrics(price_test_true.flatten(), price_naive.flatten())
    ret_naive_metrics = regression_metrics(ret_test_true.flatten(), ret_naive.flatten())
    metrics.append({"model": "naive_price", **price_naive_metrics})
    metrics.append({"model": "naive_returns", **ret_naive_metrics})

    # Information ratio for returns vs benchmark (horizon 1)
    if not benchmarks.empty:
        bench_series = benchmarks.iloc[-len(meta_ret["test_index"]):, 0]
        ir = information_ratio(ret_test_pred[:, 0], bench_series.values)
        metrics.append({"model": "patchtst_returns", "information_ratio": ir})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(metrics_dir / "forecast_metrics.csv", index=False)

    # Calibration plot
    calib_nominal = [0.8, 0.95]
    calib_empirical = [ret_cov_80, ret_cov_95]
    ax = plot_calibration_curve(calib_nominal, calib_empirical, "Returns Interval Calibration")
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "returns_calibration.png", dpi=150)

    # Plot forecast vs actual (horizon 1)
    price_idx = meta_price["test_index"]
    ret_idx = meta_ret["test_index"]
    price_actual_series = pd.Series(price_test_true[:, 0], index=price_idx)
    price_pred_series = pd.Series(price_test_pred[:, 0], index=price_idx)
    ax = plot_forecast_vs_actual(price_actual_series, price_pred_series, "Price Forecast (H1)")
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "price_forecast_h1.png", dpi=150)

    ret_actual_series = pd.Series(ret_test_true[:, 0], index=ret_idx)
    ret_pred_series = pd.Series(ret_test_pred[:, 0], index=ret_idx)
    ax = plot_forecast_vs_actual(ret_actual_series, ret_pred_series, "Returns Forecast (H1)")
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "returns_forecast_h1.png", dpi=150)

    # Prediction intervals for returns (H1)
    ret_lower_series = pd.Series(ret_lower_95[:, 0], index=ret_idx)
    ret_upper_series = pd.Series(ret_upper_95[:, 0], index=ret_idx)
    ax = plot_prediction_intervals(
        ret_actual_series,
        ret_pred_series,
        ret_lower_series,
        ret_upper_series,
        "Returns Forecast w/ 95% Interval (H1)",
    )
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "returns_interval_h1.png", dpi=150)

    # Error distributions
    ax = plot_error_distribution((price_test_true - price_test_pred).flatten(), "Price Errors")
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "price_error_hist.png", dpi=150)

    ax = plot_error_distribution((ret_test_true - ret_test_pred).flatten(), "Returns Errors")
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "returns_error_hist.png", dpi=150)

    # Horizon metrics
    ax = plot_horizon_metrics(price_horizon, "Price Horizon Metrics")
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "price_horizon_metrics.png", dpi=150)

    ax = plot_horizon_metrics(ret_horizon, "Returns Horizon Metrics")
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "returns_horizon_metrics.png", dpi=150)

    # Save training history
    pd.DataFrame(price_history).to_csv(metrics_dir / "price_training_history.csv", index=False)
    pd.DataFrame(ret_history).to_csv(metrics_dir / "returns_training_history.csv", index=False)

    logger.info("PatchTST pipeline completed. Outputs in: %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PatchTST forecast pipeline")
    parser.add_argument(
        "--config",
        default="configs/forecast_patchtst.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--output",
        default="outputs/forecast_patchtst",
        help="Output directory",
    )
    args = parser.parse_args()
    run_forecast_pipeline(args.config, args.output)


if __name__ == "__main__":
    main()