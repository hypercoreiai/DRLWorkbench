# PatchTST Crypto Price and Returns Forecaster - Implementation Outline

## Project Overview
Build a PatchTST (Patched Time Series Transformer) based forecasting system for cryptocurrency prices and returns with:
- Feature engineering and preprocessing
- Correlation analysis and feature selection
- Walk-forward backtesting
- 30x forecast horizon (30 periods ahead of original data interval)
- Comprehensive accuracy metrics at 80% and 95% confidence bands

---

## Phase 1: Data Acquisition and Preprocessing

### 1.1 Data Collection
```
├── Download OHLCV data (Open, High, Low, Close, Volume)
├── Ticker selection: src\symbols\portfolio
├── Time interval: Hourly, 4-hourly, daily (configurable)
├── Data validation and cleaning
│   ├── Handle missing values (forward fill, interpolation)
│   ├── Remove duplicate entries
│   ├── Detect and handle outliers
│   └── Check data continuity
└── Data normalization (standard scaling, log returns)
```

### 1.2 Feature Engineering
```
Technical Indicators:
├── Momentum Indicators
│   ├── RSI (Relative Strength Index) - 14, 21, 28
│   ├── Stochastic RSI - %K, %D
│   ├── MACD (Moving Average Convergence Divergence)
│   │   ├── MACD line
│   │   ├── Signal line
│   │   └── Histogram
│   ├── Rate of Change (ROC) - 12, 24
│   └── Momentum - 10, 20
│
├── Trend Indicators
│   ├── SMA (Simple Moving Average) - 20, 50, 100, 200
│   ├── EMA (Exponential Moving Average) - 12, 26
│   ├── ADX (Average Directional Index)
│   ├── Linear Regression Slope - 20, 50
│   └── TEMA (Triple Exponential Moving Average) - 14
│
├── Volatility Indicators
│   ├── Bollinger Bands
│   │   ├── Upper band
│   │   ├── Lower band
│   │   └── %B (position within bands)
│   ├── ATR (Average True Range) - 14
│   ├── Standard Deviation - 20
│   └── Coefficient of Variation
│
├── Volume-Based Indicators
│   ├── Volume Rate of Change
│   ├── OBV (On-Balance Volume)
│   ├── CMF (Chaikin Money Flow)
│   ├── Volume SMA ratio
│   └── VWAP (Volume Weighted Average Price)
│
├── Price Action Features
│   ├── Returns (1, 5, 10, 20 periods)
│   ├── Log Returns
│   ├── High-Low ratio
│   ├── Close-Open ratio
│   ├── True Range
│   └── Upper/Lower shadow ratio
│
└── Statistical Features
    ├── Rolling skewness - 20, 50
    ├── Rolling kurtosis - 20, 50
    ├── Auto-correlation - lag 1-5
    └── Cross-correlation with volume
```

### 1.3 Target Variable Creation
```
Target Variables:
├── Next Period Close Price
├── Next Period Returns (%)
├── Next Period Log Returns
├── 30-Period Ahead Price
├── 30-Period Ahead Returns
└── Direction Classification (Up/Down)
```

### 1.4 Data Structure Output
```
Final Dataset Format:
├── Index: Timestamp
├── OHLCV columns (5 features)
├── Technical indicators (40-50 features)
├── Statistical features (8-10 features)
├── Target variables (5 features)
└── Data shape: (n_samples, 50-70 features)

Missing Values Handling:
├── Drop initial NaN rows (indicator lookback period)
├── Final clean dataset shape: (n_samples - max_lookback, features)
└── Data quality report: % missing, date range, etc.
```

---

## Phase 2: Feature Selection and Correlation Analysis

### 2.1 Correlation Analysis
```
Correlation Computation:
├── Pearson Correlation (linear relationships)
│   ├── Compute correlation matrix (all features vs target)
│   ├── Correlation with target variables
│   └── Feature-to-feature correlation matrix
│
├── Spearman Correlation (rank-based relationships)
│   └── Robust to outliers
│
└── Mutual Information
    └── Non-linear relationship detection
```

### 2.2 Feature Selection Strategies
```
Method 1: Correlation Threshold
├── Remove features with |correlation| < 0.05 with target
├── Remove highly correlated feature pairs (|r| > 0.95)
└── Keep only top 30-40 features by correlation strength

Method 2: Statistical Tests
├── F-statistic (ANOVA)
├── Chi-squared test (for categorical features)
└── Select top features by p-value

Method 3: Variance Inflation Factor (VIF)
├── Detect multicollinearity
├── Remove features with VIF > 5
└── Ensure feature independence

Method 4: Recursive Feature Elimination (RFE)
├── Train lightweight model
├── Iteratively remove weak features
└── Select top 30-40 features
```

### 2.3 Correlation Visualization
```
Output Visualizations:
├── Heatmap: Full correlation matrix
├── Bar plot: Feature correlation with target
├── Scatter plots: Top features vs target (top 12)
├── VIF bar plot: Multicollinearity detection
├── Feature importance plot: Ranked by correlation
└── Correlation change analysis (before/after selection)
```

### 2.4 Feature Selection Report
```
Report Contents:
├── Total features before selection: N
├── Total features after selection: M
├── Removed features and reasons
├── Top 10 features by correlation strength
├── Correlation statistics (mean, median, std)
├── Multicollinearity assessment
└── Feature selection efficiency: M/N ratio
```

---

## Phase 3: Model Architecture - PatchTST

### 3.1 PatchTST Architecture Overview
```
Input: (batch, seq_len, n_features) 
       [e.g., (32, 504, 35) = 32 samples, 504 timesteps, 35 features]
       
├── Patching Layer
│   ├── Divide sequence into patches
│   ├── Patch size: 16 (504 / 16 = 31.5 patches)
│   ├── Overlap: optional (8 samples)
│   └── Output: (batch, n_patches, patch_embed_dim)
│
├── Projection Layer
│   ├── Linear projection: d_model (128-256)
│   └── Positional encoding addition
│
├── Transformer Encoder Stack (6-8 layers)
│   ├── Multi-head self-attention (8-16 heads)
│   │   ├── Query, Key, Value projections
│   │   ├── Scaled dot-product attention
│   │   └── Multi-head concatenation
│   │
│   ├── Feed-forward network (FFN)
│   │   ├── Linear: d_model -> d_ff (512-1024)
│   │   ├── Activation: GELU/ReLU
│   │   └── Linear: d_ff -> d_model
│   │
│   ├── Normalization: LayerNorm
│   └── Residual connections
│
├── Flattening Layer
│   └── Flatten patch sequence: (batch, n_patches * d_model)
│
├── Forecast Head
│   ├── Hidden layers: 1-2 (256, 128)
│   ├── Dropout: 0.1-0.2
│   ├── Activation: GELU/ReLU
│   └── Output layer: (batch, forecast_len)
│
└── Output: (batch, horizon=30)
    [30 timesteps ahead predictions]

Model Hyperparameters:
├── Patch size: 16
├── d_model (embedding dim): 256
├── n_heads: 8
├── n_encoder_layers: 6
├── d_ff: 512
├── dropout: 0.1
├── activation: 'gelu'
└── max_seq_len: 504 (3 weeks @ daily, or 7 days @ hourly)
```

### 3.2 Multi-Task Learning Variant (Optional)
```
Instead of single output, predict:
├── Task 1: 30-period price forecast
├── Task 2: 30-period returns forecast
├── Task 3: Volatility forecast (30-period std)
├── Shared encoder + task-specific decoders
└── Combined loss: L = loss_price + loss_returns + loss_volatility
```

---

## Phase 4: Data Preparation for Model Training

### 4.1 Sequence Generation
```
Generate overlapping sequences from time series:
├── Input sequence length: 504 timesteps (lookback window)
├── Output sequence length: 30 timesteps (forecast horizon)
├── Stride: 1 (maximum overlap for more samples)
│
├── Example:
│   ├── Sequence 1: [t=0:504] -> target [t=504:534]
│   ├── Sequence 2: [t=1:505] -> target [t=505:535]
│   ├── ...
│   └── Sequence N: [t=n:n+504] -> target [t=n+504:n+534]
│
└── Total sequences generated: len(data) - 504 - 30 + 1
```

### 4.2 Train-Validation-Test Split
```
Time-Series Split (No Data Leakage):
├── Total data: 100 weeks (configurable)
│
├── Split Method: Time-based walk-forward
│   ├── Training: 0-70 weeks (70%)
│   ├── Validation: 70-85 weeks (15%)
│   └── Test: 85-100 weeks (15%)
│
├── Normalization Strategy:
│   ├── Fit scaler on training data only
│   ├── Apply same scaler to validation and test
│   └── Inverse transform predictions using training scaler
│
└── Data shapes:
    ├── Train sequences: ~19,000 samples (504+30 window)
    ├── Val sequences: ~5,600 samples
    └── Test sequences: ~5,600 samples
```

### 4.3 DataLoader Creation
```
PyTorch DataLoaders:
├── Train DataLoader
│   ├── Batch size: 64
│   ├── Shuffle: True
│   └── Num workers: 4
│
├── Validation DataLoader
│   ├── Batch size: 256
│   ├── Shuffle: False
│   └── Num workers: 2
│
└── Test DataLoader
    ├── Batch size: 256
    ├── Shuffle: False
    └── Num workers: 2
```

---

## Phase 5: Model Training

### 5.1 Training Setup
```
Loss Function:
├── MSE Loss (Mean Squared Error) - primary
├── MAE Loss (Mean Absolute Error) - alternative
└── Huber Loss - robust to outliers

Optimization:
├── Optimizer: Adam
│   ├── Learning rate: 0.001 (scheduler: ReduceLROnPlateau)
│   ├── Beta1: 0.9
│   ├── Beta2: 0.999
│   └── Weight decay: 1e-4
│
└── Scheduler:
    ├── ReduceLROnPlateau: reduce LR if val loss plateaus
    ├── Patience: 10 epochs
    └── Factor: 0.5

Regularization:
├── Dropout: 0.1
├── Weight decay: 1e-4
├── Early stopping: patience=20 epochs
└── Batch normalization: after projection layer
```

### 5.2 Training Loop
```
For each epoch (max 200):
├── Training phase:
│   ├── Forward pass through model
│   ├── Compute loss
│   ├── Backward pass
│   ├── Gradient clipping (max_norm=1.0)
│   ├── Optimizer step
│   └── Log metrics
│
├── Validation phase (every epoch):
│   ├── No gradient computation
│   ├── Forward pass on validation set
│   ├── Compute validation loss
│   ├── Check early stopping criterion
│   └── Save best model weights
│
└── Checkpointing:
    ├── Save best model: based on val_loss
    ├── Save latest model: every N epochs
    └── Save metrics history (train/val loss)
```

### 5.3 Training Monitoring
```
Metrics to track:
├── Train loss (MSE/MAE)
├── Validation loss
├── Learning rate
├── Gradient norm
├── Batch time
└── Epoch time

Logging:
├── Console output: every N batches
├── TensorBoard: continuous metrics
│   ├── Loss curves (train/val)
│   ├── Learning rate schedule
│   ├── Gradient magnitude
│   └── Prediction samples (validation)
│
└── File logging: metrics.csv
    ├── epoch, train_loss, val_loss, lr
    ├── best_epoch, best_val_loss
    └── training_time
```

---

## Phase 6: Backtesting - Walk-Forward Validation

### 6.1 Walk-Forward Backtesting Strategy
```
Purpose: Simulate real-world trading with continuous model retraining

Setup:
├── Initial training window: 60 weeks
├── Walk-forward step: 1 week (5 days)
├── Test window: 1 week (generate 30-period forecasts)
├── Total analysis period: 100 weeks
│
└── Process:
    Week 1-60:   Train model
    Week 61:     Generate 30-day forecast, compare to actual
    Week 2-61:   Retrain model with new data (rolling window)
    Week 62:     Generate 30-day forecast, compare to actual
    ...
    Week 95:     Generate final 30-day forecast
    
Total walk-forward iterations: (100 - 60 - 30) / 1 ≈ 10 iterations
```

### 6.2 Implementation Details
```
For each walk-forward iteration:
│
├── Prepare data:
│   ├── Training set: 60 weeks before test period
│   ├── Test set: current week (1 week = 5/7 days of data)
│   └── Forecast horizon: 30 periods ahead
│
├── Fit scaler (on training data):
│   └── Store scaler for inverse transforms
│
├── Check if model needs retraining:
│   ├── Retrain from scratch: every iteration (option 1)
│   ├── Warm-start: load previous model (option 2)
│   └── Fine-tune: retrain last N layers (option 3)
│
├── Generate forecast:
│   ├── Use trained model
│   ├── Input: last 504 timesteps of training data
│   ├── Output: 30-step ahead predictions
│   └── Inverse transform to original scale
│
├── Store results:
│   ├── Predicted values
│   ├── Actual values (realized in next 30 periods)
│   ├── Forecast date
│   ├── Forecast metrics
│   └── Prediction intervals (confidence bands)
│
└── Performance metrics (per iteration):
    ├── MAE, RMSE, MAPE
    ├── Directional accuracy
    ├── Coverage of prediction intervals
    └── Sharpe ratio (if applicable)
```

### 6.3 Prediction Intervals (Confidence Bands)
```
Method 1: Quantile Regression
├── Train additional quantile regressors:
│   ├── q=0.025 (2.5th percentile - lower 95% bound)
│   ├── q=0.05 (5th percentile - lower 90% bound)
│   ├── q=0.50 (50th percentile - median/point forecast)
│   ├── q=0.95 (95th percentile - upper 90% bound)
│   └── q=0.975 (97.5th percentile - upper 95% bound)
│
└── Output: 3-tuple (lower_95, point_forecast, upper_95)

Method 2: Bootstrap / MC Dropout
├── Enable dropout during inference
├── Run inference N times (e.g., 100)
├── Collect N predictions per timestep
├── Compute percentiles from distribution
│   ├── 2.5th percentile -> lower 95% band
│   ├── 50th percentile -> point forecast
│   └── 97.5th percentile -> upper 95% band
│
└── Also compute:
    ├── Standard deviation per timestep
    └── Coefficient of variation

Method 3: Parametric (Assume Normal Distribution)
├── Point forecast: μ
├── Residual std: σ (from validation)
├── 95% CI: μ ± 1.96*σ
├── 80% CI: μ ± 1.28*σ
└── Calculate using training residuals

Method 4: Non-Parametric (Empirical)
├── Collect residuals from validation/test
├── Sort residuals by absolute value
├── Use empirical quantiles:
│   ├── lower_95 = forecast + residual_quantile(0.025)
│   ├── upper_95 = forecast + residual_quantile(0.975)
│   ├── lower_80 = forecast + residual_quantile(0.10)
│   └── upper_80 = forecast + residual_quantile(0.90)
│
└── Preferred for non-normal distributions
```

---

## Phase 7: Accuracy Metrics and Evaluation

### 7.1 Point Forecast Metrics
```
For each prediction in test set:

1. Mean Absolute Error (MAE)
   ├── Formula: MAE = (1/n) * Σ|y_actual - y_pred|
   ├── Units: Same as target (USD price or %)
   ├── Interpretation: Average prediction error
   └── Goal: Minimize

2. Root Mean Squared Error (RMSE)
   ├── Formula: RMSE = √[(1/n) * Σ(y_actual - y_pred)²]
   ├── Units: Same as target
   ├── Interpretation: Penalizes large errors
   └── Goal: Minimize

3. Mean Absolute Percentage Error (MAPE)
   ├── Formula: MAPE = (1/n) * Σ|((y_actual - y_pred) / y_actual) * 100|
   ├── Units: Percentage (%)
   ├── Interpretation: Relative error (scale-independent)
   ├── Range: 0-100% (lower is better)
   └── Goal: Minimize
   ├── Threshold interpretation:
   │   ├── < 5%: Excellent accuracy
   │   ├── 5-10%: Good accuracy
   │   ├── 10-20%: Fair accuracy
   │   └── > 20%: Poor accuracy

4. Symmetric Mean Absolute Percentage Error (SMAPE)
   ├── Formula: SMAPE = (100/n) * Σ(2 * |y_actual - y_pred| / (|y_actual| + |y_pred|))
   ├── Units: Percentage (%)
   ├── Range: 0-200% (0-100% more typical)
   ├── Advantage: Symmetric and bounded
   └── Goal: Minimize

5. Mean Absolute Scaled Error (MASE)
   ├── Formula: MASE = MAE / MAE_naive
   ├── Where MAE_naive = baseline model (e.g., persistence/random walk)
   ├── Interpretation: 
   │   ├── MASE < 1: Better than baseline
   │   ├── MASE = 1: Same as baseline
   │   └── MASE > 1: Worse than baseline
   └── Goal: Minimize (target: MASE < 0.7)

6. R² Score (Coefficient of Determination)
   ├── Formula: R² = 1 - (SS_res / SS_tot)
   ├── Where:
   │   ├── SS_res = Σ(y_actual - y_pred)²
   │   ├── SS_tot = Σ(y_actual - mean(y_actual))²
   ├── Range: -∞ to 1 (higher is better)
   ├── Interpretation:
   │   ├── R² = 1: Perfect fit
   │   ├── R² = 0.7-0.9: Very good
   │   ├── R² = 0.5-0.7: Good
   │   └── R² < 0.5: Poor
   └── Goal: Maximize (target: > 0.7)
```

### 7.2 Directional Accuracy Metrics
```
Classification-based metrics (for directional forecasts):

1. Directional Accuracy
   ├── Definition: % of time prediction direction matches actual
   ├── Formula: DA = (1/n) * Σ(sign(y_actual - y_prev) == sign(y_pred - y_prev))
   ├── Range: 0-100%
   ├── Interpretation:
   │   ├── > 50%: Better than random
   │   ├── 50-55%: Weak signal
   │   ├── 55-60%: Good (tradeable)
   │   └── > 60%: Excellent
   └── Goal: Maximize (target: > 55%)

2. Confusion Matrix & Accuracy
   ├── True Positives (TP): Predicted Up, Actual Up
   ├── True Negatives (TN): Predicted Down, Actual Down
   ├── False Positives (FP): Predicted Up, Actual Down
   ├── False Negatives (FN): Predicted Down, Actual Up
   │
   ├── Accuracy: (TP + TN) / (TP + TN + FP + FN)
   ├── Precision: TP / (TP + FP) - when we say up, how often right?
   ├── Recall: TP / (TP + FN) - how often do we catch ups?
   ├── F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
   └── Goal: Maximize all metrics

3. Matthews Correlation Coefficient (MCC)
   ├── Formula: MCC = (TP*TN - FP*FN) / √((TP+FP)(TP+FN)(TN+FP)(TN+FN))
   ├── Range: -1 to +1
   ├── Interpretation:
   │   ├── +1: Perfect correlation
   │   ├── 0: Random prediction
   │   └── -1: Perfect negative correlation
   ├── Advantage: Better for imbalanced classes
   └── Goal: Maximize (target: > 0.2)
```

### 7.3 Prediction Interval Metrics (Confidence Bands)
```
1. Interval Coverage Rate (ICR)
   ├── Definition: % of actual values falling within prediction interval
   ├── Formula: ICR = (1/n) * Σ(y_actual in [lower_bound, upper_bound])
   ├── Interpretation:
   │   ├── For 95% band: Target ≈ 95% (acceptable: 90-99%)
   │   ├── For 80% band: Target ≈ 80% (acceptable: 75-85%)
   │   └── Under-coverage: Too narrow bands
   │   └── Over-coverage: Too wide bands (uninformative)
   │
   └── Expected coverage by confidence level:
       ├── 68%: 68% of values (±1σ)
       ├── 80%: 80% of values (±1.28σ)
       └── 95%: 95% of values (±1.96σ)

2. Mean Interval Width (MIW)
   ├── Definition: Average width of prediction intervals
   ├── Formula: MIW = (1/n) * Σ(upper_bound - lower_bound)
   ├── Interpretation:
   │   ├── Narrower = more informative
   │   ├── Wider = less precise but safer
   │   └── Trade-off: Coverage vs Width
   │
   └── Optimal: Maximum coverage with minimum width

3. Prediction Interval Normalized Average Width (PINAW)
   ├── Formula: PINAW = (1/n) * Σ(upper - lower) / (max(y) - min(y))
   ├── Range: 0-1 (unitless)
   ├── Interpretation:
   │   ├── Normalized to data range
   │   ├── PINAW < 0.5: Good precision
   │   └── PINAW > 1.0: Overly conservative
   │
   └── Goal: Maximize coverage while minimizing PINAW

4. Sharpness (Sharp confidence bands)
   ├── Definition: How concentrated predictions are
   ├── Measured by: Coefficient of variation of interval widths
   ├── Formula: Sharpness = std(upper - lower) / mean(upper - lower)
   ├── Interpretation:
   │   ├── Low sharpness: Consistent intervals
   │   ├── High sharpness: Highly variable intervals
   │   └── Goal: Low sharpness + high coverage
   │
   └── Alternative: Interquartile range of widths

5. Calibration Analysis
   ├── For each confidence level (80%, 95%):
   │   ├── Compute empirical coverage (% of values in band)
   │   ├── Plot calibration curve: Nominal vs Empirical coverage
   │   ├── Perfect: 45-degree line
   │   └── Assess: Over/under-confident
   │
   ├── Calibration error: |Nominal - Empirical|
   ├── Target: < 5% for well-calibrated predictions
   └── Generate calibration report

6. Continuous Ranked Probability Score (CRPS)
   ├── Definition: Measure forecast probability distribution quality
   ├── Formula: CRPS = ∫(F(y) - H(y - y_actual))² dy
   │   ├── F(y) = predicted CDF
   │   ├── H = Heaviside step function
   │   └── Approximate using prediction samples
   │
   ├── Range: 0 to ∞ (lower is better)
   ├── Interpretation: Average distance from true value in CDF
   └── Goal: Minimize
```

### 7.4 Returns Forecast Metrics (if predicting returns)
```
If predicting percentage returns instead of prices:

1. Return Distribution Analysis
   ├── Mean predicted return vs actual
   ├── Std dev of returns (volatility forecast)
   ├── Skewness (tail risk)
   └── Kurtosis (extreme events)

2. Excess Return (for crypto with risk-free rate ~0)
   ├── Predicted excess return
   └── Compare to baseline models

3. Information Ratio
   ├── Formula: IR = (Return_forecast - Return_benchmark) / Tracking_Error
   ├── Interpretation: Excess return per unit of tracking error
   └── Goal: Maximize (target: > 0.5)

4. Maximum Drawdown
   ├── Using predicted returns
   ├── Compare to actual max drawdown
   ├── Goal: Minimize (lower is safer)
   └── Acceptable: < 20% for crypto

5. Calmar Ratio
   ├── Formula: Calmar = Annual Return / Max Drawdown
   ├── Interpretation: Return per unit of downside risk
   └── Goal: Maximize (target: > 1.0)

6. Win Rate (for directional returns)
   ├── % of periods with predicted return direction correct
   ├── Equivalent to directional accuracy
   └── Target: > 52% (slightly better than coin flip)
```

### 7.5 Residual Analysis
```
Analyze prediction errors for model diagnostics:

1. Residual Statistics
   ├── Mean residual (should be ≈ 0)
   ├── Std dev of residuals (error magnitude)
   ├── Min/Max residuals (largest errors)
   ├── Skewness (bias toward over/under predictions)
   └── Kurtosis (frequency of extreme errors)

2. Residual Plots
   ├── Histogram: Check normality (for parametric CI)
   ├── Q-Q plot: Compare to normal distribution
   ├── ACF plot: Check autocorrelation (should be random)
   ├── Time series: Residuals over time (check patterns)
   └── Residuals vs Fitted: Check homoscedasticity

3. Autocorrelation Tests
   ├── Ljung-Box test: Check if residuals are white noise
   │   ├── H0: Residuals are independent
   │   ├── H1: Residuals have autocorrelation
   │   └── Target: p-value > 0.05 (accept H0)
   │
   └── Durbin-Watson statistic:
       ├── Range: 0-4
       ├── 2: No autocorrelation
       ├── < 2: Positive correlation
       └── > 2: Negative correlation

4. Error Distribution Tests
   ├── Shapiro-Wilk test: Normality
   │   ├── H0: Data is normal
   │   └── Target: p-value > 0.05 (for parametric CI)
   │
   ├── Anderson-Darling test: Normality (more sensitive)
   └── Jarque-Bera test: Skewness and Kurtosis

5. Heteroscedasticity Tests
   ├── Breusch-Pagan test: Constant variance
   │   ├── H0: Homoscedastic (constant variance)
   │   └── Target: p-value > 0.05
   │
   └── White test: General heteroscedasticity
```

### 7.6 Comparative Analysis
```
Comparison Metrics:

1. Baseline Models
   ├── Naive (Persistence): y_pred = y_last
   ├── Seasonal Naive: y_pred = y_(t-seasonal_period)
   ├── ARIMA: Statistical model
   ├── Exponential Smoothing: Alternative statistical
   └── Linear Regression: Simple ML baseline
   
   Purpose: Ensure PatchTST beats baselines

2. Model Comparison Table
   ├── PatchTST vs each baseline
   ├── Metrics: MAE, RMSE, MAPE, SMASE, R², DA
   ├── Statistical significance tests (t-tests)
   └── % improvement over best baseline

3. Sensitivity Analysis
   ├── Impact of sequence length (500, 504, 512, 1000)
   ├── Impact of forecast horizon (1, 7, 14, 30)
   ├── Impact of feature count (20, 35, 50 features)
   ├── Impact of batch size (32, 64, 128)
   └── Impact of learning rate (0.0001, 0.001, 0.01)

4. Forecast Horizon Analysis
   ├── Accuracy degradation over forecast steps:
   │   ├── Step 1: y_hat(t+1) - accuracy level A1
   │   ├── Step 5: y_hat(t+5) - accuracy level A5
   │   ├── Step 15: y_hat(t+15) - accuracy level A15
   │   └── Step 30: y_hat(t+30) - accuracy level A30
   │
   ├── Plot: Accuracy vs Forecast horizon
   ├── Identify "accuracy horizon" (where accuracy drops below threshold)
   └── Recommend: Use predictions up to X steps ahead only
```

---

## Phase 8: Results Analysis and Reporting

### 8.1 Comprehensive Results Summary
```
Report Structure:

1. Executive Summary
   ├── Model name: PatchTST
   ├── Crypto asset: BTC-USD, ETH-USD, etc.
   ├── Data period: Start date - End date
   ├── Feature count: N features (selected from M)
   ├── Key findings:
   │   ├── Best accuracy metric: MAPE = X.XX%
   │   ├── Best directional accuracy: XX%
   │   ├── Prediction interval coverage: 95.2% @ 95% CI
   │   └── Recommended use: Forecasts reliable up to X steps ahead
   │
   └── Conclusion: Suitable for trading / Requires refinement

2. Data Overview
   ├── Dataset: OHLCV from crypto exchange
   ├── Time period: N weeks / months
   ├── Number of samples: N
   ├── Feature engineering: M indicators created
   ├── Feature selection: N features selected, X removed
   ├── Missing data: X% (after handling)
   ├── Outliers detected: Y
   └── Data quality score: Z/100

3. Feature Analysis
   ├── Top 10 features by correlation:
   │   ├── Feature 1: correlation = 0.XX
   │   ├── Feature 2: correlation = 0.XX
   │   └── ...
   │
   ├── Feature groups:
   │   ├── Momentum: N features
   │   ├── Trend: N features
   │   ├── Volatility: N features
   │   ├── Volume: N features
   │   └── Price action: N features
   │
   ├── Multicollinearity assessment: VIF analysis
   └── Feature importance ranking

4. Model Architecture & Training
   ├── Architecture details
   ├── Total parameters: N
   ├── Trainable parameters: M
   ├── Training epochs: N
   ├── Final validation loss: X.XXXX
   ├── Training time: Z minutes
   ├── Best epoch: N
   ├── Early stopping: Triggered at epoch N
   └── Learning rate schedule: ReduceLROnPlateau

5. Backtesting Results
   ├── Backtesting period: Start - End (N weeks)
   ├── Walk-forward iterations: N
   ├── Retraining frequency: Every N days
   │
   ├── Overall Performance:
   │   ├── MAE: X.XX (USD or %)
   │   ├── RMSE: X.XX
   │   ├── MAPE: X.XX%
   │   ├── SMAPE: X.XX%
   │   ├── MASE: X.XX
   │   ├── R²: X.XX
   │   ├── Directional Accuracy: XX.XX%
   │   └── Win Rate: XX.XX%
   │
   ├── By Forecast Step:
   │   ├── Step 1: Metrics
   │   ├── Step 10: Metrics
   │   ├── Step 20: Metrics
   │   └── Step 30: Metrics
   │
   ├── By Market Condition:
   │   ├── Uptrend periods: Metrics
   │   ├── Downtrend periods: Metrics
   │   ├── High volatility: Metrics
   │   └── Low volatility: Metrics
   │
   └── Walk-forward progression:
       ├── Iteration 1: MAE, RMSE, MAPE
       ├── Iteration 2: MAE, RMSE, MAPE
       └── ... (show trend)

6. Prediction Interval Analysis
   ├── 80% Confidence Bands:
   │   ├── Coverage rate: XX.X% (target: 80%)
   │   ├── Mean interval width: X.XX
   │   ├── PINAW: X.XX
   │   ├── Calibration: Nominal 80%, Empirical XX%
   │   └── Assessment: Well/Over/Under-calibrated
   │
   ├── 95% Confidence Bands:
   │   ├── Coverage rate: XX.X% (target: 95%)
   │   ├── Mean interval width: X.XX
   │   ├── PINAW: X.XX
   │   ├── Calibration: Nominal 95%, Empirical XX%
   │   └── Assessment: Well/Over/Under-calibrated
   │
   └── Sharpness analysis:
       ├── Interval width consistency
       ├── Days with very wide/narrow intervals
       └── Correlation with volatility

7. Residual Diagnostics
   ├── Residual statistics:
   │   ├── Mean: X.XX (target: ≈ 0)
   │   ├── Std dev: X.XX
   │   ├── Skewness: X.XX
   │   └── Kurtosis: X.XX
   │
   ├── Tests:
   │   ├── Ljung-Box (autocorrelation): p = X.XXX
   │   ├── Durbin-Watson: X.XX
   │   ├── Shapiro-Wilk (normality): p = X.XXX
   │   └── Breusch-Pagan (homoscedasticity): p = X.XXX
   │
   └── Interpretation: Residuals are white noise / have patterns

8. Model Comparison
   ├── Performance vs baselines:
   │   ├── PatchTST: MAE = X.XX
   │   ├── Naive: MAE = X.XX
   │   ├── ARIMA: MAE = X.XX
   │   ├── Linear Reg: MAE = X.XX
   │   └── Improvement over best baseline: X%
   │
   └── Superiority metrics:
       ├── Win rate: PatchTST wins X% of forecast periods
       └── Statistical significance: t-test p-value = X.XXX

9. Accuracy by Horizon (30-step forecast)
   ├── Step 1: MAPE = X.XX%, RMSE = X.XX
   ├── Step 5: MAPE = X.XX%, RMSE = X.XX
   ├── Step 10: MAPE = X.XX%, RMSE = X.XX
   ├── Step 15: MAPE = X.XX%, RMSE = X.XX
   ├── Step 20: MAPE = X.XX%, RMSE = X.XX
   ├── Step 25: MAPE = X.XX%, RMSE = X.XX
   ├── Step 30: MAPE = X.XX%, RMSE = X.XX
   │
   ├── Degradation analysis:
   │   ├── % accuracy loss from step 1 to step 30
   │   ├── Exponential fit: y = a * exp(-b*t)
   │   └── "Useful forecast horizon": X steps
   │
   └── Visualization: Metrics vs forecast horizon

10. Risk Analysis
    ├── Predicted vs actual max drawdown
    ├── Downside deviation
    ├── Sortino ratio (if returns forecasted)
    ├── Days with forecast errors > 2σ
    ├── Largest prediction errors (top 10)
    └── Error patterns and causes

11. Temporal Analysis
    ├── Performance by time of day (for hourly data)
    ├── Performance by day of week
    ├── Performance during market events (gaps, crashes)
    ├── Seasonal patterns
    └── Trend of accuracy over test period

12. Recommendations
    ├── Trading strategies:
    │   ├── Use point forecasts for: Medium-term positions (5-15 days)
    │   ├── Use confidence bands for: Risk management (80% CI)
    │   ├── Use directional forecast for: Trade direction signals
    │   └── Avoid predictions beyond: Step 20 (if accuracy degraded)
    │
    ├── Model improvements:
    │   ├── Feature engineering: Add/remove features
    │   ├── Architecture: Increase model capacity
    │   ├── Training: Different loss function or data augmentation
    │   ├── Ensemble: Combine with other models
    │   └── Retraining: More frequent updates
    │
    ├── Monitoring:
    │   ├── Re-evaluate monthly
    │   ├── Track live forecast accuracy
    │   ├── Alert if MAPE > threshold
    │   └── Retrain if accuracy drops > X%
    │
    └── Next steps:
        ├── Implement trading strategy based on signals
        ├── Backtest trading profitability
        ├── Paper trade before live deployment
        └── Monitor and adjust thresholds
```

### 8.2 Visualization Suite
```
Plots to Generate:

1. Training History
   ├── Train loss vs Epoch
   ├── Validation loss vs Epoch
   ├── Learning rate schedule
   └── Gradient norm vs Epoch

2. Prediction vs Actual (Sample from test set)
   ├── Time series plot: Actual vs Predicted
   ├── Scatter plot: Actual vs Predicted
   │   ├── Diagonal line: Perfect predictions
   │   ├── Color: Forecast step (1-30)
   │   └── Size: Forecast error magnitude
   │
   └── Residual plot: Time series of errors

3. Forecast Horizon Accuracy
   ├── Line plot: MAPE vs Forecast Step (1-30)
   ├── Bar plot: RMSE by forecast step
   ├── Degradation curve: Fitted exponential
   └── Confidence bands: ±1σ around accuracy

4. Prediction Intervals
   ├── Time series: Actual price + 80% + 95% bands
   ├── Scatter: Coverage rate @ 80% and 95%
   ├── Heatmap: Interval width by time and horizon
   └── Calibration curve: Nominal vs Empirical coverage

5. Residual Diagnostics
   ├── Histogram + KDE: Distribution of residuals
   ├── Q-Q plot: Normality check
   ├── ACF/PACF: Autocorrelation check
   ├── Time series: Residuals over time
   └── Volatility: Error magnitude over time

6. Feature Analysis
   ├── Correlation heatmap: All features
   ├── Bar plot: Top 15 features by correlation
   ├── Scatter matrix: Top 6 features vs target
   └── VIF ranking: Multicollinearity

7. Walk-Forward Results
   ├── Metrics over iterations: MAE, RMSE, MAPE trend
   ├── Coverage rates: 80% and 95% CI over iterations
   ├── Forecast accuracy by date (heatmap)
   └── Cumulative prediction error

8. Comparative Analysis
   ├── Bar chart: MAE/RMSE/MAPE for each model
   ├── Box plot: Distribution of errors by model
   ├── Time series: Prediction errors (PatchTST vs Baselines)
   └── Scatter: Accuracy by forecast step (all models)

9. Market Condition Analysis
   ├── Scatter: Accuracy vs Volatility (rolling)
   ├── Scatter: Accuracy vs Trend strength
   ├── Bar: MAE by volatility regime (high/medium/low)
   ├── Bar: Directional accuracy by trend (up/down/sideways)
   └── Heatmap: Accuracy by time and volatility

10. Risk Visualization
    ├── Distribution: Predicted returns vs Actual
    ├── Max Drawdown: Cumulative predicted vs actual
    ├── Scatter: Forecast error magnitude vs realized volatility
    └── Histogram: Large errors (|error| > 2σ)

11. Summary Dashboard
    ├── KPI cards: MAE, RMSE, MAPE, DA, R²
    ├── Mini charts: Key metrics across all analyses
    ├── Status indicators: Good/Warning/Poor for each metric
    └── Recommendation box: Suitable for trading? Yes/No
```

### 8.3 Export Formats
```
Results export options:

1. PDF Report
   ├── All text sections
   ├── All visualizations
   ├── Formatted professionally
   ├── Table of contents
   └── Page numbers

2. Excel Workbook
   ├── Sheet 1: Summary metrics (all numbers)
   ├── Sheet 2: Walk-forward results (iteration by iteration)
   ├── Sheet 3: Daily predictions vs actual
   ├── Sheet 4: Feature importance
   ├── Sheet 5: Residuals analysis
   └── Sheet 6: Predictions with confidence intervals

3. CSV Files
   ├── predictions.csv: Date, Actual, Predicted, Lower_95, Upper_95, Lower_80, Upper_80
   ├── metrics.csv: Metric, Value, Interpretation
   ├── features.csv: Feature, Correlation, VIF, Selected
   ├── residuals.csv: Date, Residual, Abs_Error, Forecast_Step
   └── walk_forward_results.csv: Iteration, MAE, RMSE, MAPE, DA, Coverage_95, Coverage_80

4. JSON Output
   ├── Model hyperparameters
   ├── Training configuration
   ├── All metrics (nested structure)
   ├── Feature list with properties
   └── Walk-forward iteration details

5. Interactive HTML Dashboard
   ├── Plotly plots (interactive)
   ├── Metric cards (KPIs)
   ├── Tabs: Overview, Detailed Analysis, Residuals, Comparisons
   ├── Filters: Date range, forecast step, market condition
   └── Export buttons: PNG, CSV from each visualization
```

---

## Phase 9: Implementation Checklist

### 9.1 Code Organization
```
src/* 
├── 1_data/
│   ├── Download OHLCV
│   ├── Calculate indicators
│   ├── Handle missing values
│   └── Create sequences
│
├── 2_features/
│   ├── Correlation analysis
│   ├── VIF calculation
│   ├── Feature selection
│   └── Visualization
│
├── 3_model/
│   ├── PatchTST class
│   ├── Attention blocks
│   ├── Encoder/Decoder
│   └── Training functions
│
├── 4_model/
│   ├── Data loading
│   ├── Training loop
│   ├── Validation
│   ├── Model checkpointing
│   └── Learning curves
│
├── 5_backtest/
│   ├── Walk-forward logic
│   ├── Prediction interval generation
│   ├── Metrics calculation
│   ├── Result storage
│   └── Iteration management
│
├── 6_metrics/
│   ├── All metric calculations
│   ├── Statistical tests
│   ├── Residual analysis
│   ├── Calibration checks
│   └── Comparative analysis
│
├── 7_display/
│   ├── All plot functions
│   ├── Dashboard creation
│   ├── Batch visualization
│   └── Export to image/PDF
│
├── 8_display/
│   ├── PDF report creation
│   ├── Excel export
│   ├── CSV outputs
│   ├── HTML dashboard
│   └── Summary document
│
├── config.py
│   ├── Hyperparameters
│   ├── Data parameters
│   ├── Path configurations
│   └── Backtesting parameters
│
├── utils/
│   ├── Helper functions
│   ├── Scaler management
│   ├── Device setup (GPU/CPU)
│   └── Logging setup
│
└── run\run_pipeline_forecast_patchTST.py
    └── Orchestrate all phases
```

### 9.2 Dependencies
```
Core Libraries:
├── pytorch / pytorch-lightning (deep learning)
├── pandas / numpy (data manipulation)
├── scikit-learn (preprocessing, metrics, statistical tests)
├── yfinance (data download)
├── ta / pandas-ta / talib (technical indicators)
├── matplotlib / seaborn (visualization)
├── plotly (interactive plots)
├── scipy (statistical functions)
├── statsmodels (ARIMA, statistical tests)
├── reportlab / fpdf (PDF generation)
├── openpyxl (Excel generation)
└── python-dotenv (configuration)

Optional:
├── wandb (experiment tracking)
├── optuna (hyperparameter optimization)
├── shap (model interpretability)
└── backtest/backtesting.py (alternative backtesting framework)
```

### 9.3 Computational Requirements
```
Hardware:
├── GPU: NVIDIA GPU with CUDA 12.6 (24GB VRAM)
│   ├── Training: 2-4 hours (full dataset)
│   ├── Backtesting: 30-60 minutes (10 walk-forward iterations)
│   └── Inference: < 1 second per forecast
│
└── CPU: Intel Core i9 with 32 GB RAM (fallback)]:
    ├── Training: 8-16 hours
    ├── Backtesting: 2-4 hours
    └── RAM: 16GB+ recommended
```

---

## Phase 10: Advanced Extensions (Optional)

### 10.1 Ensemble Methods
```
Combine multiple models:
├── Model 1: PatchTST (price forecast)
├── Model 2: PatchTST (returns forecast)
├── Model 3: LSTM baseline
├── Model 4: GRU alternative
│
├── Ensemble strategies:
│   ├── Simple average
│   ├── Weighted average (by validation accuracy)
│   ├── Stacking (meta-learner)
│   └── Voting (majority for direction)
│
└── Benefits:
    ├── Reduced overfitting
    ├── More robust predictions
    └── Better confidence intervals
```

### 10.2 Multi-Asset Modeling
```
Forecast multiple cryptocurrencies jointly:
├── Transfer learning: Pre-train on BTC, fine-tune on ALT
├── Multi-task learning: Share encoder, task-specific decoders
├── Cross-asset attention: Correlations between assets
├── Portfolio optimization: Combine forecasts for allocation
└── Correlation forecasting: Predict asset correlations
```

### 10.3 Market Regime Detection
```
Identify and adapt to market conditions:
├── HMM (Hidden Markov Model): 3 regimes (bull, bear, sideways)
├── GMM (Gaussian Mixture Model): Clustering
├── Supervised classification: Classify regime from features
│
├── Adaptive model:
│   ├── Different hyperparameters per regime
│   ├── Separate models per regime
│   └── Switching strategy
│
└── Results:
    ├── Separate metrics per regime
    ├── Regime prediction accuracy
    └── Strategy allocation to regimes
```

### 10.4 Online Learning
```
Continuous model updates:
├── Incremental learning: Update model on new data
├── Concept drift detection: Identify when model performance declines
├── Automated retraining: Trigger when drift detected
├── Model versioning: Track model evolution
└── A/B testing: Compare model versions live
```

### 10.5 Explainability (SHAP/LIME)
```
Understand model predictions:
├── SHAP values: Feature importance per prediction
├── Partial dependence plots: Feature effects
├── LIME: Local linear approximations
├── Attention weights: Which input timesteps matter?
└── Ablation studies: Feature removal impact
```

---

## Phase 11: Deployment Considerations

### 11.1 Production System
```
├── Model serving: FastAPI / Flask endpoint
├── Real-time inference: Streaming data pipeline
├── Database: Store predictions and actuals
├── Monitoring: Track live accuracy
├── Alerting: Notify if accuracy degrades
├── Versioning: Model registry (MLflow)
└── Rollback: Previous model versions
```

### 11.2 Risk Management
```
├── Stop-loss thresholds: Max allowed error
├── Position sizing: Based on forecast confidence
├── Portfolio constraints: Max allocation per asset
├── Leverage limits: Risk-adjusted sizing
├── Correlation hedging: Offset correlated positions
└── Daily reconciliation: Forecast vs actual tracking
```

---

## Success Criteria and Thresholds

### Phase 2 (Features)
✓ Successfully create 40-50 technical indicators
✓ Identify top 10 features with |correlation| > 0.15
✓ Remove multicollinear features (VIF > 5)
✓ Final feature count: 30-40 features

### Phase 4 (Data Prep)
✓ Generate 20,000+ training sequences
✓ No data leakage between train/val/test
✓ Normalization applied correctly
✓ DataLoaders created without errors

### Phase 5 (Training)
✓ Training loss decreases consistently
✓ Validation loss < baseline loss after 50 epochs
✓ No NaN losses or exploding gradients
✓ Final model saved with best weights

### Phase 6 (Backtesting)
✓ Complete 10 walk-forward iterations
✓ Generate 30-step ahead forecasts
✓ Store all predictions and actuals
✓ No forward-looking bias in results

### Phase 7 (Metrics)
✓ MAPE < 15% on test set (good)
✓ Directional accuracy > 53% (better than 50%)
✓ MASE < 1 (better than naive baseline)
✓ R² > 0.5 (explains 50%+ of variance)
✓ 95% CI coverage between 90-99%
✓ 80% CI coverage between 75-85%
✓ Residuals pass white noise test (Ljung-Box p > 0.05)

### Phase 8 (Reporting)
✓ Comprehensive report generated (10+ sections)
✓ 10+ visualizations created
✓ Multiple export formats (PDF, Excel, CSV, HTML)
✓ Actionable recommendations provided

---

## Timeline Estimate (for experienced ML engineer)

```
Phase 1 - Data Prep:          2-3 days
Phase 2 - Feature Analysis:   2 days
Phase 3 - Model Design:       1 day
Phase 4 - Data Preparation:   2 days
Phase 5 - Model Training:     2 days (+ computation time)
Phase 6 - Backtesting:        2 days (+ computation time)
Phase 7 - Metrics:            2 days
Phase 8 - Reporting:          2 days

Total: 15-17 days (+ computation time)
```

---

## References and Resources

### Key Papers
- Dosovitskiy et al. (2021): "An Image is Worth 16x16 Words" (Vision Transformer)
- Nie et al. (2023): "A Time Series is Worth 64 Words" (PatchTST)
- Transformer architecture: "Attention is All You Need" (Vaswani et al., 2017)

### Libraries & Frameworks
- PyTorch: https://pytorch.org/
- PyTorch Lightning: https://www.pytorchlightning.ai/
- TA-Lib: https://mrjbq7.github.io/ta-lib/
- Scikit-learn: https://scikit-learn.org/

### Performance Benchmarks
- Expected accuracy: MAPE 8-12% for 1-step forecast
- Expected accuracy: MAPE 15-20% for 30-step forecast
- Expected directional accuracy: 54-58%
- Expected Sharpe ratio (trading): 0.5-1.5

---

## End of Implementation Outline

This comprehensive outline covers every aspect of building a production-ready PatchTST crypto forecaster.
Each phase includes specific code structures, metrics, and success criteria.

Start with Phase 1 and proceed sequentially. Use this as a reference during implementation.

Good luck with your crypto forecasting system!
```

Code stubs:
"""
PatchTST Crypto Forecaster - Code Architecture & Skeleton
Provides structure and function signatures for implementation
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# PHASE 1: DATA PREPARATION
# ============================================================================

class DataPreprocessor:
    """Handle OHLCV data download, cleaning, and normalization"""
    
    def __init__(self, ticker: str, start_date: str, end_date: str, interval: str = '1d'):
        """
        Initialize data preprocessor
        
        Args:
            ticker: Cryptocurrency ticker (e.g., 'BTC-USD')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            interval: '1d', '4h', '1h' etc.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.raw_data = None
        self.clean_data = None
        self.feature_data = None
        
    def download_ohlcv(self) -> pd.DataFrame:
        """
        Download OHLCV data from yfinance
        
        Returns:
            DataFrame with OHLCV columns
        """
        # TODO: Implement
        # - Use yfinance.download()
        # - Handle intervals properly
        # - Validate data integrity
        # - Return (n_samples, 5) DataFrame
        pass
    
    def handle_missing_values(self, method: str = 'ffill') -> pd.DataFrame:
        """
        Handle missing values in OHLCV data
        
        Args:
            method: 'ffill' (forward fill), 'bfill', 'interpolate'
        
        Returns:
            DataFrame with no missing values
        """
        # TODO: Implement
        # - Forward/backward fill
        # - Linear interpolation
        # - Validate no NaN remaining
        pass
    
    def detect_outliers(self, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using IQR and z-score methods
        
        Args:
            threshold: z-score threshold (3.0 = 3 sigma)
        
        Returns:
            Boolean array of outlier positions
        """
        # TODO: Implement
        # - Calculate z-scores for OHLCV
        # - Identify extreme moves
        # - Flag suspicious patterns
        # - Return outlier mask
        pass
    
    def normalize_data(self, method: str = 'standard') -> pd.DataFrame:
        """
        Normalize data for model input
        
        Args:
            method: 'standard' (z-score), 'minmax', 'log'
        
        Returns:
            Normalized DataFrame
        """
        # TODO: Implement
        # - StandardScaler: (x - mean) / std
        # - MinMaxScaler: (x - min) / (max - min)
        # - Log returns: log(close_t / close_t-1)
        # - Fit on training data only
        pass
    
    def preprocess(self) -> pd.DataFrame:
        """
        Execute full preprocessing pipeline
        
        Returns:
            Clean, normalized OHLCV DataFrame
        """
        print("Starting data preprocessing...")
        self.raw_data = self.download_ohlcv()
        self.clean_data = self.handle_missing_values()
        # Handle outliers if needed
        # Normalize data
        return self.clean_data


# ============================================================================
# PHASE 2: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Create technical indicators and features"""
    
    def __init__(self, ohlcv_df: pd.DataFrame):
        """
        Initialize feature engineer
        
        Args:
            ohlcv_df: DataFrame with OHLCV columns
        """
        self.ohlcv = ohlcv_df.copy()
        self.features = pd.DataFrame(index=ohlcv_df.index)
        self.feature_names = []
    
    def add_momentum_indicators(self) -> None:
        """
        Add momentum indicators:
        - RSI (14, 21, 28)
        - Stochastic RSI
        - MACD
        - ROC
        - Momentum
        """
        # TODO: Implement
        # self.features['RSI_14'] = ta.momentum.rsi(self.ohlcv['Close'], 14)
        # self.features['MACD'] = ta.trend.macd_diff(self.ohlcv['Close'])
        # ... add others
        pass
    
    def add_trend_indicators(self) -> None:
        """
        Add trend indicators:
        - SMA (20, 50, 100, 200)
        - EMA (12, 26)
        - ADX
        - Linear regression slope
        """
        # TODO: Implement
        # self.features['SMA_20'] = self.ohlcv['Close'].rolling(20).mean()
        # self.features['EMA_12'] = self.ohlcv['Close'].ewm(12).mean()
        # ... add others
        pass
    
    def add_volatility_indicators(self) -> None:
        """
        Add volatility indicators:
        - Bollinger Bands
        - ATR
        - Standard Deviation
        """
        # TODO: Implement
        # self.features['BB_Upper'] = ...
        # self.features['ATR_14'] = ...
        # ... add others
        pass
    
    def add_volume_indicators(self) -> None:
        """
        Add volume-based indicators:
        - OBV
        - CMF
        - Volume SMA ratio
        """
        # TODO: Implement
        # self.features['OBV'] = ...
        # self.features['CMF'] = ...
        # ... add others
        pass
    
    def add_price_action_features(self) -> None:
        """
        Add price action features:
        - Returns (1, 5, 10, 20 periods)
        - High-Low ratio
        - Close-Open ratio
        """
        # TODO: Implement
        # self.features['Return_1'] = self.ohlcv['Close'].pct_change(1)
        # self.features['Return_5'] = self.ohlcv['Close'].pct_change(5)
        # ... add others
        pass
    
    def add_statistical_features(self) -> None:
        """
        Add statistical features:
        - Rolling skewness
        - Rolling kurtosis
        - Autocorrelation
        """
        # TODO: Implement
        # self.features['Skew_20'] = self.ohlcv['Close'].rolling(20).skew()
        # self.features['Kurt_20'] = self.ohlcv['Close'].rolling(20).kurt()
        # ... add others
        pass
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Execute full feature engineering pipeline
        
        Returns:
            DataFrame with all OHLCV + engineered features
        """
        print("Engineering features...")
        self.add_momentum_indicators()
        self.add_trend_indicators()
        self.add_volatility_indicators()
        self.add_volume_indicators()
        self.add_price_action_features()
        self.add_statistical_features()
        
        # Combine with OHLCV
        result = pd.concat([self.ohlcv, self.features], axis=1)
        
        # Remove NaN from indicator lookback periods
        result = result.dropna()
        
        print(f"Created {result.shape[1] - 5} features from {self.ohlcv.shape[0]} samples")
        return result


# ============================================================================
# PHASE 3: FEATURE SELECTION & CORRELATION ANALYSIS
# ============================================================================

class FeatureSelector:
    """Analyze correlations and select best features"""
    
    def __init__(self, data: pd.DataFrame, target_col: str = 'Close'):
        """
        Initialize feature selector
        
        Args:
            data: DataFrame with features and target
            target_col: Column name of target variable
        """
        self.data = data
        self.target_col = target_col
        self.correlation_matrix = None
        self.selected_features = None
        self.vif_scores = None
    
    def compute_correlations(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlation matrix
        
        Args:
            method: 'pearson', 'spearman', 'kendall'
        
        Returns:
            Correlation matrix DataFrame
        """
        # TODO: Implement
        # self.correlation_matrix = self.data.corr(method=method)
        # Extract correlations with target
        # Return sorted by abs correlation
        pass
    
    def calculate_vif(self) -> pd.Series:
        """
        Calculate Variance Inflation Factor (VIF)
        
        Returns:
            Series of VIF scores per feature
        """
        # TODO: Implement
        # from statsmodels.stats.outliers_influence import variance_inflation_factor
        # Calculate VIF for all features
        # self.vif_scores = ...
        # Return sorted
        pass
    
    def select_features(self, 
                       corr_threshold: float = 0.05,
                       vif_threshold: float = 5.0,
                       n_features: int = 40) -> List[str]:
        """
        Select features based on correlation and VIF
        
        Args:
            corr_threshold: Min abs correlation with target
            vif_threshold: Max VIF allowed
            n_features: Target number of features
        
        Returns:
            List of selected feature names
        """
        # TODO: Implement
        # 1. Remove low correlation features (|r| < threshold)
        # 2. Remove high VIF features (VIF > threshold)
        # 3. Select top n features by correlation
        # 4. Ensure no multicollinearity
        # self.selected_features = ...
        # return self.selected_features
        pass
    
    def visualize_correlations(self, figsize: Tuple[int, int] = (12, 10)):
        """
        Create correlation heatmap and scatter plots
        
        Args:
            figsize: Figure size
        """
        # TODO: Implement
        # - Heatmap of correlation matrix
        # - Top 10 features by correlation with target
        # - Scatter matrix of top 6 features
        # - VIF bar chart
        # - Save plots to file
        pass
    
    def generate_report(self) -> Dict:
        """
        Generate feature selection report
        
        Returns:
            Dictionary with selection statistics
        """
        # TODO: Implement
        # Return dict with:
        # - Total features before/after
        # - Removed features
        # - Top features
        # - Correlation statistics
        # - Multicollinearity summary
        pass


# ============================================================================
# PHASE 4: DATASET AND DATALOADER
# ============================================================================

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series sequences"""
    
    def __init__(self, 
                 data: np.ndarray,
                 seq_len: int = 504,
                 pred_len: int = 30,
                 stride: int = 1):
        """
        Initialize dataset
        
        Args:
            data: (n_samples, n_features) array
            seq_len: Input sequence length
            pred_len: Prediction horizon
            stride: Step between sequences
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        
        # Number of possible sequences
        self.num_samples = (len(data) - seq_len - pred_len) // stride + 1
    
    def __len__(self) -> int:
        """Return dataset size"""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample
        
        Args:
            idx: Sample index
        
        Returns:
            (input_sequence, target_sequence) tensors
        """
        # TODO: Implement
        # start_idx = idx * self.stride
        # X = data[start_idx:start_idx+seq_len]
        # y = data[start_idx+seq_len:start_idx+seq_len+pred_len, target_idx]
        # Return (X, y) as tensors
        pass


def create_dataloaders(data: pd.DataFrame,
                      feature_cols: List[str],
                      target_col: str = 'Close',
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders
    
    Args:
        data: Full dataset
        feature_cols: List of feature columns to use
        target_col: Target column name
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        batch_size: Batch size
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # TODO: Implement
    # 1. Extract features and target
    # 2. Normalize using training data
    # 3. Time-based split (no shuffling)
    # 4. Create TimeSeriesDataset for each split
    # 5. Create DataLoaders with appropriate settings
    # Return loaders
    pass


# ============================================================================
# PHASE 5: MODEL ARCHITECTURE - PatchTST
# ============================================================================

@dataclass
class PatchTSTConfig:
    """Configuration for PatchTST model"""
    seq_len: int = 504  # Input sequence length
    pred_len: int = 30  # Forecast horizon
    d_model: int = 256  # Embedding dimension
    n_heads: int = 8  # Number of attention heads
    n_encoder_layers: int = 6  # Number of encoder layers
    d_ff: int = 512  # Feed-forward hidden dimension
    dropout: float = 0.1  # Dropout rate
    patch_len: int = 16  # Patch size
    stride: int = 8  # Stride between patches
    n_features: int = 35  # Number of input features
    activation: str = 'gelu'  # Activation function


class PatchEmbedding(nn.Module):
    """Patch embedding layer for PatchTST"""
    
    def __init__(self, 
                 patch_len: int,
                 stride: int,
                 d_model: int,
                 n_features: int):
        """
        Initialize patch embedding
        
        Args:
            patch_len: Length of each patch
            stride: Stride between patches
            d_model: Embedding dimension
            n_features: Number of input features
        """
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        
        # Linear projection from patch to embedding
        self.linear = nn.Linear(patch_len * n_features, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, n_features)
        
        Returns:
            (batch, n_patches, d_model)
        """
        # TODO: Implement
        # 1. Create patches using stride
        # 2. Flatten each patch
        # 3. Project to d_model dimension
        # Return patched embedding
        pass


class TransformerEncoder(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, config: PatchTSTConfig):
        """
        Initialize encoder
        
        Args:
            config: PatchTSTConfig object
        """
        super().__init__()
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Activation
        self.activation = nn.GELU() if config.activation == 'gelu' else nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            (batch, seq_len, d_model)
        """
        # TODO: Implement
        # 1. Multi-head attention with residual
        # 2. Layer norm
        # 3. Feed-forward with residual
        # 4. Layer norm
        # Return output
        pass


class PatchTST(nn.Module):
    """Patch Time Series Transformer"""
    
    def __init__(self, config: PatchTSTConfig):
        """
        Initialize PatchTST model
        
        Args:
            config: PatchTSTConfig object
        """
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.embedding = PatchEmbedding(
            config.patch_len,
            config.stride,
            config.d_model,
            config.n_features
        )
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, (config.seq_len - config.patch_len) // config.stride + 1, config.d_model)
        )
        
        # Transformer encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(config) for _ in range(config.n_encoder_layers)
        ])
        
        # Forecast head
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.pred_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, n_features)
        
        Returns:
            (batch, pred_len)
        """
        # TODO: Implement
        # 1. Patch embedding
        # 2. Add positional encoding
        # 3. Pass through encoder layers
        # 4. Flatten output
        # 5. Pass through forecast head
        # Return predictions
        pass


# ============================================================================
# PHASE 6: TRAINING
# ============================================================================

class Trainer:
    """Handle model training and validation"""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: PatchTSTConfig,
                 device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initialize trainer
        
        Args:
            model: PatchTST model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            config: Model configuration
            device: Device to train on (cuda/cpu)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> float:
        """
        Train for one epoch
        
        Returns:
            Average training loss
        """
        # TODO: Implement
        # 1. Set model to train mode
        # 2. Iterate through batches
        # 3. Forward pass
        # 4. Backward pass
        # 5. Optimizer step
        # 6. Gradient clipping
        # Return average loss
        pass
    
    def validate(self) -> float:
        """
        Validate on validation set
        
        Returns:
            Average validation loss
        """
        # TODO: Implement
        # 1. Set model to eval mode
        # 2. No gradient computation
        # 3. Iterate through batches
        # 4. Forward pass
        # 5. Compute loss
        # Return average loss
        pass
    
    def train(self, epochs: int = 100, early_stopping_patience: int = 20):
        """
        Train model for multiple epochs
        
        Args:
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        """
        # TODO: Implement
        # 1. Loop for epochs
        # 2. Train epoch
        # 3. Validate
        # 4. Check early stopping
        # 5. Save best model
        # 6. Log metrics
        pass
    
    def save_model(self, path: str):
        """Save model weights"""
        # TODO: Implement
        # torch.save(self.model.state_dict(), path)
        pass
    
    def load_model(self, path: str):
        """Load model weights"""
        # TODO: Implement
        # self.model.load_state_dict(torch.load(path))
        pass


# ============================================================================
# PHASE 7: BACKTESTING - WALK FORWARD
# ============================================================================

class WalkForwardBacktester:
    """Implement walk-forward backtesting"""
    
    def __init__(self,
                 data: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str = 'Close',
                 train_window: int = 60,  # 60 weeks
                 test_window: int = 1,    # 1 week
                 forecast_horizon: int = 30):
        """
        Initialize backtester
        
        Args:
            data: Full dataset
            feature_cols: Feature column names
            target_col: Target column name
            train_window: Training period (in weeks)
            test_window: Test period (in weeks)
            forecast_horizon: Forecast horizon
        """
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.train_window = train_window
        self.test_window = test_window
        self.forecast_horizon = forecast_horizon
        
        self.results = []
        self.predictions = []
    
    def run_backtest(self, model_class, config: PatchTSTConfig):
        """
        Run walk-forward backtest
        
        Args:
            model_class: PatchTST class
            config: Model configuration
        """
        # TODO: Implement
        # 1. Calculate date splits
        # 2. For each iteration:
        #    a. Prepare training data
        #    b. Prepare test data
        #    c. Train model (or retrain)
        #    d. Generate 30-step forecast
        #    e. Store predictions and actuals
        #    f. Compute metrics
        # 3. Aggregate results across iterations
        pass
    
    def generate_prediction_intervals(self, method: str = 'quantile'):
        """
        Generate confidence bands
        
        Args:
            method: 'quantile', 'bootstrap', 'parametric', 'empirical'
        """
        # TODO: Implement
        # - Generate lower/upper bounds for 80% and 95% CI
        # - Store with predictions
        pass
    
    def save_results(self, filepath: str):
        """Save backtest results to CSV"""
        # TODO: Implement
        # Save:
        # - Predictions and actuals
        # - Confidence intervals
        # - Metrics per iteration
        pass


# ============================================================================
# PHASE 8: METRICS AND EVALUATION
# ============================================================================

class MetricsCalculator:
    """Calculate comprehensive metrics"""
    
    @staticmethod
    def mae(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        # TODO: Implement
        pass
    
    @staticmethod
    def rmse(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        # TODO: Implement
        pass
    
    @staticmethod
    def mape(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        # TODO: Implement
        pass
    
    @staticmethod
    def smape(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        # TODO: Implement
        pass
    
    @staticmethod
    def mase(y_actual: np.ndarray, y_pred: np.ndarray, y_naive: np.ndarray) -> float:
        """Mean Absolute Scaled Error"""
        # TODO: Implement
        pass
    
    @staticmethod
    def r2_score(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared score"""
        # TODO: Implement
        pass
    
    @staticmethod
    def directional_accuracy(y_actual: np.ndarray, y_pred: np.ndarray) -> float:
        """Proportion of correct directional predictions"""
        # TODO: Implement
        pass
    
    @staticmethod
    def interval_coverage_rate(y_actual: np.ndarray, 
                              lower: np.ndarray, 
                              upper: np.ndarray) -> float:
        """Proportion of actual values within interval"""
        # TODO: Implement
        pass
    
    @staticmethod
    def mean_interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
        """Average width of prediction intervals"""
        # TODO: Implement
        pass
    
    @staticmethod
    def pinaw(lower: np.ndarray, upper: np.ndarray, 
              y_actual: np.ndarray) -> float:
        """Prediction Interval Normalized Average Width"""
        # TODO: Implement
        pass
    
    @staticmethod
    def calculate_all_metrics(y_actual: np.ndarray,
                             y_pred: np.ndarray,
                             lower_95: np.ndarray,
                             upper_95: np.ndarray,
                             lower_80: np.ndarray,
                             upper_80: np.ndarray) -> Dict[str, float]:
        """
        Calculate all metrics at once
        
        Returns:
            Dictionary with all metric values
        """
        # TODO: Implement
        # Compute all metrics
        # Return as dict
        pass


# ============================================================================
# PHASE 9: VISUALIZATION
# ============================================================================

class Visualizer:
    """Create visualizations for results"""
    
    @staticmethod
    def plot_prediction_vs_actual(dates, y_actual, y_pred, lower_95, upper_95, 
                                  lower_80, upper_80, figsize=(15, 6)):
        """Plot predictions with confidence bands"""
        # TODO: Implement
        pass
    
    @staticmethod
    def plot_forecast_horizon_accuracy(horizons, mape_scores, figsize=(10, 6)):
        """Plot accuracy degradation over forecast steps"""
        # TODO: Implement
        pass
    
    @staticmethod
    def plot_residual_diagnostics(residuals, figsize=(15, 10)):
        """Plot residual analysis panels"""
        # TODO: Implement
        # - Histogram + KDE
        # - Q-Q plot
        # - ACF plot
        # - Time series
        pass
    
    @staticmethod
    def plot_feature_correlation(correlation_matrix, top_n=15):
        """Plot top features by correlation"""
        # TODO: Implement
        pass
    
    @staticmethod
    def plot_walk_forward_results(iterations, metrics_dict):
        """Plot metrics across walk-forward iterations"""
        # TODO: Implement
        pass
    
    @staticmethod
    def create_dashboard(results: Dict):
        """Create comprehensive results dashboard"""
        # TODO: Implement
        # Create figure with multiple subplots
        # Show all key metrics and plots
        pass


# ============================================================================
# PHASE 10: REPORTING
# ============================================================================

class ReportGenerator:
    """Generate comprehensive reports"""
    
    @staticmethod
    def generate_pdf_report(results: Dict, output_path: str):
        """Generate PDF report with all results"""
        # TODO: Implement
        # - Use reportlab or similar
        # - Include all sections from outline
        # - Embed visualizations
        pass
    
    @staticmethod
    def generate_excel_report(results: Dict, output_path: str):
        """Generate Excel workbook with detailed results"""
        # TODO: Implement
        # - Multiple sheets
        # - Formatted tables
        # - Charts embedded
        pass
    
    @staticmethod
    def generate_csv_exports(results: Dict, output_dir: str):
        """Export results to CSV files"""
        # TODO: Implement
        # - predictions.csv
        # - metrics.csv
        # - features.csv
        # - residuals.csv
        pass
    
    @staticmethod
    def generate_html_dashboard(results: Dict, output_path: str):
        """Generate interactive HTML dashboard"""
        # TODO: Implement
        # - Use plotly for interactive plots
        # - Add filters and tabs
        # - Self-contained HTML file
        pass


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

class PatchTSTForecaster:
    """Main orchestrator for entire pipeline"""
    
    def __init__(self, config_dict: Dict):
        """
        Initialize forecaster
        
        Args:
            config_dict: Configuration dictionary
        """
        self.config = config_dict
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
    
    def run_full_pipeline(self):
        """Execute complete forecasting pipeline"""
        
        print("=" * 60)
        print("PATCHTST CRYPTO PRICE & RETURNS FORECASTER")
        print("=" * 60)
        
        # Phase 1: Data Preparation
        print("\n[PHASE 1] Data Preparation...")
        # TODO: Implement
        
        # Phase 2: Feature Engineering
        print("\n[PHASE 2] Feature Engineering...")
        # TODO: Implement
        
        # Phase 3: Feature Selection
        print("\n[PHASE 3] Feature Selection & Correlation Analysis...")
        # TODO: Implement
        
        # Phase 4: Data Preparation for Model
        print("\n[PHASE 4] Creating DataLoaders...")
        # TODO: Implement
        
        # Phase 5: Model Training
        print("\n[PHASE 5] Training Model...")
        # TODO: Implement
        
        # Phase 6: Walk-Forward Backtesting
        print("\n[PHASE 6] Walk-Forward Backtesting...")
        # TODO: Implement
        
        # Phase 7: Metrics Calculation
        print("\n[PHASE 7] Calculating Metrics...")
        # TODO: Implement
        
        # Phase 8: Visualization
        print("\n[PHASE 8] Creating Visualizations...")
        # TODO: Implement
        
        # Phase 9: Report Generation
        print("\n[PHASE 9] Generating Reports...")
        # TODO: Implement
        
        print("\n" + "=" * 60)
        print("FORECASTING COMPLETE")
        print("=" * 60)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    config = {
        'ticker': 'BTC-USD',
        'start_date': '2021-01-01',
        'end_date': '2024-01-01',
        'interval': '1d',
        
        # Model config
        'seq_len': 504,
        'pred_len': 30,
        'd_model': 256,
        'n_heads': 8,
        'n_encoder_layers': 6,
        'd_ff': 512,
        'dropout': 0.1,
        'patch_len': 16,
        'stride': 8,
        
        # Training config
        'batch_size': 64,
        'epochs': 100,
        'learning_rate': 0.001,
        
        # Backtesting config
        'train_window': 60,  # weeks
        'test_window': 1,    # week
    }
    
    # Run full pipeline
    forecaster = PatchTSTForecaster(config)
    # forecaster.run_full_pipeline()
    
    print("\nCode skeleton complete!")
    print("All TODO items marked for implementation")
    print("\nRefer to PATCHTST_IMPLEMENTATION_OUTLINE.md for detailed guidance")
