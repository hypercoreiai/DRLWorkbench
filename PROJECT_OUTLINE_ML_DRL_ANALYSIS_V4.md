# Modular ML-DRL Analysis & Display Project — Outline Version 4 (Crypto-First Research Workbench)

This outline refines **V3** based on specific user requirements: **Crypto-priority**, **Research/Backtesting focus**, **Hybrid+Ensemble architecture**, and **High-End Hardware** (RTX 4090). It is designed to be a "Production-Grade Research Lab" — stable enough for serious analysis, but flexible for rapid experimentation.

---

## 1. High-Level Architecture (Refined for Research & Speed)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED INTERFACE (CLI / Experiment Config)                 │
│  run_experiment(config) → data → ensemble_model → walk_forward → stress_test │
└──────────────────────────────────────────────────────────────────────────────┘
         │               │             │             │              │
         ▼               ▼             ▼             ▼              ▼
   ┌──────────┐    ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────────┐
   │ config/  │    │ data/    │ │ models/    │ │backtest/ │ │ analysis/    │
   │ strict   │    │ crypto   │ │ hybrid     │ │walk_fwd  │ │ metrics      │
   │ .yaml    │    │ loaders  │ │ ensemble   │ │synthetic │ │ display      │
   │          │    │ validate │ │ drl_gpu    │ │costs     │ │ reporting    │
   └──────────┘    └──────────┘ └────────────┘ └──────────┘ └──────────────┘
```

**Key Enhancements in V4:**
- **Crypto-Specifics**: 24/7 handling, exchange-specific liquidity issues, high-frequency data potential.
- **Hardware Acceleration**: GPU-accelerated training (Torch 2.6 + CUDA) for DRL/Ensembles.
- **Stress Testing**: Synthetic data generation for crash scenarios (Flash Crash resilience).
- **Focused Metrics**: Sharpe > 1.5, RoMaD > 1, MaxDD < 20%.

---

## 2. Data Layer — Crypto-Optimized Pipeline

### 2.1 Data Ingestion (Crypto Focus)
**`src/data/loader.py`**:
- **Sources**: CCXT (Exchange data) or local CSV archives.
- **Granularity**: 1h, 15m, 5m (leveraging 32GB RAM).
- **Constraint**: 24/7 markets (no "market closed" gaps, unlike stocks).

### 2.2 Validation & Cleaning (Strict)
**`src/data/validator.py`**:
- **Checks**: 
    - `check_missing_data`: Strict timestamp continuity (critical for crypto).
    - `check_outliers`: "Wick" detection (flash crash artifacts vs real moves).
    - `check_liquidity`: Amihud ratio or Volume/MarketCap filters to ensure tradability.

### 2.3 Feature Engineering
- **On-Chain Metrics**: (Optional/Future) Hashrate, active addresses.
- **Technical Indicators**: Standard Momentum/Trend (RSI, MACD, Bollinger).
- **Normalization**: `RevinTransform` or `RobustScaler` (handles crypto volatility better).

---

## 3. Model Layer — Hybrid + Ensemble Architecture (GPU Accelerated)

**Hardware Target**: Python 3.12, PyTorch 2.6, CUDA (RTX 4090).

### 3.1 Ensemble Core
**`src/models/ensemble.py`**:
- **Hybrid Strategy**: Combine **Supervised Forecasting** (Price Direction/Volatility) with **DRL Agents** (Execution/allocation).
- **Voting Mechanisms**: 
    - `SoftVoting`: Weighted average of probabilities.
    - `Meta-Learner`: A small neural net learning to weight sub-models.

### 3.2 Forecasting Registry (Supervised)
- **NeuralForecast**: LSTM, NHITS, TFT (GPU accelerated).
- **Custom PyTorch**: 
    - `LSTM-Attention`: Transformer-like attention on time steps.
    - `CNN-LSTM`: Feature extraction via 1D Conv layers.

### 3.3 DRL Registry (Agents)
**`src/models/drl.py`**:
- **Algorithms**: PPO, SAC, TD3 (Stable Baselines3 or CleanRL).
- **State Space**: Windowed technicals + Account State + *Risk Regime Indicator*.
- **Action Space**: Continuous weights (portfolio allocation) or Discrete (Hold/Buy/Sell).
- **Reward**: Calmar or Sortino ratio (punish downside volatility heavy).

---

## 4. Backtesting — Robustness & Regimes

### 4.1 Walk-Forward Validation (Rolling Window)
- **Train/Test**: e.g., Train 2 Years, Test 3 Months, Step 1 Month.
- **Re-training**: Full model retrain or fine-tuning at each step.

### 4.2 Synthetic Stress Testing (New Requirement)
**`src/backtest/synthetic.py`**:
- **Agent-Based Modeling**: Simulate "Panic Selling" behavior.
- **Bootstrapping**: Block-bootstrap historical crash periods (e.g., May 2021, Nov 2022) to test survival.
- **Goal**: Ensure logic handles rapid "Bull to Bear" switches without >20% drawdown.

### 4.3 Transaction Costs
- **Fee Model**: Maker/Taker fees (e.g., 0.1%).
- **Slippage**: Volatility-adjusted slippage model.

---

## 5. Analysis & Success Criteria

### 5.1 Key Performance Indicators (KPIs)
**`src/analysis/metrics.py`**:
- **Primary**:
    - **Sharpe Ratio**: Target > 1.5.
    - **Max Drawdown**: Hard Limit < 20%.
    - **RoMaD**: Return / MaxDrawdown > 1.0.
- **Secondary**:
    - Win Rate, Profit Factor, Sortino Ratio.
    - **Regime specific performance**: Performance in "Bull" vs "Bear" vs "Chop".

### 5.2 Regime Coverage Analysis
- **Classification**:
    - High Vol (Crash/Manic)
    - Low Vol (Range/Trending)
- **Check**: Does the agent go to cash/short in High Vol Bear?

---

## 6. Implementation Stages

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up Python 3.12 / Torch 2.6 env.
- [ ] Implement `DataPipeline` with Crypto validation (CCXT/CSV).
- [ ] Establish `WalkForward` backtest skeleton.

### Phase 2: Models & Ensemble (Weeks 3-4)
- [ ] Implement `LSTM` and `CNN-LSTM` baselines.
- [ ] Implement basic `PPO` Agent.
- [ ] Create `EnsembleAgent` logic.

### Phase 3: Robustness & Stress Testing (Weeks 5-6)
- [ ] Implement Regime Detection.
- [ ] Build `SyntheticGenerator` for crash scenarios.
- [ ] Run massive parameter sweeps on RTX 4090 to find stable params.

### Phase 4: Analysis & Reporting (Week 7+)
- [ ] Build Comparative Dashboard.
- [ ] Generate final "Paper Trading" report proving >1.5 Sharpe.

---

## 7. Configuration Example

```yaml
experiment:
  name: "Crypto_Hybrid_V1"
  gpu_id: 0 # RTX 4090

data:
  tickers: ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
  timeframe: "1h"
  source: "binance_historical"

constraints:
  max_drawdown: 0.20
  min_sharpe: 1.0

models:
  ensemble:
    - type: "LSTM_Attn"
      weight: 0.4
    - type: "PPO_Agent"
      weight: 0.6
```
