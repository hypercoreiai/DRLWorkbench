1. Project Scope & Success Criteria
1.1 Primary use case: research/backtesting only, paper trading to distill the wide array of options down to efficient/effective Hybrid+Ensemble architecture. 

1.2 1. Minimum Sharpe Ratio: Minimum (Lower Bound): 1.0 (annualized, net of transaction costs and slippage). Target (Healthy): 1.5 - 2.0+.

1.2.2. Maximum Drawdown (Max DD) Cap: Recommended Maximum: Below 20%. Conservative/Institutional: 5% - 10%. Speculative: Up to 30% - 40% (Only for experienced, high-risk tolerance managers, as recovery from >50% drawdown is extremely difficult). Key Metric (RoMaD): Return over Maximum Drawdown (RoMaD) should ideally be > 1.

1.2.3. Regime Coverage & Robustness: 
    Regime Coverage Requirements:
        High Volatility (Panic/Crash): Must have learned to cut exposure, go to cash, or go short.
        Low Volatility (Trending/Quiet): Must maximize returns without excessive churn.
        Market Regime Switching: Must handle rapid shifts (e.g., bull to bear) without a complete "blowup" (total loss).
    Evaluation Techniques:
        Out-of-Sample Testing: The agent must be trained on a diverse set of historical data and validated on at least 3-5 years of out-of-sample data, including a major market downturn (e.g., 2008, 2020, 2022).
        Synthetic Data/Agent-Based Modeling: Using simulated environments to test edge cases, such as "flash crash" scenarios, to ensure safe exploration constraints.
1.3 I am the only consumer.
1.4 No backward compatibility required. Python 3.12 and Torch 2.6 (cuda). Hardware CPU Core I9 w/32GB Ram and RTX 4090 w/24GB VRam.

2. Data & Universe 
2.1 Asset universe is crypto currencies (high priority) and possible stock trades.
| 2.1 | Primarily crypto (major exchanges like Binance, Coinbase Pro, Kraken, and possibly some decentralized venues like Uniswap for diversity). No equities or FX unless needed for hedging correlations—keeping it crypto-focused to avoid data sprawl and API complexity. | Crypto data is readily available via APIs (e.g., Binance API), and as a hobbyist, I can use free tiers for backtesting without hitting paywalls too soon. Equities would require paid sources like Alpha Vantage or yfinance, which might complicate things for a local setup. Liquidity is a bigger issue in crypto, so I need filters for that. |
| 2.2 | Ticker list: Top 50-100 cryptos by market cap (e.g., BTC, ETH, and alts like ADA, SOL, DOGE) plus some stablecoins for cash equivalents. History depth: 2-3 years for training, but configurable up to 5 years if needed for robustness. | This fits my typical runs—enough data for meaningful backtests without overloading my local storage. As a hobbyist, I don't want to deal with gigabytes of historical data daily; 2-3 years balances realism with practicality. |
| 2.3 | Yes, need Amihud illiquidity for liquidity filtering. Acceptable: Min volatility 10% annualized (to avoid dead coins), max 200% (to dodge extreme pumps/dumps), and top-N based on composite score (e.g., top 20-50 for a portfolio). | Crypto is illiquid for many assets, so filtering is crucial to avoid slippage disasters. As a hobbyist, I've seen backtests fail on low-volume coins, so this prevents that. Amihud helps quantify real trading costs. |
| 2.4 | Missing data: Fail if >5% NaNs per series (strict to ensure data quality). Outliers: Flag and auto-clip using IQR (less aggressive than z-score for crypto's fat tails). | I'm risk-averse as a hobbyist—bad data leads to garbage predictions. Clipping outliers keeps things realistic without over-filtering volatile crypto moves. |
| 2.5 | Caching: Local directory (e.g., ~/crypto_trader/cache) for simplicity. Valid for 24 hours before re-download (to handle price updates without constant API hits). | Local caching keeps it self-contained and fast for my machine. 24 hours is enough for backtesting without stale data issues in volatile crypto markets. |
| 2.6 | Config-driven feature set (e.g., RSI, MACD, ATR, Bollinger as defaults, but extensible). No PCA initially—correlation-based selection for reducing multicollinearity in crypto features. | As a hobbyist, I want flexibility without over-engineering. Config-driven lets me tweak for experiments, and correlation selection avoids overfitting on highly correlated crypto pairs. |

### 3. Models & Optimization

My platform needs to evolve from simple to advanced, but as a hobbyist, I'm starting practical—focusing on forecasting first since predicting crypto prices is core, then adding DRL for dynamic strategies. I want models that run on my GPU (if I have one) without needing massive compute.

| # | Answer | Rationale (from hobbyist perspective) |
|---|--------|-------------------------------------|
| 3.1 | Both NeuralForecast (for quick baselines like LSTM, NHITS) and custom PyTorch (LSTM-Attention, LSTM-GRU for more control). Start with NeuralForecast for ease. | NeuralForecast is user-friendly for a hobbyist like me (less coding), but custom PyTorch lets me experiment. Crypto forecasting benefits from attention for patterns in volatile data. |
| 3.2 | PPO first (stable and popular), then add A2C for simplicity if needed. Skip DDPG/SAC/TD3 initially. | PPO is robust for trading (I've tried it in smaller projects), and as a hobbyist, I don't want to debug unstable agents. One solid DRL is enough to start. |
| 3.3 | Discrete actions (hold/buy/sell per asset) with a config switch to continuous if I scale up. | Discrete is simpler for crypto (where full allocations are common), and easier to implement as a hobbyist. Continuous can come later for fine-tuning. |
| 3.4 | Configurable: Sharpe-based as default, with options for returns-based or adding penalties (turnover and drawdown). | Sharpe fits my risk-focused approach to crypto trading. Penalties prevent over-trading in volatile markets—realism matters for local backtests. |
| 3.5 | Subset: Risk parity, MVO, HRP, and efficient frontier (skip the rest for now). | These cover core needs without overwhelming a hobbyist build. Full 12 would bloat the codebase—start essential, add later. |
| 3.6 | Both forecasters and DRL, with mean aggregation initially (weighted by validation performance if possible). | Combining gives robustness in crypto's unpredictability. Mean is simple for me to implement and understand. |
| 3.7 | Grid + random search. Params: neurons, dropout, lr, batch_size, epochs. Runtimes: <30 mins per run on my machine. | Grid for thoroughness, random for exploration. These params are key for neural nets, and short runtimes keep it hobby-friendly (no overnight waits). |

### 4. Backtesting & Realism

Backtesting needs to mimic real crypto trading—high costs, slippage, and execution delays are deal-breakers. As a hobbyist, I want accurate PnL without over-complicating.

| # | Answer | Rationale (from hobbyist perspective) |
|---|--------|-------------------------------------|
| 4.1 | Default: 252/63/21 days (train/test/rebalance), configurable per run. | Standard for financial backtests, but config lets me tweak for crypto's faster cycles. As a hobbyist, flexibility helps experiments. |
| 4.2 | Separate commission (0.1%) + bid-ask spread (0.05-0.1% per asset, based on exchange). | Crypto fees vary by exchange—Binance is low, others higher. Realism prevents over-optimistic results I've seen in naive backtests. |
| 4.3 | Model slippage linearly with order size (default 0.05% per $10k trade). | Ignoring it leads to unrealistic profits in crypto. Linear is simple for a hobbyist to implement and matches my observations. |
| 4.4 | Intraday assumptions: Assume fills at next-open for day-trading, close for EOD. | Crypto trades 24/7, so next-open simulates realistic delays. As a hobbyist, this adds needed friction without intraday data complexity. |
| 4.5 | Mandatory: Cumulative returns, Sharpe, max DD, turnover, regime breakdown, and weight history. | Covers essentials for evaluating crypto strategies—regime breakdown helps with market phases. Full history for debugging. |

### 5. Regime Analysis

Crypto markets have clear regimes (bull runs, crashes), so this is key for adaptive strategies. As a hobbyist, I want it robust but not overkill.

| # | Answer | Rationale (from hobbyist perspective) |
|---|--------|-------------------------------------|
| 5.1 | All: Volatility, return (bull/bear/sideways), and correlation. | Crypto shifts quickly—volatility spikes, correlation breakdowns (e.g., altcoin decoupling). All types give a full picture. |
| 5.2 | 63-day rolling window, 33/67 percentiles for all types. | Balances sensitivity and stability. As a hobbyist, I've tested this in personal scripts—it works for crypto without too much noise. |
| 5.3 | Always regime-conditioned (Sharpe/DD by regime as default). | Strategies perform differently in bull vs. bear—mandatory for crypto to avoid overfitting to one phase. |
| 5.4 | Use uniform regime labels if fails (graceful fallback). | Better than failing—crypto data can be spotty. As a hobbyist, I prefer robustness over strictness. |

### 6. Analysis & Reporting

Reporting needs to be insightful but simple—I'm not building a dashboard, just local outputs. QuantStats is great for crypto metrics.

| # | Answer | Rationale (from hobbyist perspective) |
|---|--------|-------------------------------------|
| 6.1 | Point (MSE, MAPE, MAE) and directional (hit rate, FP rate). Skip probabilistic for now. | Covers basics for forecasting accuracy. As a hobbyist, CRPS is advanced—start simple. |
| 6.2 | Use QuantStats for full set (Sharpe, Sortino, etc.). | It's comprehensive and free—perfect for crypto risk analysis without custom coding. |
| 6.3 | All six: Summary table, rolling metrics, drawdown, regime, correlation, sensitivity. | As a hobbyist, I want thorough comparisons to iterate on strategies. All outputs help debug crypto-specific issues. |
| 6.4 | Parameters: look_ahead, train_window, transaction_cost. Range: ±20% (broader for crypto variability). | These affect backtests most. Wider range accounts for crypto's extremes. |
| 6.5 | Residual analysis and OOS stability. Skip prediction intervals initially. | Enough for diagnostics. As a hobbyist, calibration is nice but not critical yet. |
| 6.6 | CSV + PNG sufficient—skip PDF/HTML for simplicity. | I can view in Excel/Matplotlib locally. Fancy exports are overkill for my workflow. |

### 7. Operational & Deployment

Local deployment means reliability and ease—failure handling is crucial since I'm solo.

| # | Answer | Rationale (from hobbyist perspective) |
|---|--------|-------------------------------------|
| 7.1 | All four entry points (single run, grid, validate_data, resume). | Flexibility for experiments. As a hobbyist, I need all to test and iterate without hassle. |
| 7.2 | Resumable: Config + data + models + backtest results. No size limits, but <1GB checkpoints. | Full resumption saves time in crypto backtests. Size limit keeps it local-friendly. |
| 7.3 | Console + file (DEBUG) + error file. No structured logs or monitoring. | Simple for debugging. As a hobbyist, I don't need enterprise logging. |
| 7.4 | API failure: Use cache and continue. Validation failure: Fail fast. Single-model failure in ensemble: Continue with remaining. | Graceful for crypto APIs (outages happen). Fail-fast on bad data prevents errors. Ensemble robustness is key. |

### 8. Prioritization & Timeline

I'm aiming for something usable in 6 months, not a full rewrite. Defer advanced stuff to keep it MVP.

| # | Answer | Rationale (from hobbyist perspective) |
|---|--------|-------------------------------------|
| 8.1 | Target: 6 months. Defer DRL, full optimization set, PDF export to later phases. | Matches the estimate—focus on core (forecasting, backtest) for a working platform. DRL can wait; crypto forecasting is the meat. |
| 8.2 | Agree: Walk-forward backtest first, then regime, then error handling/reporting. | Backtest is foundational for crypto. As a hobbyist, getting realism right early prevents rework. |
| 8.3 | Unit tests + integration tests on synthetic data. Skip regression tests for V2 compatibility. | Production-ready means reliable. Integration ensures end-to-end; synthetic avoids real data hassles. |

