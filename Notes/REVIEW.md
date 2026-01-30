# Codebase Review & Gap Analysis

## Current Status
The codebase is currently a **partial implementation of the V3 Outline**.
- **Structure**: matches V3 exactly.
- **Implemented**: 
  - `src/data/validator.py` (Data checks implemented).
  - `src/models/ensemble.py` (Basic voting logic implemented).
- **Missing / Skeleton**:
  - `src/backtest/walker.py`: Class exists but `run()` method is empty (no training/trading loop).
  - `src/run_pipeline.py`: Orchestrator is a stub.
  - No GPU/PyTorch integration visible yet in inspected files.

## Comparison to Plans

### V3 (Current Codebase Target)
- **Status**: ~30% Complete.
- **Critical Components Missing**:
  - **Walk-Forward Loop**: The core logic to iterate time windows is missing in `walker.py`.
  - **Pipeline Wiring**: `run_pipeline.py` does not connect data -> model -> backtest.

### V4 (New Requirements)
- **Status**: 0% Implemented (V4 adds to V3).
- **New Requirements vs Current Codebase**:
  - **Crypto-First Validation**: `validator.py` needs strict "no gap" checks for 24/7 markets.
  - **Synthetic Stress Testing**: Missing entirely (`src/backtest/synthetic.py` needed).
  - **GPU Acceleration**: Current `ensemble.py` uses CPU-bound Numpy. Needs PyTorch migration.
  - **Regime Analysis**: `regime.py` exists but needs to be integrated into the backtest loop.

## Recommendations for Improvement

1. **Prioritize Backtest Engine (V3 Core)**
   - Implement `WalkForwardBacktester.run()` in `src/backtest/walker.py`. This is the engine of the entire project. Without it, models cannot be tested.

2. **Implement Stress Testing (V4 Feature)**
   - Create `src/backtest/synthetic.py` to generate "Flash Crash" scenarios as requested in V4.

3. **Upgrade to Crypto Standards (V4 Feature)**
   - Enhance `src/data/validator.py` with `check_crypto_continuity` (handling 24/7 timestamps).

4. **GPU Activation**
   - Ensure the `models` module imports `torch` and moves tensors to `cuda` if available.

## Proposed Next Steps
1. Create `implementation_plan.md` to formalize the upgrade to V4.
2. Fill in logic for `walker.py`.
3. Create `synthetic.py`.
