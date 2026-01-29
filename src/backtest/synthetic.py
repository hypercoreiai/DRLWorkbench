# Synthetic data generation for stress testing (V4 — PROJECT_OUTLINE Section 4.2)

import numpy as np
import pandas as pd
from typing import Optional, List, Union

class SyntheticGenerator:
    """
    Generates synthetic market data for stress testing/regime analysis.
    Focuses on "Flash Crash" scenarios and regime shifts.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: Historical dataframe (must contain numeric columns like 'Close').
        """
        self.data = data

    def inject_flash_crash(
        self,
        magnitude: float = 0.20,
        duration: int = 5,
        recover: bool = True,
        col_name: str = "close"
    ) -> pd.DataFrame:
        """
        Simulate a flash crash scenario.

        Args:
            magnitude: Percentage drop (0.20 = 20% drop).
            duration: Number of periods for the drop to occur.
            recover: If True, price recovers to original level after drop.
            col_name: Column to apply shock to (case-insensitive search).

        Returns:
            DataFrame with injected crash.
        """
        df_shook = self.data.copy()
        
        # Find column
        target_col = None
        for c in df_shook.columns:
            if c.lower() == col_name.lower():
                target_col = c
                break
        
        if not target_col:
            # If specified column not found, try to use first numeric column or skip
            # For robustness, we'll return original if fails, or raise error
            # Here we act defensively and return original with warning (printed)
            print(f"Warning: Column {col_name} not found for stress test.")
            return df_shook

        # Pick a random start point, ensuring enough runway
        n = len(df_shook)
        if n < duration * 2 + 10:
            return df_shook # Too short
            
        start_idx = np.random.randint(10, n - duration - 10)
        
        # Generate crash curve
        # Linear drop for simplicity, or exponential
        prices = df_shook[target_col].values
        start_price = prices[start_idx]
        crash_price = start_price * (1 - magnitude)
        
        # Crash phase
        crash_trajectory = np.linspace(start_price, crash_price, duration)
        df_shook.iloc[start_idx : start_idx + duration, df_shook.columns.get_loc(target_col)] = crash_trajectory
        
        # Recovery phase (optional)
        if recover:
            recovery_duration = duration # Symmetric recovery
            if start_idx + duration + recovery_duration < n:
                recover_trajectory = np.linspace(crash_price, start_price, recovery_duration)
                df_shook.iloc[
                    start_idx + duration : start_idx + duration + recovery_duration,
                    df_shook.columns.get_loc(target_col)
                ] = recover_trajectory

        return df_shook

    def boost_volatility(
        self,
        factor: float = 2.0,
        regime_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Artificially increase volatility.
        
        Args:
            factor: Multiplier for returns standard deviation.
            regime_col: If provided, only boost vol when this column indicates 'high_vol' or similar.
                        (Not fully implemented in V3, placeholder for V4 logic).
        
        Returns:
            DataFrame with higher volatility.
        """
        # Simple method: diff -> multiply -> cumsum
        # Note: This is destructive to trend if not careful. 
        # Better approach: (return - mean) * factor + mean
        
        df_vol = self.data.copy()
        numeric_cols = df_vol.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if "volume" in col.lower() or "time" in col.lower():
                continue
                
            returns = df_vol[col].pct_change().fillna(0)
            mean_ret = returns.mean()
            
            # Boost volatility centered around mean
            new_returns = (returns - mean_ret) * factor + mean_ret
            
            # Reconstruct price series
            # We need the first price to start the reconstruction
            start_price = df_vol[col].iloc[0]
            new_prices = start_price * (1 + new_returns).cumprod()
            df_vol[col] = new_prices
            
        return df_vol
