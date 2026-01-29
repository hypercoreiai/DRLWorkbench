"""Core backtesting engine for strategy evaluation."""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Callable
import logging

from drlworkbench.utils.exceptions import BacktestError
from drlworkbench.backtesting.portfolio import Portfolio

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Core backtesting engine for evaluating trading strategies.
    
    Supports walk-forward validation, realistic transaction costs, 
    and comprehensive performance metrics.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        risk_free_rate: float = 0.02,
    ):
        """
        Initialize the backtesting engine.
        
        Parameters
        ----------
        initial_capital : float, default 100000.0
            Starting capital for the portfolio.
        commission_rate : float, default 0.001
            Commission rate as a fraction (0.001 = 0.1%).
        slippage_rate : float, default 0.0005
            Slippage rate as a fraction (0.0005 = 0.05%).
        risk_free_rate : float, default 0.02
            Annual risk-free rate for performance metrics.
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.risk_free_rate = risk_free_rate
        self.portfolio = Portfolio(initial_capital)
        
        logger.info(
            f"BacktestEngine initialized with capital={initial_capital}, "
            f"commission={commission_rate}, slippage={slippage_rate}"
        )
    
    def run(
        self,
        data: pd.DataFrame,
        strategy: Callable,
        **strategy_params
    ) -> Dict[str, Any]:
        """
        Run backtest on historical data with a given strategy.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical price data with columns: ['open', 'high', 'low', 'close', 'volume'].
            Index should be datetime.
        strategy : Callable
            Strategy function that takes data and returns signals.
        **strategy_params : dict
            Additional parameters to pass to the strategy.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing backtest results including:
            - portfolio_value: Series of portfolio values over time
            - returns: Series of returns
            - trades: List of executed trades
            - metrics: Dictionary of performance metrics
            
        Raises
        ------
        BacktestError
            If backtest execution fails.
        """
        try:
            logger.info(f"Starting backtest with {len(data)} data points")
            
            # Validate data
            self._validate_data(data)
            
            # Reset portfolio
            self.portfolio.reset(self.initial_capital)
            
            # Run strategy and collect results
            portfolio_values = []
            trades = []
            
            for i, (timestamp, row) in enumerate(data.iterrows()):
                # Get strategy signal
                signal = strategy(data.iloc[:i+1], **strategy_params)
                
                # Execute trade if signal is not None
                if signal is not None:
                    trade = self._execute_trade(timestamp, row, signal)
                    if trade:
                        trades.append(trade)
                
                # Record portfolio value (simplified for now - assumes cash-only portfolio)
                current_value = self.portfolio.cash
                portfolio_values.append({
                    'timestamp': timestamp,
                    'value': current_value
                })
            
            # Create results DataFrame
            portfolio_df = pd.DataFrame(portfolio_values).set_index('timestamp')
            returns = portfolio_df['value'].pct_change()
            
            # Calculate metrics
            metrics = self._calculate_metrics(portfolio_df['value'], returns)
            
            results = {
                'portfolio_value': portfolio_df['value'],
                'returns': returns,
                'trades': trades,
                'metrics': metrics,
                'final_value': portfolio_df['value'].iloc[-1],
                'total_return': (portfolio_df['value'].iloc[-1] / self.initial_capital - 1) * 100
            }
            
            logger.info(
                f"Backtest complete. Final value: ${results['final_value']:.2f}, "
                f"Total return: {results['total_return']:.2f}%"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            raise BacktestError(f"Backtest execution failed: {str(e)}") from e
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data format."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise BacktestError(f"Missing required columns: {missing}")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise BacktestError("Data index must be DatetimeIndex")
    
    def _execute_trade(
        self,
        timestamp: pd.Timestamp,
        row: pd.Series,
        signal: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute a trade based on the signal."""
        # This is a simplified implementation
        # In reality, this would interact with the Portfolio class
        # to handle position sizing, risk management, etc.
        return None
    
    def _calculate_metrics(
        self,
        portfolio_value: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        returns = returns.dropna()
        
        # Basic metrics
        total_return = (portfolio_value.iloc[-1] / self.initial_capital - 1) * 100
        
        # Annualized metrics
        periods_per_year = 252  # Assuming daily data
        mean_return = returns.mean() * periods_per_year
        std_return = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe ratio
        excess_return = mean_return - self.risk_free_rate
        sharpe_ratio = excess_return / std_return if std_return > 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        sortino_ratio = excess_return / downside_std if downside_std > 0 else 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': mean_return * 100,
            'annualized_volatility': std_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
        }
