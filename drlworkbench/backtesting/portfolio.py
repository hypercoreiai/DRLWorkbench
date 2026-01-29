"""Portfolio management and tracking."""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Portfolio manager for tracking positions, cash, and performance.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize portfolio.
        
        Parameters
        ----------
        initial_capital : float, default 100000.0
            Starting capital.
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.history: List[Dict] = []
        
        logger.debug(f"Portfolio initialized with capital={initial_capital}")
    
    def reset(self, initial_capital: Optional[float] = None) -> None:
        """
        Reset portfolio to initial state.
        
        Parameters
        ----------
        initial_capital : float, optional
            New initial capital. If None, uses existing initial_capital.
        """
        if initial_capital is not None:
            self.initial_capital = initial_capital
        
        self.cash = self.initial_capital
        self.positions = {}
        self.history = []
        
        logger.debug("Portfolio reset")
    
    def get_value(self, prices: Dict[str, float]) -> float:
        """
        Get total portfolio value.
        
        Parameters
        ----------
        prices : Dict[str, float]
            Current prices for all symbols.
            
        Returns
        -------
        float
            Total portfolio value (cash + positions).
        """
        position_value = sum(
            qty * prices.get(symbol, 0.0)
            for symbol, qty in self.positions.items()
        )
        return self.cash + position_value
    
    def get_position(self, symbol: str) -> float:
        """
        Get position size for a symbol.
        
        Parameters
        ----------
        symbol : str
            Asset symbol.
            
        Returns
        -------
        float
            Position size (quantity).
        """
        return self.positions.get(symbol, 0.0)
