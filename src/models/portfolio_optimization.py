"""
Portfolio Optimization Models
Implements: Risk Parity, Omega, CVaR, HRP, Efficient Frontier (V3)
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform


class PortfolioWeights:
    """Standard output container for portfolio weights."""
    
    def __init__(self, weights: Dict[str, float], method: str, metadata: Optional[Dict] = None):
        self.weights = weights
        self.method = method
        self.metadata = metadata or {}
        
    def to_array(self, tickers: list) -> np.ndarray:
        """Convert to numpy array in specified ticker order."""
        return np.array([self.weights.get(t, 0.0) for t in tickers])


class RiskParityOptimizer:
    """
    Risk Parity: Inverse volatility weighting.
    Each asset contributes equally to portfolio risk.
    """
    
    def optimize(self, returns: pd.DataFrame) -> PortfolioWeights:
        """
        Args:
            returns: DataFrame of asset returns (T x N).
        
        Returns:
            PortfolioWeights object.
        """
        # Calculate volatilities
        vols = returns.std()
        
        # Inverse volatility weights
        inv_vols = 1.0 / vols
        weights_array = inv_vols / inv_vols.sum()
        
        weights_dict = dict(zip(returns.columns, weights_array))
        
        return PortfolioWeights(
            weights=weights_dict,
            method="risk_parity",
            metadata={"volatilities": vols.to_dict()}
        )


class OmegaRatioOptimizer:
    """
    Omega Ratio: Maximize probability-weighted ratio of gains vs losses.
    Omega = E[max(R - L, 0)] / E[max(L - R, 0)]
    """
    
    def __init__(self, target_return: float = 0.0):
        self.target_return = target_return
    
    def optimize(self, returns: pd.DataFrame) -> PortfolioWeights:
        """
        Args:
            returns: DataFrame of asset returns (T x N).
        
        Returns:
            PortfolioWeights object.
        """
        n_assets = returns.shape[1]
        
        # Objective: negative Omega ratio (to minimize)
        def neg_omega(w):
            port_returns = returns @ w
            excess = port_returns - self.target_return
            gains = excess[excess > 0].sum()
            losses = -excess[excess < 0].sum()
            if losses == 0:
                return -100.0  # High positive value
            return -gains / losses
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Bounds: 0 <= w <= 1
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        # Initial guess: equal weights
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            neg_omega,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            # Fallback to equal weights
            weights_array = w0
        else:
            weights_array = result.x
            
        weights_dict = dict(zip(returns.columns, weights_array))
        
        return PortfolioWeights(
            weights=weights_dict,
            method="omega",
            metadata={
                "target_return": self.target_return,
                "omega_ratio": -result.fun if result.success else None,
                "success": result.success
            }
        )


class CVaROptimizer:
    """
    CVaR (Conditional Value at Risk): Minimize expected loss in worst cases.
    Also known as Expected Shortfall (ES).
    """
    
    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence
        self.alpha = 1.0 - confidence
    
    def optimize(self, returns: pd.DataFrame) -> PortfolioWeights:
        """
        Args:
            returns: DataFrame of asset returns (T x N).
        
        Returns:
            PortfolioWeights object.
        """
        n_assets = returns.shape[1]
        T = len(returns)
        
        # CVaR objective
        def cvar_objective(w):
            port_returns = returns @ w
            var_threshold = np.percentile(port_returns, self.alpha * 100)
            # CVaR is mean of returns below VaR
            cvar = port_returns[port_returns <= var_threshold].mean()
            return -cvar  # Minimize loss (maximize negative loss)
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        
        # Bounds: 0 <= w <= 1
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        # Initial guess: equal weights
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            cvar_objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            weights_array = w0
        else:
            weights_array = result.x
            
        # Calculate final CVaR
        port_returns = returns @ weights_array
        var_threshold = np.percentile(port_returns, self.alpha * 100)
        cvar_value = port_returns[port_returns <= var_threshold].mean()
        
        weights_dict = dict(zip(returns.columns, weights_array))
        
        return PortfolioWeights(
            weights=weights_dict,
            method="cvar",
            metadata={
                "confidence": self.confidence,
                "cvar": float(cvar_value),
                "var": float(var_threshold),
                "success": result.success
            }
        )


class HRPOptimizer:
    """
    Hierarchical Risk Parity (HRP): Machine learning-based diversification.
    Uses hierarchical clustering to allocate based on correlation structure.
    """
    
    def optimize(self, returns: pd.DataFrame) -> PortfolioWeights:
        """
        Args:
            returns: DataFrame of asset returns (T x N).
        
        Returns:
            PortfolioWeights object.
        """
        # Compute correlation and distance matrices
        corr = returns.corr()
        dist = np.sqrt((1 - corr) / 2)
        
        # Hierarchical clustering
        link = linkage(squareform(dist.values), method='single')
        
        # Get quasi-diagonal matrix
        sort_idx = self._get_quasi_diag(link)
        
        # Recursive bisection
        weights_array = self._get_recursive_bisection(returns.iloc[:, sort_idx])
        
        # Map back to original order
        weights_ordered = np.zeros(len(returns.columns))
        for i, idx in enumerate(sort_idx):
            weights_ordered[idx] = weights_array[i]
        
        weights_dict = dict(zip(returns.columns, weights_ordered))
        
        return PortfolioWeights(
            weights=weights_dict,
            method="hrp",
            metadata={"linkage": link.tolist()}
        )
    
    def _get_quasi_diag(self, link: np.ndarray) -> list:
        """Get quasi-diagonal ordering from linkage matrix."""
        link = link.astype(int)
        sort_idx = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_idx.max() >= num_items:
            sort_idx.index = range(0, sort_idx.shape[0] * 2, 2)
            df0 = sort_idx[sort_idx >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_idx[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_idx = pd.concat([sort_idx, df0]).sort_index()
            sort_idx.index = range(sort_idx.shape[0])
        
        return sort_idx.tolist()
    
    def _get_recursive_bisection(self, returns: pd.DataFrame) -> np.ndarray:
        """Recursive bisection for HRP weights."""
        cov = returns.cov()
        n = len(returns.columns)
        weights = np.ones(n)
        
        def _recurse(items):
            if len(items) == 1:
                return
            
            # Split in half
            mid = len(items) // 2
            left = items[:mid]
            right = items[mid:]
            
            # Calculate cluster variances
            cov_left = cov.iloc[left, left]
            cov_right = cov.iloc[right, right]
            
            w_left = weights[left]
            w_right = weights[right]
            
            var_left = np.dot(w_left, np.dot(cov_left, w_left))
            var_right = np.dot(w_right, np.dot(cov_right, w_right))
            
            # Allocate weight inversely proportional to variance
            alpha = 1 - var_left / (var_left + var_right)
            
            weights[left] *= alpha
            weights[right] *= (1 - alpha)
            
            # Recurse
            _recurse(left)
            _recurse(right)
        
        _recurse(list(range(n)))
        return weights


class EfficientFrontierOptimizer:
    """
    Mean-Variance Optimization (Markowitz): Efficient Frontier.
    Optimize expected return vs variance trade-off.
    """
    
    def __init__(self, target_return: Optional[float] = None, risk_aversion: float = 1.0):
        self.target_return = target_return
        self.risk_aversion = risk_aversion
    
    def optimize(self, returns: pd.DataFrame) -> PortfolioWeights:
        """
        Args:
            returns: DataFrame of asset returns (T x N).
        
        Returns:
            PortfolioWeights object.
        """
        n_assets = returns.shape[1]
        
        # Expected returns and covariance
        mu = returns.mean()
        cov = returns.cov()
        
        # Objective: minimize variance - risk_aversion * return
        def objective(w):
            portfolio_return = np.dot(w, mu)
            portfolio_var = np.dot(w, np.dot(cov, w))
            return portfolio_var - self.risk_aversion * portfolio_return
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        
        # If target return specified, add constraint
        if self.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.dot(w, mu) - self.target_return
            })
        
        # Bounds: 0 <= w <= 1
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
        
        # Initial guess: equal weights
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if not result.success:
            weights_array = w0
        else:
            weights_array = result.x
            
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights_array, mu)
        portfolio_var = np.dot(weights_array, np.dot(cov, weights_array))
        portfolio_vol = np.sqrt(portfolio_var)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        weights_dict = dict(zip(returns.columns, weights_array))
        
        return PortfolioWeights(
            weights=weights_dict,
            method="efficient_frontier",
            metadata={
                "target_return": self.target_return,
                "risk_aversion": self.risk_aversion,
                "portfolio_return": float(portfolio_return),
                "portfolio_volatility": float(portfolio_vol),
                "sharpe_ratio": float(sharpe),
                "success": result.success
            }
        )


def get_optimizer(method: str, params: Optional[Dict] = None) -> object:
    """
    Factory function to get portfolio optimizer by name.
    
    Args:
        method: Optimizer name (risk_parity, omega, cvar, hrp, efficient_frontier).
        params: Optional parameters for the optimizer.
    
    Returns:
        Optimizer instance.
    """
    params = params or {}
    
    optimizers = {
        "risk_parity": RiskParityOptimizer,
        "omega": OmegaRatioOptimizer,
        "cvar": CVaROptimizer,
        "hrp": HRPOptimizer,
        "efficient_frontier": EfficientFrontierOptimizer,
    }
    
    if method not in optimizers:
        raise ValueError(f"Unknown optimizer: {method}. Available: {list(optimizers.keys())}")
    
    return optimizers[method](**params)
