from analytics.portfolio_types import CovariancesReturns, Covariances, Portfolio, WeightedAsset
from utils.strong_types import AssetId, Kappa, Returns
import numpy.typing as npt
import numpy as np



def optimize_weights(assets: list[AssetId], covariant_returns: CovariancesReturns, kappa: Kappa) -> Portfolio:
    """
    Return optimized weights

    >>> covs = Covariances({AssetId("A"): {AssetId("B"): 0.5, AssetId("A"): 1.0}, AssetId("B"): {AssetId("B"): 2.0}})
    >>> returns = Returns({AssetId("A"): 0.05, AssetId("B"): 0.10})
    >>> cov_returns = CovariancesReturns(returns, covs)
    >>> print(optimize_weights([AssetId("A"), AssetId("B")], cov_returns, Kappa(1.0)))
    [WeightedAsset(asset='A', weight=np.float64(0.3999999999999999)), WeightedAsset(asset='B', weight=np.float64(0.6))]
    """
    assert len(assets) > 0
    sorted_weights = np.linalg.solve(kappa * covariant_returns.covariances.to_numpy(assets),
                                    np.array([covariant_returns.returns[asset] for asset in assets]))
    sorted_weights /= sum(sorted_weights)
    return Portfolio([
        WeightedAsset(asset, w) for asset, w in zip(assets, sorted_weights)
    ])

def imply_returns_from_optimal_weights(portfolio: Portfolio, covs: Covariances, kappa: Kappa) -> Returns:
    """
    Given the optimal weights and the coveriance, return the implied returns. 
    
    >>> covs = Covariances({AssetId("A"): {AssetId("B"): 0.5, AssetId("A"): 1.0}, AssetId("B"): {AssetId("B"): 2.0}})
    >>> portfolio = Portfolio([WeightedAsset(AssetId("A"), 0.4), WeightedAsset(AssetId("B"), 0.6)])
    >>> imply_returns_from_optimal_weights(portfolio, covs, Kappa(1.0))
    {'A': np.float64(0.7), 'B': np.float64(1.4)}
    """
    assets = [weighted_asset.asset for weighted_asset in portfolio.weighted_assets]
    weights = [weighted_asset.weight for weighted_asset in portfolio.weighted_assets]
    returns = kappa * covs.to_numpy(assets).dot(weights)
    return Returns({asset: r for asset,r in zip(assets, returns)})
