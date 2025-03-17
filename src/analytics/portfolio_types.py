from itertools import combinations
from typing import NamedTuple

import numpy as np
import polars as pl

from utils.strong_types import AssetId, Covariance, Return, Returns
from analytics.exceptions import UnavailableCovarianceError, WeightsDontSumToHundredPercentError



class WeightedAsset(NamedTuple):
    asset: AssetId
    weight: float


class Portfolio:
    TOL = 1e-6

    def __init__(self, weighted_assets: list[WeightedAsset]):
        sum_of_weights = sum(weighted_asset.weight for weighted_asset in weighted_assets)
        if abs(sum_of_weights - 1.0) > self.TOL:
            raise WeightsDontSumToHundredPercentError(sum_of_weights)
        self.weighted_assets = weighted_assets

    def __str__(self):
        return str(self.weighted_assets)


class Covariances:
    TOL = 1e-6

    def __init__(self, covariances_map: dict[AssetId, dict[AssetId, Covariance]]):
        """
        Store covariances and check their integrity (positive-definite).
        """
        self.covariances_map = covariances_map
        self.assets = set(covariances_map.keys()).union(covariances_map[next(iter(covariances_map))])
        self.check_covariances_integrity()

    def check_covariances_integrity(self):
        """
        We check that every pair of asset has a covariance defined and that coveriance is positive definite.
        """
        for asset_1, asset_2 in combinations(self.covariances_map.keys(),2):
            self.covariance(asset_1, asset_2) >= 0.0
            abs(self.covariance(asset_1, asset_2) - self.covariance(asset_2, asset_1)) < self.TOL

    def covariance(self, asset_1: AssetId, asset_2: AssetId) -> Covariance:
        """
        Return the covariance of two assets.
        >>> covs = Covariances({AssetId("A"): {AssetId("B"): 0.5, AssetId("A"): 1.0}, AssetId("B"): {AssetId("B"): 2.0}})
        >>> covs.covariance(AssetId("A"), AssetId("B"))
        0.5
        >>> try:
        ...     covs.covariance(AssetId("A"), AssetId("C"))
        ... except Exception as e:
        ...     e
        UnavailableCovarianceError('Could not find covariance for the pair: A, C')
        """
        try:
            return self.covariances_map[asset_1][asset_2]
        except KeyError:
            try:
                return self.covariances_map[asset_2][asset_1]
            except KeyError:
                raise UnavailableCovarianceError(asset_1, asset_2) from None

    def to_numpy(self, identifiers: list[AssetId] | None):
        """
        Convert to covariance matrix

        >>> covs = Covariances({AssetId("A"): {AssetId("B"): 0.5, AssetId("A"): 1.0}, AssetId("B"): {AssetId("B"): 2.0}})
        >>> covs.to_numpy(None)
        array([[1. , 0.5],
               [0.5, 2. ]])
        """
        if identifiers is None:
            identifiers = sorted(self.assets)
        return np.array([[self.covariance(asset_i, asset_j) for asset_j in identifiers] for asset_i in identifiers])  # noqa: SIM118

    def to_dataframe(self, identifiers: list[AssetId] | None):
        """
        Convert to polars dataframe

        >>> covs = Covariances({AssetId("A"): {AssetId("B"): 0.5, AssetId("A"): 1.0}, AssetId("B"): {AssetId("B"): 2.0}})
        >>> covs.to_dataframe(None)
        shape: (2, 2)
        ┌─────┬─────┐
        │ A   ┆ B   │
        │ --- ┆ --- │
        │ f64 ┆ f64 │
        ╞═════╪═════╡
        │ 1.0 ┆ 0.5 │
        │ 0.5 ┆ 2.0 │
        └─────┴─────┘
        """
        if identifiers is None:
            identifiers = sorted(self.assets)
        return pl.from_numpy(self.to_numpy(identifiers), schema=identifiers)


class CovariancesReturns(NamedTuple):
    returns: Returns
    covariances: Covariances
