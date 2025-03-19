import numpy as np

from utils.strong_types import AssetId


class CovarianceSizeMismatchError(Exception):
    def __init__(self, covariance_size: int, portfolio_size: int) -> None:
        super().__init__(f"Covariance matrix size {covariance_size} must match portfolio size {portfolio_size}")


class MatrixIsNotSquareError(Exception):
    def __init__(self, num_lines: int, num_columns: int) -> None:
        super().__init__(f"Covariance matrix must be square, lines: {num_lines}, columns: {num_columns}")


class WeightsDontSumToHundredPercentError(Exception):
    def __init__(self, total_weights: np.float64) -> None:
        super().__init__(f"Weight don't sum to 100%, sum of weights: {total_weights:.6%}")


class UnavailableCovarianceError(Exception):
    def __init__(self, asset_1: AssetId, asset_2: AssetId) -> None:
        super().__init__(f"Could not find covariance for the pair: {asset_1}, {asset_2}")


class EmptyAssetListError(Exception):
    def __init__(self) -> None:
        super().__init__("List of asset is empty.")
