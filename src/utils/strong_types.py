from typing import NewType

import numpy as np

AssetId = NewType("AssetId", str)
Covariance = NewType("Covariance", np.float64)
Return = NewType("Return", np.float64)
Returns = NewType("Returns", dict[AssetId, Return])
Kappa = NewType("Kappa", np.float64)