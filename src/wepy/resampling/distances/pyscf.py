"""Distance metrics for PySCF based simulations."""

# Third Party Library
import numpy as np

# First Party Library
from wepy.resampling.distances.distance import Distance


class QMGridDensityDistance(Distance):
    """Distance between walkers using electron density sampled on a grid.

    Expects walker states to include the ``density_grid`` field produced by
    :class:`wepy.runners.pyscf.PySCFRunner`.
    """

    def __init__(self, grid_key="density_grid", normalize=True):
        self.grid_key = grid_key
        self.normalize = normalize

    def image(self, state):
        rho = np.asarray(state[self.grid_key], dtype=float).ravel()

        if self.normalize:
            total = np.sum(np.abs(rho))
            if total > 0:
                rho = rho / total

        return rho

    def image_distance(self, image_a, image_b):
        if image_a.shape != image_b.shape:
            raise ValueError("Density images must have the same shape")

        return np.sqrt(np.mean((image_a - image_b) ** 2))
