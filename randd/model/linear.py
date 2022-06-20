import numpy as np
from typing import Dict
from randd.model.base import GRD
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class Linear(GRD):
    r"""1D Linear rate-distortion function interpolator.

    Extrapolation is enabled, but not always reliable.

    Args:
        r (NDArray): Encoding representations.
        d (NDArray): Corresponding distortions.
        d_measure (str): Name of the distortion measure.
        ndim (int): Number of dimensions of the RD function domain.
    """
    def __init__(self, r: NDArray, d: NDArray, d_measure: str = 'psnr', ndim: int = 1) -> None:
        super().__init__(r, d, d_measure, ndim)
        dic = self._group_input(r, d)
        for key in dic:
            ri, di = dic[key]
            self.f[key] = interp1d(ri, di, fill_value='extrapolate')

    def __call__(self, r: NDArray) -> NDArray:
        r"""Predict the distortion at the given representation.

        Args:
            r (NDArray): Input encoding representation.

        Returns:
            NDArray: Predicted distortion.
        """
        d = np.zeros(r.shape[0])
        for i, row in enumerate(r):
            row = np.expand_dims(row, axis=0)
            dic: Dict = self._group_input(row)
            hparam, value = dic.popitem()
            rate, _ = value
            d[i] = self.f[hparam](rate[0]) if hparam in self.f else np.nan

        return d
