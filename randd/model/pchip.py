import numpy as np
from typing import Dict
from randd.model.base import GRD
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator


class LogPCHIP(GRD):
    r"""1D PCHIP rate-distortion function interpolator.

    Bitrate is converted into log scale according to the reference below.
    Extrapolation is enabled, but not always reliable.

    Args:
        r (NDArray): Encoding representations.
        d (NDArray): Corresponding distortions.
        d_measure (str): Name of the distortion measure.
        ndim (int): Number of dimensions of the RD function domain.

    References:
        https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
    """
    def __init__(self, r: NDArray, d: NDArray, d_measure: str, ndim: int) -> None:
        super().__init__(r, d, d_measure, ndim)
        dic = self._group_input(r, d)
        self.f = {}
        for key in dic:
            ri, di = dic[key]
            logr = np.log10(ri)
            self.f[key] = PchipInterpolator(logr, di, extrapolate=True)

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
            logr = np.log10(rate[0])
            d[i] = self.f[hparam](logr) if hparam in self.f else np.nan

        return d
