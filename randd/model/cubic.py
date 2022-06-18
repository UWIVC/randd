import numpy as np
from typing import Dict
from randd.model.base import GRD
from numpy.typing import NDArray


class LogCubic(GRD):
    r"""1D log cubic function estimator. Bitrate is converted into log scale according to the reference below.
        Extrapolation is automatically enabled, but not reliable.
        Used in estimation of BD-PSNR and BD-Rate.

    References:
        G. Bjøntegaard, "Calculation of average PSNR differences between rdcurves,
        Austin, TX, USA, Tech. Rep. VCEG-M33, ITU-T SG 16/Q6, 13th VCEG Meeting, Apr. 2001.
        G. Bjøntegaard, "Improvements of the BD-PSNR model," Berlin, Germany, Tech.
        Rep. VCEG-AI11, ITU-T SG 16/Q6, 35th VCEG Meeting, Jul. 2008.
    """
    def __init__(self, r: NDArray, d: NDArray) -> None:
        dic = self._group_input(r, d)
        self.f = {}
        for key in dic:
            ri, di = dic[key]
            logr = np.log10(ri)
            self.f[key] = np.poly1d(np.polyfit(logr, di, deg=3))

    def __call__(self, r: NDArray) -> NDArray:
        d = np.zeros(r.shape[0])
        for i, row in enumerate(r):
            dic: Dict = self._group_input(row)
            hparam, value = dic.popitem()
            rate, _ = value
            logr = np.log10(rate[0])
            d[i] = self.f[hparam](logr) if hparam in self.f else np.nan

        return d
