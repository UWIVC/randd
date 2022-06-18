import numpy as np
from typing import Dict
from randd.model.base import GRD
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class Linear(GRD):
    def __init__(self, r: NDArray, d: NDArray) -> None:
        dic = self._group_input(r, d)
        self.f = {}
        for key in dic:
            ri, di = dic[key]
            self.f[key] = interp1d(ri, di, fill_value='extrapolate')

    def __call__(self, r: NDArray) -> NDArray:
        d = np.zeros(r.shape[0])
        for i, row in enumerate(r):
            row = np.expand_dims(row, axis=0)
            dic: Dict = self._group_input(row)
            hparam, value = dic.popitem()
            rate, _ = value
            d[i] = self.f[hparam](rate[0]) if hparam in self.f else np.nan

        return d
