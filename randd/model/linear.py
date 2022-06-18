import numpy as np
from typing import Dict
from .utils import group_input
from randd.model.base import GRD
from numpy.typing import NDArray
from scipy.interpolate import interp1d


class Linear(GRD):
    def __init__(self, r: NDArray, d: NDArray) -> GRD:
        dic = group_input(r, d)
        self.f = {}
        for key in dic:
            ri, di = dic[key]
            self.f[key] = interp1d(ri, di, fill_value='extrapolate')

    def __call__(self, r: NDArray) -> NDArray:
        d = np.nan(r.shape[0])
        for i, row in enumerate(r):
            dic: Dict = group_input(row)
            rep, rate = dic.popitem()
            d[i] = self.f[rep](rate) if rep in self.f else np.nan

        return d
