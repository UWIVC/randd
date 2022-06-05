from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.interpolate
from .rd_functions import Func1d, NanFunc1d
from .base_estimator import BaseEstimator1d, BaseEstimator2d


class LinearEstimator1d(BaseEstimator1d):
    """1D piece-wise linear rd/dr function interpolator. Linear extrapolation is enabled.
    """
    def __call__(self, r: ArrayLike, q: ArrayLike) -> Tuple[Func1d, Func1d]:
        r = np.array(r, dtype=float)
        q = np.array(q, dtype=float)
        sort_idx = np.argsort(r)
        r = r[sort_idx]
        q = q[sort_idx]
        try:
            rq_func = self._rq_fit(r, q)
        except ValueError or TypeError as e:
            print('Warning: Raised {} internally.'.format(str(e)))
            rq_func = NanFunc1d()

        try:
            qr_func = self._qr_fit(q, r)
        except ValueError or TypeError as e:
            print('Warning: Raised {} internally.'.format(str(e)))
            qr_func = NanFunc1d()
        return rq_func, qr_func

    @staticmethod
    def _rq_fit(r: NDArray, q: NDArray) -> Func1d:
        q = np.maximum.accumulate(q)
        _, uniq_idxs = np.unique(q, return_index=True)
        if uniq_idxs[-1] != q.size - 1:
            uniq_idxs = np.concatenate((uniq_idxs, [q.size - 1]))
        r = r[uniq_idxs]
        q = q[uniq_idxs]
        rq_func = scipy.interpolate.interp1d(r, q, fill_value='extrapolate')
        return Func1d(f=rq_func)

    @staticmethod
    def _qr_fit(q: NDArray, r: NDArray) -> Func1d:
        q = np.maximum.accumulate(q)
        q, unique_idxs = np.unique(q, return_index=True)
        r = r[unique_idxs]
        qr_func = scipy.interpolate.interp1d(q, r, fill_value='extrapolate')
        return Func1d(qr_func)


class LinearEstimator2d(BaseEstimator2d):
    """2D piece-wise linear rd/dr function envelop interpolator.
       Linear extrapolation is enabled on each resolution.
    """
    def __init__(self) -> None:
        super().__init__(LinearEstimator1d())
