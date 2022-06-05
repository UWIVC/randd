from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
import scipy.interpolate
from .rd_functions import Func1d, NanFunc1d
from .base_estimator import BaseEstimator1d, BaseEstimator2d


class LogPchipEstimator1d(BaseEstimator1d):
    """1D PCHIP rd/dr function interpolator.
       Bitrate is converted into log scale according to the reference below.
       Extrapolation is enabled, but not always reliable.

       References:
            https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
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
        def rq_func(rsamples):
            return logr_q_func(np.log10(rsamples))

        sort_idxs = np.argsort(r)
        r = r[sort_idxs]
        q = q[sort_idxs]
        logr = np.log10(r)
        logr_q_func = scipy.interpolate.PchipInterpolator(logr, q, extrapolate=True)
        return Func1d(f=rq_func)

    @staticmethod
    def _qr_fit(q: NDArray, r: NDArray) -> Func1d:
        def qr_func(qsamples):
            return np.power(10, q_logr_func(qsamples))

        q, unique_idxs = np.unique(q, return_index=True)
        r = r[unique_idxs]
        sort_idxs = np.argsort(q)
        q = q[sort_idxs]
        r = r[sort_idxs]
        logr = np.log10(r)
        q_logr_func = scipy.interpolate.PchipInterpolator(q, logr, extrapolate=True)
        return Func1d(qr_func)


class LogPchipEstimator2d(BaseEstimator2d):
    def __init__(self) -> None:
        super().__init__(LogPchipEstimator1d())
