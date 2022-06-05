from abc import ABCMeta, abstractmethod
from typing import Tuple
from numpy.typing import ArrayLike, NDArray
import numpy as np
from .rd_functions import Func1d, GRDSurface2d, NanFunc1d


class BaseEstimator1d(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, r: ArrayLike, q: ArrayLike) -> Tuple[Func1d, Func1d]:
        pass

    def estimate_rq_func(self, r: ArrayLike, q: ArrayLike) -> Func1d:
        rq_func, _ = self(r, q)
        return rq_func

    def estimate_qr_func(self, r: ArrayLike, q: ArrayLike) -> Func1d:
        _, qr_func = self(r, q)
        return qr_func


class BaseEstimator2d(BaseEstimator1d):
    def __init__(
        self,
        estimator1d: BaseEstimator1d
    ) -> None:
        self.base_estimator1d = estimator1d

    def __call__(self, reps: ArrayLike, q: ArrayLike) -> Tuple[Func1d, Func1d]:
        reps = np.array(reps, dtype=float)
        q = np.array(q, dtype=float)

        # Check compatibility of shapes of arguments
        if reps.ndim != 2 or reps.shape[1] != 2:
            raise ValueError("Expected reps to be a 2d Nx2 array, but got a {:d}d array of shape {} instead.".format(reps.ndim, reps.shape))
        num_rep = reps.shape[0]

        if q.size != num_rep:
            raise ValueError("Expected {:d} entries in q, but got {:d} entries instead.".format(num_rep, q.size))
        q = q.flatten()

        # First construct a 2D GRD surface
        grd2d = GRDSurface2d(reps, q, interp1d=self.base_estimator1d.estimate_rq_func)

        # Get the envelop RD function, i.e. the highest qualities that can be achieved from each sampled r_hat
        try:
            q_hat, r_hat, _ = grd2d.get_rq_envelop()
            q_hat: NDArray
            r_hat: NDArray

            r_hat = r_hat.flatten()
            q_hat = q_hat.flatten()
            rq_func, qr_func = self.base_estimator1d(r_hat, q_hat)
        except ValueError as e:
            print('Warning: Raised {} inside GRDSurface2d.get_rq_envelop.'.format(str(e)))
            rq_func = NanFunc1d()
            qr_func = NanFunc1d()
        return rq_func, qr_func
