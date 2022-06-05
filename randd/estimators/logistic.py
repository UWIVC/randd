from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import curve_fit
from randd.estimators.rd_functions import Func1d, NanFunc1d
from randd.estimators.base_estimator import BaseEstimator1d, BaseEstimator2d


class LogLogisticEstimator1d(BaseEstimator1d):
    """1D logistic rd/dr function estimator. Bitrate is converted into log scale according to the reference below.
       Extrapolation is automatically enabled.

    References:
            P. Hanhart, T. Ebrahimi, "Calculation of average coding efficiency based on subjective quality scores,"
            Journal of Visual Communication and Image Representation, vol. 25, issue 3, pp 555--564. Apr. 2014.
    """
    def __call__(self, r: ArrayLike, q: ArrayLike) -> Tuple[Func1d, Func1d]:
        r = np.array(r, dtype=float)
        q = np.array(q, dtype=float)
        sort_idx = np.argsort(r)
        r = r[sort_idx]
        q = q[sort_idx]

        def r_q_func(rsamples: NDArray) -> NDArray:
            return logr_q_func(np.log10(rsamples))

        def q_r_func(qsamples: NDArray) -> NDArray:
            return np.power(10, q_logr_func(qsamples))

        try:
            logr = np.log10(r)
            logr_q_func = LogisticFunction(a=min(q), b=max(q)).fit(logr, q)
            rq_func = Func1d(r_q_func)
        except (ValueError, TypeError, RuntimeError) as e:
            print('Warning: Raised {} internally.'.format(str(e)))
            rq_func = NanFunc1d()

        try:
            q_logr_func = logr_q_func.inverse()
            qr_func = Func1d(q_r_func)
        except (ValueError, TypeError, RuntimeError, UnboundLocalError) as e:
            print('Warning: Raised {} internally.'.format(str(e)))
            qr_func = NanFunc1d()

        return rq_func, qr_func


class LogisticFunction:
    def __init__(
        self,
        a: float = 0,
        b: float = 100,
        c: float = 1,
        d: float = 0
    ):
        self.a, self.b, self.c, self.d = a, b, c, d

    def set_params(self, a=None, b=None, c=None, d=None):
        self.a = self.a if a is None else a
        self.b = self.b if b is None else b
        self.c = self.c if c is None else c
        self.d = self.d if d is None else d

    def fit(self, x, y):
        def func(xdata, a, b, c, d):
            return a + (b - a) / (1 + np.exp(-c * (xdata - d)))

        popt, _ = curve_fit(
            func, x, y, p0=(self.a, self.b, self.c, self.d),
            bounds=((min(0, self.a), 0, 0, 0), (100, max(100, self.b), np.inf, np.inf))
        )
        self.a, self.b, self.c, self.d = popt
        return self

    def integral(self, x, const=0):
        """Integrated logistic function

        Arguments:
            x {np.array} -- input

        Keyword Arguments:
            const {int} -- constant of the integral function (default: {0})

        Returns:
            np.array -- y = (b - a) / c * ln{1 + exp[-c * (r - d)]} + b * x + (a - b) * d + const
        """
        return (self.b - self.a) / self.c * np.log(1 + np.exp(-self.c * (x - self.d))) + self.b * x + (self.a - self.b) * self.d + const

    def inverse(self):
        return InverseLogisticFunction(self.a, self.b, self.c, self.d)

    def __call__(self, x):
        """Logistic function

        Arguments:
            x {np.array} -- input

        Returns:
            np.array -- y = a + (b - a) / (1 + exp(-c * (r - d)))
        """
        return self.a + (self.b - self.a) / (1 + np.exp(-self.c * (x - self.d)))


class InverseLogisticFunction:
    def __init__(
        self,
        a: float = 0,
        b: float = 100,
        c: float = 1,
        d: float = 0
    ):
        self.a, self.b, self.c, self.d = a, b, c, d

    def set_params(self, a=None, b=None, c=None, d=None):
        self.a = self.a if a is None else a
        self.b = self.b if b is None else b
        self.c = self.c if c is None else c
        self.d = self.d if d is None else d

    def fit(self, x, y):
        def func(xdata, a, b, c, d):
            return a + (b - a) / (1 + np.exp(-c * (xdata - d)))

        popt, _ = curve_fit(
            func, y, x, p0=(self.a, self.b, self.c, self.d),
            bounds=((0, 0, 0, 0), (100, 100, np.inf, np.inf))
        )
        self.a, self.b, self.c, self.d = popt
        return self

    def __call__(self, x):
        """Inverse logistic function

        Arguments:
            x {np.array} -- input

        Returns:
            np.array -- y = - (1 / c) * ln[(b - x) / (x - a)] + d
        """
        return (-1 / self.c) * np.log((self.b - x) / (x - self.a)) + self.d

    def inverse(self):
        return LogisticFunction(self.a, self.b, self.c, self.d)

    def integral(self, x, const=0):
        """Integrated inverse logistic function
        Arguments:
            x {np.array} -- input
        Keyword Arguments:
            const {int} -- constant of the integral function (default: {0})
        Returns:
            np.array -- y = (b - x) / c * [ln(b - x) - 1] + (x - a) / c * [ln(x - a) - 1] + d * x + const
        """
        return (self.b - x) / self.c * (np.log(self.b - x) - 1) + (x - self.a) / self.c * (np.log(x - self.a) - 1) + self.d * x + const


class LogLogisticEstimator2d(BaseEstimator2d):
    def __init__(self) -> None:
        super().__init__(LogLogisticEstimator1d())
