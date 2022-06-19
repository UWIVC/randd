from typing import Tuple, Type
from numpy.typing import NDArray
from randd.model.base import GRD
from scipy.interpolate import interp1d


class Estimator:
    """Base RD/DR function estimator.

    RandD produces a pair RD/DR functions with the following steps:
    #. Fit a generalized rate-distortion function with given representations and distortion.
    #. Produce dense samples of the rate-distortion function, based on the convex hull of GRD.
    #. Produce continuous approximation of the RD/DR functions.

    Each RD/DR estimator should be its subclass, and provide an implementation of
    :meth:`randd.estimators.Estimator.fit` for how the GRD should be estimated.

    Args:
        step (int): number of steps used in the interpolation.
            Defaults to ``1000``.
        mode (str): interpolation mode.
            Defaults to ``linear``.
    """
    def __init__(
        self,
        model: Type[GRD],
        ndim: int = 1,
        d_measure: str = 'psnr',
        n_samples: int = 1001,
        r_roi: Tuple[float, float] = (100, 10000),
        mode: str = "linear",
    ) -> None:
        self.n_samples = n_samples
        self.mode = mode
        self.grd = model
        self.ndim = ndim
        self.r_roi = r_roi
        self.d_measure = d_measure

    def __call__(self, r: NDArray, d: NDArray) -> Tuple[interp1d, interp1d]:
        # fit the generalized rate-distortion function
        grd = self.grd(r, d, d_measure=self.d_measure, ndim=self.ndim)
        # densely sample rate-distortion function from the convex hull of the grd
        r, d = grd.convex_hull(r_roi=self.r_roi, n_samples=self.n_samples)
        # produce continuous approximation of the RD/DR functions
        rd = interp1d(r, d, kind=self.mode, fill_value='extrapolate')
        dr = interp1d(d, r, kind=self.mode, fill_value='extrapolate')
        return rd, dr
