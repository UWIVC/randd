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
        d_measure: str = 'psnr',
        step: int = 1000,
        mode: str = "linear",
    ) -> None:
        self.step = step
        self.mode = mode
        self.grd = model
        self.d_measure = d_measure

    def __call__(self, r: NDArray, d: NDArray) -> Tuple[interp1d, interp1d]:
        # fit the generalized rate-distortion function
        grd = self.grd(r, d, d_measure=self.d_measure)
        # densely sample rate-distortion function from the convex hull of the grd
        r, d = grd.convex_hull(step=self.step)
        # produce continuous approximation of the RD/DR functions
        rd = interp1d(r, d, kind=self.mode)
        dr = interp1d(d, r, kind=self.mode)
        return rd, dr
