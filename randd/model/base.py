import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Optional


class GRD:
    r"""Base rate-distortion function estimator.

    Any generalized rate-distortion function model should be its subclass.
    The inherited class should provide the implementation of :meth:`__init__`
    and :meth:`__call__`.

    :meth:`__init__` takes the RD samples, and produces a continuous RD function at each
    encoding attribute other than bitrate. The function at each encoding attribute is
    represented by a dictionary ``self.f``.

    :meth:`__call__` employs ``self.f`` to predict the distortion level at the given r.

    Args:
        r (NDArray): Encoding representations.
        d (NDArray): Corresponding distortions.
        d_measure (str): Name of the distortion measure.
            Defaults to ``psnr``.
        ndim (int): Number of dimensions of the RD function domain.
            Defaults to ``1``.
    """
    def __init__(self, r: NDArray, d: NDArray, d_measure: str = 'psnr', ndim: int = 1) -> None:
        self.d_measure = d_measure
        self.ndim = ndim
        self.f = {}

    def __call__(self, r: NDArray) -> NDArray:
        pass

    def _group_input(
        self, r: NDArray, d: Optional[NDArray] = None
    ) -> Dict[Any, Tuple[NDArray, NDArray]]:
        out = {}
        if d is None:
            d = np.empty(r.shape[0])

        if r.shape[1] == 1:
            key = 'default'
            out[key] = (r.flatten(), d)
            return out

        for (row, di) in zip(r, d):
            key, value = tuple(row[1:]), row[0]
            out.setdefault(key, ([], []))
            out[key][0].append(value)
            out[key][1].append(di)

        for key in out:
            r, d = out[key]
            out[key] = (np.asarray(r, dtype=float), np.asarray(d, dtype=float))

        return out

    def convex_hull(
        self, r_roi: Tuple[float, float] = (100, 10000), n_samples: int = 1001
    ) -> Tuple[NDArray, NDArray, NDArray]:
        r_min = r_roi[0]
        r_max = r_roi[1]
        assert r_max > r_min > 0

        r_grid = np.linspace(r_min, r_max, num=n_samples)
        q_on_grid = [self.f[key](r_grid) for key in self.f.keys()]
        q_on_grid = np.stack(q_on_grid, axis=0)

        valid_mask = np.logical_not(np.all(np.isnan(q_on_grid), axis=0))
        q_on_grid = q_on_grid[:, valid_mask]
        r_grid: NDArray = r_grid[valid_mask]

        if r_grid.size < n_samples * 0.5:
            raise ValueError("Too many NaNs in the estimated GRD surface.")

        d_opt = np.nanmax(q_on_grid, axis=0)
        return r_grid, d_opt
