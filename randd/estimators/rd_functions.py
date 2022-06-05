from typing import Callable, Optional, Union
import scipy.interpolate
import numpy as np
from numpy.typing import ArrayLike, NDArray


class FuncNd:
    def __init__(
        self,
        f: Callable[[ArrayLike], NDArray],
        n: int = 1
    ) -> None:
        """A general class of n-variate function. This class will check dimension compability of input.

        Args:
            f (Callable[[ArrayLike], NDArray]): The real function being excuted
            n (int, optional): The correct dimension of input variable. Defaults to 1.

        TODO: Check whether f takes exactly n arguments.
        """
        self.func = f
        self.n = n

    def __call__(self, x: ArrayLike) -> NDArray:
        x = np.asarray(x)

        if self.n == 1:
            if x.ndim <= 1:
                x = x[..., np.newaxis]

        if self.n != x.shape[-1]:
            raise ValueError("The size of the last dimension of the input array to a {:d}-variate function should be {:d} as well."
                             "Got an input array of {} instead".format(self.n, self.n, x.shape))

        split_x = [x[..., i] for i in range(self.n)]
        return self.func(*split_x)


class Func1d(FuncNd):
    def __init__(self, f: Callable[[ArrayLike], NDArray]) -> None:
        super().__init__(f, n=1)


class Func2d(FuncNd):
    def __init__(self, f: Callable[[ArrayLike], NDArray]) -> None:
        super().__init__(f, n=2)


def nan_func(x: ArrayLike) -> NDArray:
    x = np.asarray(x)
    return np.full(x.shape, np.nan)


class NanFunc1d(Func1d):
    def __init__(self) -> None:
        super().__init__(f=nan_func)


class GRDSurface2d:
    """This class defines a 2d GRD surface, which takes continuous bitrate and discrete resolution as input.

    Args:
        reps (ArrayLike): An (N, 2) array. Each row represents a data point.
            First column: bitrates; second column: resolution.
        qualities (ArrayLike): Quality scores of the sampled data points.
        interp1d (Callable[[NDArray, NDArray], Callable[[Union[ArrayLike, NDArray]], NDArray]], optional):
            A 1d function estimator to generate the RQ/RD curve for each resolution.
            Defaults to scipy.interpolate.interp1d.

    Raises:
        ValueError: ValueError raised when reps.shape is not compatible.
        ValueError: ValueError raised when number of reps and qualities does not match.
    """
    def __init__(
        self,
        reps: ArrayLike,
        qualities: ArrayLike,
        interp1d: Callable[[NDArray, NDArray], Callable[[Union[ArrayLike, NDArray]], NDArray]] = scipy.interpolate.interp1d
    ) -> None:
        # Handle input data
        reps = np.array(reps, dtype=float)
        qualities = np.array(qualities, dtype=float)

        # Currently, we require that the input representations have at most two dimensions, bitrate and resolution.
        if reps.ndim != 2 or reps.shape[1] != 2:
            raise ValueError("Expected reps to be a 2d Nx2 array, but got a {:d}d array of shape {} instead.".format(reps.ndim, reps.shape))
        self.reps = reps
        self.num_rep = self.reps.shape[0]

        if qualities.size != self.num_rep:
            raise ValueError("Expected {:d} quality entries, but got {:d} entries instead.".format(self.num_rep, qualities.size))
        self.qualities = qualities.flatten()

        resolutions, groups_of_reps = np.unique(self.reps[:, 1], return_inverse=True)
        self.resolutions = resolutions.astype(int)
        self.num_resolution = self.resolutions.size

        # Unerlying 1d rq/rd curve estimator
        self.interp1d = interp1d

        # Estimate a 1d function for each resolution
        self.continuous_rq_funcs = {
            int(self.resolutions[i]):
            self.interp1d(self.reps[:, 0][groups_of_reps == i],
                          self.qualities[groups_of_reps == i])
            for i in range(self.num_resolution)
        }

        self.bitrates = {int(self.resolutions[i]): self.reps[:, 0][groups_of_reps == i]
                         for i in range(self.num_resolution)}

    def __call__(self, input_r: NDArray, input_s: NDArray, strict: bool = True) -> NDArray:
        assert input_r.shape == input_s.shape, "The shapes of input bitrates and spatial resolution must match with each other."
        out = np.empty(input_r.shape)
        out[:] = np.nan

        for res in self.resolutions:
            mask = input_s == int(res)
            if np.any(mask):
                continuous_rq_func = self.continuous_rq_funcs[int(res)]
                out[mask] = continuous_rq_func(input_r[mask])

        if np.any(np.isnan(out)):
            if strict:
                raise ValueError("Some queried quality values cannot be resolved, "
                                 "because the rate-quality functions at corresponding resolutions do not exist or "
                                 "the queried bitrates are out of the support domain.")
            else:
                print("Warning: Some queried quality values cannot be resolved, "
                      "because the rate-quality functions at corresponding resolutions do not exist or "
                      "the queried bitrates are out of the support domain.")

        return out

    def get_discrete_version(self):
        return self.reps, self.qualities

    def get_rq_envelop(self, r_min: Optional[float] = None, r_max: Optional[float] = None, n_samples: int = 1001):
        if r_min is None:
            r_min = np.min(self.reps[:, 0])
        if r_max is None:
            r_max = np.max(self.reps[:, 0])
        assert r_max > r_min > 0

        r_grid = np.linspace(r_min, r_max, num=n_samples)
        q_on_grid = [self.continuous_rq_funcs[res](r_grid) for res in self.resolutions]
        q_on_grid = np.stack(q_on_grid, axis=0)

        valid_mask = np.logical_not(np.all(np.isnan(q_on_grid), axis=0))
        q_on_grid = q_on_grid[:, valid_mask]
        r_grid = r_grid[valid_mask]

        if r_grid.size < n_samples * 0.5:
            raise ValueError("Too many NaNs in the estimated GRD surface.")

        best_q_on_r_grid = np.nanmax(q_on_grid, axis=0)
        res_id_for_best_q = np.nanargmax(q_on_grid, axis=0)
        res_for_best_q = self.resolutions[res_id_for_best_q]
        return best_q_on_r_grid, r_grid, res_for_best_q
