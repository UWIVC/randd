import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Optional


class GRD:
    def __init__(self, r: NDArray, d: NDArray) -> None:
        pass

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
        self, r_min: Optional[float] = None, r_max: Optional[float] = None, n_samples: int = 1001
    ) -> Tuple[NDArray, NDArray, NDArray]:
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
