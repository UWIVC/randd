import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Optional


class GRD:
    def __init__(self, r: NDArray, d: NDArray) -> None:
        pass

    def __call__(self, r: NDArray) -> NDArray:
        pass

    def _group_input(
        r: NDArray, d: Optional[NDArray] = None
    ) -> Dict[Any, Tuple[NDArray, NDArray]]:
        out = {}
        if d is None:
            d = np.empty(r.shape[0])

        if r.ndim == 1:
            key = 'default'
            # rhat, dhat = sort_rd(r=r, d=d)
            out[key] = (r, d)
            return out

        for (row, di) in zip(r, d):
            key, value = tuple(row[1:]), row[0]
            out.setdefault(key, ([], []))
            out[key][0].append(value)
            out[key][1].append(di)

        for key in out:
            r, d = out[key]
            # r, d = sort_rd(r=r, d=d)
            out[key] = (r, d)

        return out
