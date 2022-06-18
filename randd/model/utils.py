import numpy as np
from typing import Tuple, Dict, Any
from numpy.typing import ArrayLike, NDArray


def sort_rd(r: ArrayLike, d: ArrayLike) -> Tuple[NDArray, NDArray]:
    r = np.array(r, dtype=float)
    d = np.array(d, dtype=float)
    sort_idx = np.argsort(r)
    r = r[sort_idx]
    d = d[sort_idx]

    return r, d


def group_input(r: NDArray, d: NDArray) -> Dict[Any, Tuple[NDArray, NDArray]]:
    out = {}
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
