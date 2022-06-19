import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike, NDArray


def sort_rd(r: ArrayLike, d: ArrayLike) -> Tuple[NDArray, NDArray]:
    r = np.array(r, dtype=float)
    d = np.array(d, dtype=float)
    sort_idx = np.argsort(r)
    r = r[sort_idx]
    d = d[sort_idx]

    return r, d
