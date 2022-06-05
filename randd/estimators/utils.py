from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike, NDArray


def sort_r_q(r: ArrayLike, q: ArrayLike) -> Tuple[NDArray, NDArray]:
    r = np.array(r, dtype=float)
    q = np.array(q, dtype=float)
    sort_idx = np.argsort(r)
    r = r[sort_idx]
    q = q[sort_idx]

    return r, q
