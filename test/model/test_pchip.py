import pytest
import numpy as np
from randd.model.pchip import LogPCHIP


r1 = (np.array([ 100.,  300., 1200., 4000.], dtype=float)).reshape(-1, 1)


q1 = np.array([26.6 , 34.71, 44.36, 48.77], dtype=float)


r1_hat = (10**(np.linspace(np.log10(100.), np.log10(4500.), num=6))).reshape(-1, 1)


expected_q1_hat = np.array(
    [26.6, 32.26732798, 37.90293207, 43.28930106, 46.80303567, 48.99775316])


r2 = np.array(
    [[ 100.,  400.],
     [ 300.,  400.],
     [1200.,  400.],
     [4000.,  400.],
     [ 100., 2203.],
     [ 300., 2203.],
     [1200., 2203.],
     [4000., 2203.]], dtype=float)


q2 = np.array(
    [33.11, 34.41, 34.86, 34.98,
     26.6 , 34.71, 44.36, 48.77], dtype=float)


r2_hat = np.stack(
    (10**(np.linspace(np.log10(100.), np.log10(4500.), num=6)),
     np.array([2203., 400., 2203.1, 400., 2203., 400.])), 
    axis=1
)


expected_q2_hat = np.array(
    [26.6, 34.14542912, np.nan, 34.82522604, 46.80303567, 34.97823547],
    dtype=float
)


class TestLogPCHIP:
    def test_01(self):
        grd = LogPCHIP(r1, q1, 'psnr', 1)
        y1_hat = grd(r1_hat)
        assert np.allclose(y1_hat, expected_q1_hat, rtol=1e-4, equal_nan=True)

    def test_02(self):
        grd = LogPCHIP(r2, q2, 'psnr', 1)
        assert (2203, ) in grd.f and (400, ) in grd.f
        y2_hat = grd(r2_hat)
        assert np.allclose(y2_hat, expected_q2_hat, rtol=1e-4, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__])
