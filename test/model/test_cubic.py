import pytest
import numpy as np
from randd.model.cubic import LogCubic


r1 = (np.array([ 100.,  300., 1200., 4000.], dtype=float)).reshape(-1, 1)


q1 = np.array([26.6 , 34.71, 44.36, 48.77], dtype=float)


r1_hat = (10**(np.linspace(np.log10(100.), np.log10(4500.), num=6))).reshape(-1, 1)


expected_q1_hat = np.array(
    [26.6, 32.13125945, 37.91688738, 43.16459013, 47.08207405, 48.87704548])


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


res = np.array([2203., 400., 2203.1, 400., 2203., 400.]).reshape(-1, 1)


r2_hat = np.stack(
    (10**(np.linspace(np.log10(100.), np.log10(4500.), num=6)),
     np.array([2203., 400., 2203.1, 400., 2203., 400.])), 
    axis=1
)


expected_q2_hat = np.array(
    [26.6, 34.13067052, np.nan, 34.83893132, 47.08207405, 35.00778963])


class TestLogCubic:
    def test_01(self):
        grd = LogCubic(r1, q1, 'psnr', 1)
        y1_hat = grd(r1_hat)
        assert np.allclose(y1_hat, expected_q1_hat, rtol=1e-4, equal_nan=True)

    def test_02(self):
        grd = LogCubic(r2, q2, 'psnr', 2)
        assert (2203, ) in grd.f and (400, ) in grd.f
        y2_hat = grd(r2_hat)
        assert np.allclose(y2_hat, expected_q2_hat, rtol=1e-4, equal_nan=True)



if __name__ == "__main__":
    pytest.main([__file__])
