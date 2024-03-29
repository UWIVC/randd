import numpy as np
import pytest

from randd.model.linear import Linear

r1 = (np.array(
    [100.,  200.,  300.,  400.,  500.,  600.,  700.,  800.,  900.,
     1000., 1100., 1200., 1300., 1400., 1500., 1600., 1700., 1800.,
     1900., 2000., 2100., 2200., 2300., 2400., 2500., 2600., 2700.,
     2800., 2900., 3000., 3100., 3200., 3300., 3400., 3500., 3600.,
     3700., 3800., 3900., 4000.], dtype=float)).reshape(-1, 1)


q1 = np.array(
    [26.60, 31.40, 34.71, 37.12, 38.90, 40.22, 41.30, 42.14, 42.82,
     43.42, 43.93, 44.36, 44.74, 45.07, 45.39, 45.66, 45.89, 46.11,
     46.32, 46.51, 46.68, 46.84, 46.99, 47.13, 47.28, 47.41, 47.53,
     47.65, 47.77, 47.88, 47.98, 48.08, 48.17, 48.27, 48.36, 48.45,
     48.53, 48.61, 48.70, 48.77], dtype=float)


r1_hat = (10**(np.linspace(np.log10(100.), np.log10(4500.), num=6))).reshape(-1, 1)


expected_q1_hat = np.array(
    [26.6, 31.86713159, 38.16027901, 43.30950451, 46.68271448, 49.12]
)


r2 = np.array([
    [100.,  400.],
    [200.,  400.],
    [300.,  400.],
    [400.,  400.],
    [500.,  400.],
    [600.,  400.],
    [700.,  400.],
    [800.,  400.],
    [900.,  400.],
    [1000.,  400.],
    [1100.,  400.],
    [1200.,  400.],
    [1300.,  400.],
    [1400.,  400.],
    [1500.,  400.],
    [1600.,  400.],
    [1700.,  400.],
    [1800.,  400.],
    [1900.,  400.],
    [2000.,  400.],
    [2100.,  400.],
    [2200.,  400.],
    [2300.,  400.],
    [2400.,  400.],
    [2500.,  400.],
    [2600.,  400.],
    [2700.,  400.],
    [2800.,  400.],
    [2900.,  400.],
    [3000.,  400.],
    [3100.,  400.],
    [3200.,  400.],
    [3300.,  400.],
    [3400.,  400.],
    [3500.,  400.],
    [3600.,  400.],
    [3700.,  400.],
    [3800.,  400.],
    [3900.,  400.],
    [4000.,  400.],
    [100., 2203.],
    [200., 2203.],
    [300., 2203.],
    [400., 2203.],
    [500., 2203.],
    [600., 2203.],
    [700., 2203.],
    [800., 2203.],
    [900., 2203.],
    [1000., 2203.],
    [1100., 2203.],
    [1200., 2203.],
    [1300., 2203.],
    [1400., 2203.],
    [1500., 2203.],
    [1600., 2203.],
    [1700., 2203.],
    [1800., 2203.],
    [1900., 2203.],
    [2000., 2203.],
    [2100., 2203.],
    [2200., 2203.],
    [2300., 2203.],
    [2400., 2203.],
    [2500., 2203.],
    [2600., 2203.],
    [2700., 2203.],
    [2800., 2203.],
    [2900., 2203.],
    [3000., 2203.],
    [3100., 2203.],
    [3200., 2203.],
    [3300., 2203.],
    [3400., 2203.],
    [3500., 2203.],
    [3600., 2203.],
    [3700., 2203.],
    [3800., 2203.],
    [3900., 2203.],
    [4000., 2203.]
], dtype=float)


q2 = np.array([
    33.11, 34.11, 34.41, 34.56, 34.65, 34.71, 34.75, 34.79, 34.81,
    34.83, 34.85, 34.86, 34.87, 34.88, 34.89, 34.90, 34.91, 34.92,
    34.92, 34.93, 34.93, 34.94, 34.94, 34.94, 34.95, 34.95, 34.95,
    34.96, 34.96, 34.96, 34.97, 34.97, 34.97, 34.97, 34.97, 34.98,
    34.98, 34.98, 34.98, 34.98, 26.60, 31.40, 34.71, 37.12, 38.90,
    40.22, 41.30, 42.14, 42.82, 43.42, 43.93, 44.36, 44.74, 45.07,
    45.39, 45.66, 45.89, 46.11, 46.32, 46.51, 46.68, 46.84, 46.99,
    47.13, 47.28, 47.41, 47.53, 47.65, 47.77, 47.88, 47.98, 48.08,
    48.17, 48.27, 48.36, 48.45, 48.53, 48.61, 48.70, 48.77
], dtype=float)


r2_hat = np.stack(
    (10**(np.linspace(np.log10(100.), np.log10(4500.), num=6)),
     np.array([2203., 400., 2203.1, 400., 2203., 400.])),
    axis=1
)


expected_q2_hat = np.array(
    [26.6, 34.15233821, np.nan, 34.82631682, 46.68271448, 34.98], dtype=float
)


class TestLinear:
    def test_01(self):
        grd = Linear(r1, q1, 'psnr', 1)
        y1_hat = grd(r1_hat)
        assert np.allclose(y1_hat, expected_q1_hat, rtol=1e-4, equal_nan=True)

    def test_02(self):
        grd = Linear(r2, q2, 'psnr', 2)
        assert (2203, ) in grd.f and (400, ) in grd.f
        y2_hat = grd(r2_hat)
        assert np.allclose(y2_hat, expected_q2_hat, rtol=1e-4, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__])
