import pytest
import numpy as np
from randd.model.egrd import EGRD
from randd.internal import Estimator


#=============================test_01 data==========================================
r1 = (np.array([ 100.,  300., 1200., 4000.], dtype=float)).reshape(-1, 1)


psnr1 = np.array([26.6 , 34.71, 44.36, 48.77], dtype=float)


r1_hat = (10**(np.linspace(np.log10(100.), np.log10(4500.), num=6))).reshape(-1, 1)


expected_psnr1_hat = np.array(
    [26.59999988, 32.04926197, 38.20641348, 43.32343616, 46.90754677, 49.07139174])


#=============================test_02 data==========================================
r2 = np.array(
    [[4000.,  400.],
     [ 200., 2203.],
     [1600., 2203.],
     [3600., 2203.]], dtype=float)


psnr2 = np.array(
    [34.98, 31.4, 45.66, 48.45], dtype=float)


r2_hat = np.stack(
    (10**(np.linspace(np.log10(100.), np.log10(4500.), num=6)),
     np.array([2203., 400., 2203.1, 400., 2203., 400.])), 
    axis=1
)


expected_psnr2_hat = np.array(
    [[23.28615482, 33.67312648, np.nan, 34.81788455, 46.41863002, 35.01058209],
     [33.56969564, 37.9133305, 42.47100336, 46.25736341, 48.84103507, 50.4593277]])


class TestLogCubic:
    def test_01(self):
        # test 1d fit with PSNR
        grd = EGRD(r1, psnr1, 'psnr', 1)
        y1_hat = grd(r1_hat)
        assert np.allclose(y1_hat, expected_psnr1_hat, rtol=1e-4, equal_nan=True)

        # test 1d estimator
        estim = Estimator(
            EGRD,
            ndim=1,
            d_measure='psnr',
            r_roi=(100, 4500)
        )

        rd_func, dr_func = estim(r1, psnr1)
        y1_hat = rd_func(r1_hat.flatten())
        assert np.allclose(y1_hat, expected_psnr1_hat, rtol=1e-4, equal_nan=True)

        bitrate1_hat = dr_func(expected_psnr1_hat)
        assert np.allclose(bitrate1_hat, r1_hat.flatten(), rtol=1e-4, equal_nan=True)

    def test_02(self):
        # test 2d fit with PSNR
        grd = EGRD(r2, psnr2, 'psnr', 2)
        for res in (2203, 400, 1469, 480, 865, 640):
            assert (res, ) in grd.f
        y2_hat = grd(r2_hat)
        assert np.allclose(y2_hat, expected_psnr2_hat[0], rtol=1e-4, equal_nan=True)

        # test 2d estimator
        estim = Estimator(
            EGRD,
            ndim=2,
            d_measure='psnr',
            r_roi=(100, 4500)
        )
        rd_func, dr_func = estim(r2, psnr2)
        y2_hat = rd_func(r2_hat[:, 0])
        assert np.allclose(y2_hat, expected_psnr2_hat[1], rtol=1e-4, equal_nan=True)

        bitrate2_hat = dr_func(expected_psnr2_hat[1])
        assert np.allclose(bitrate2_hat, r2_hat[:, 0], rtol=1e-4, equal_nan=True)



if __name__ == "__main__":
    pytest.main([__file__])
