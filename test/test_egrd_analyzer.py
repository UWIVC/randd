import numpy as np
import pytest

from randd.analyzer import Analyzer


class TestEgrdAnalyzer:
    def test_01(self):
        # test 1d case with psnr
        r1 = np.array([100., 400., 1500., 6000.])
        d1 = np.array([23.83, 25.89, 31.28, 38.22])
        r2 = np.array([100., 400., 1500., 6000.])
        d2 = np.array([16.25, 29.75, 33.52, 39.12])
        expected_result = np.array([1.4148574071866933, -33.16950901624122])
        analyzer = Analyzer(d_measure='psnr')
        quality_gain, bitrate_saving, _ = analyzer(r1, d1, r2, d2, codec1='h264', codec2='vp9')
        assert np.allclose((quality_gain, bitrate_saving), expected_result, rtol=1e-4, equal_nan=True)

    def test_02(self):
        # test 2d case with psnr
        r1 = np.array([[6000.,  640.],
                       [700.,  865.],
                       [200., 2203.],
                       [2500., 2203.]])
        d1 = np.array([31.28, 30.26, 18.89, 34.4])
        r2 = np.array([[6000.,  640.],
                       [700.,  865.],
                       [200., 2203.],
                       [2500., 2203.]])
        d2 = np.array([31.41, 31.07, 21.68, 36.52])
        expected_result = np.array([1.545422821252965, -31.43398905598459])

        analyzer = Analyzer(d_measure='psnr', ndim=2, r_roi=[100, 3000])
        quality_gain, bitrate_saving, _ = analyzer(r1, d1, r2, d2)
        assert np.allclose((quality_gain, bitrate_saving), expected_result, rtol=1e-4, equal_nan=True)

    def test_03(self):
        # test 2d case with vmaf
        r1 = np.array([[2400.,  400.], [1700.,  865.], [300., 1469.], [2100., 2203.]])
        d1 = np.array([28.39, 71.87, 26.98, 89.35])
        r2 = np.array([[2400.,  400.], [1700.,  865.], [300., 1469.], [2100., 2203.]])
        d2 = np.array([28.86, 75.03, 54.35, 96.49])
        expected_result = np.array([11.364702006653086, -45.76491833838688])
        analyzer = Analyzer(d_measure='vmaf', ndim=2, r_roi=[100, 3000])
        quality_gain, bitrate_saving, _ = analyzer(r1, d1, r2, d2, codec1='h264', codec2='vp9')
        assert np.allclose((quality_gain, bitrate_saving), expected_result, rtol=1e-4, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__])
