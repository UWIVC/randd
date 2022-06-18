import pytest
import numpy as np
import randd as rd


expected_result = np.array(
    [[2.003466, -44.592343], [2.190101, -45.964366]]
)


class TestAnalyzer:
    def test_01(self):
        # test for default mode
        # prepare input
        r1 = np.array([100., 300., 1200., 4000.])
        r2 = np.array([100., 300., 1200., 4000.])
        d1 = np.array([34.31, 83.44, 96.74, 99.])
        d2 = np.array([75.56, 91.98, 97.89, 98.96])
        # initialize the analyzer
        roi = np.array([100., 4000.])
        analyzer = rd.Analyzer(roi=roi)
        # perform the RD analysis
        rd_summary = analyzer(r1, d1, r2, d2)
        assert np.allclose(rd_summary, expected_result[0], rtol=1e-4, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__])
