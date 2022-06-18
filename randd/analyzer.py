import numpy as np
from randd.model import GRD, EGRD
from dataclasses import dataclass
from randd.internal import Estimator
from numpy.typing import ArrayLike, NDArray
from typing import Callable, Optional, Tuple, Type


@dataclass
class Profile:
    name: str
    rq_func: Callable[[ArrayLike], NDArray]
    qr_func: Callable[[ArrayLike], NDArray]
    sampled_r: ArrayLike
    sampled_q: ArrayLike


@dataclass
class Summary:
    codec1: Profile
    codec2: Profile
    r_roi: Tuple[float, float]
    d_roi: Tuple[float, float]
    quality_gain: float
    bitrate_saving: float
    log_scale: bool


class Analyzer:
    def __init__(
        self,
        model1: Type[GRD] = EGRD,
        model2: Type[GRD] = EGRD,
        r_roi: Optional[ArrayLike] = None,
        d_roi: Optional[ArrayLike] = None,
        log_scale: bool = False
    ) -> None:
        self.estimator1 = Estimator(model=model1)
        self.estimator2 = Estimator(model=model2)
        self.r_roi = r_roi
        self.d_roi = d_roi
        self.log_scale = log_scale

    def __call__(
        self,
        r1: ArrayLike, d1: ArrayLike,
        r2: ArrayLike, d2: ArrayLike,
        codec_name1: str = 'codec1',
        codec_name2: str = 'codec2',
    ) -> Tuple[float, float, Summary]:
        r"""_summary_

        Args:
            r1 (ArrayLike): _description_
            d1 (ArrayLike): _description_
            r2 (ArrayLike): _description_
            d2 (ArrayLike): _description_
            codec_name1 (str, optional): _description_. Defaults to 'codec1'.
            codec_name2 (str, optional): _description_. Defaults to 'codec2'.

        Returns:
            Tuple[float, float, Summary]: _description_
        """
        r1, d1, r2, d2 = self._validate_input(r1, d1, r2, d2)

        # get the rate-distortion/distortion-rate functions for each encoder
        rd1, dr1 = self.estimator1(r1, d1)
        rd2, dr2 = self.estimator2(r2, d2)

        # get roi for rate and distortion
        r_roi = self._r_roi(r1, r2)
        d_roi = self._d_roi(d1, d2)

        # calculate average quality difference
        d_gain = self._dist_gain(rd1, rd2, r_roi)
        r_save = self._rate_save(dr1, dr2, d_roi)

        # build the comparison result object
        codec1 = Profile(**{
            'rd_func': rd1, 'dr_func': dr1,
            'sampled_r': r1, 'sampled_d': d1,
            'name': codec_name1,
        })

        codec2 = Profile(**{
            'rd_func': rd2, 'dr_func': dr2,
            'sampled_r': r2, 'sampled_d': d2,
            'name': codec_name2,
        })

        summary = Summary(
            codec1=codec1, codec2=codec2,
            r_roi=r_roi, d_roi=d_roi,
            quality_gain=d_gain,
            bitrate_saving=r_save,
            log_scale=self.log_scale
        )

        return d_gain, r_save, summary

    def _validate_input(
        self,
        r1: ArrayLike, q1: ArrayLike,
        r2: ArrayLike, q2: ArrayLike,
    ) -> None:
        if len(r1) == 0 or len(r2) == 0 or len(q1) == 0 or len(q2) == 0:
            raise ValueError('Invalid input. Input array cannot be empty.')

        if len(r1) != len(q1):
            raise ValueError(f'Input dimension mismatch: r1 length {len(r1)} != q1 length {len(q1)}.')

        if len(r2) != len(q2):
            raise ValueError(f'Input dimension mismatch: r1 length {len(r2)} != q1 length {len(q2)}.')

        r1 = np.asarray(r1)
        q1 = np.asarray(q1)
        r2 = np.asarray(r2)
        q2 = np.asarray(q2)
        return r1, q1, r2, q2

    def _r_roi(self, r1: NDArray, r2: NDArray) -> Tuple[float, float]:
        y1 = r1 if r1.ndim == 1 else r1[:, 0]
        y2 = r2 if r2.ndim == 1 else r2[:, 0]
        min_r = max(np.amin(y1), np.amin(y2))
        max_r = min(np.amax(y1), np.amax(y2))
        if self.r_roi:
            min_r = max(min_r, self.r_roi[0])
            max_r = min(max_r, self.r_roi[1])

        return min_r, max_r

    def _d_roi(self, d1: NDArray, d2: NDArray) -> Tuple[float, float]:
        min_d = max(np.amin(d1), np.amin(d2))
        max_d = min(np.amax(d1), np.amax(d2))
        if self.d_roi:
            min_d = max(min_d, self.d_roi[0])
            max_d = min(max_d, self.d_roi[1])

        return min_d, max_d

    def _dist_gain(
        self,
        f1: Callable[[ArrayLike], NDArray],
        f2: Callable[[ArrayLike], NDArray],
        r_roi: Tuple[float, float],
        num_samples: int = 10000
    ) -> float:
        r"""Calculate average quality gain based on trapz method. Suitable for any funcitons.

        Arguments:
            p1 {Callable[[ArrayLike], NDArray]} -- RD function for reference codec
            p2 {Callable[[ArrayLike], NDArray]} -- RD function for comparing codec
            min_int {float} -- lowest bitrate for integration
            max_int {float} -- highest bitrate for integration

        Keyword Arguments:
            log_scale {bool} -- Indicate whether the integration is taken in log space. (default: {False})
            num_samples {int} -- Number of samples to approximate the integral.
                                 The greater the sample number, the more accurate the estimate. (default: {10000})

        Returns:
            float -- average quality difference
        """
        if r_roi[0] >= r_roi[1] or f1 is None or f2 is None:
            return np.nan

        if self.log_scale:
            x = np.logspace(np.log10(r_roi[0]), np.log10(r_roi[1]), num=num_samples, base=10)
        else:
            x = np.linspace(r_roi[0], r_roi[1], num=num_samples)

        y1 = f1(x)
        y2 = f2(x)
        y = y2 - y1

        valid_mask = np.logical_not(np.isnan(y))
        x: NDArray = x[valid_mask]
        y: NDArray = y[valid_mask]

        # return nan if no valid data for comparison
        if len(y) == 0:
            return np.nan

        int_diff = np.trapz(y, x=x)
        avg_diff = int_diff / (x.max() - x.min())
        return avg_diff

    def _rate_save(
        self,
        f1: Callable[[ArrayLike], NDArray],
        f2: Callable[[ArrayLike], NDArray],
        d_roi: Tuple[float, float],
        num_samples: int = 10000
    ) -> float:
        """Calculate average bitrate-saving based on trapz method. Suitable for any funcitons.

        Arguments:
            p1 {Callable[[ArrayLike], NDArray]} -- quality_rate function for reference codec
            p2 {Callable[[ArrayLike], NDArray]} -- quality_rate function for comparing codec
            min_int {float} -- lowest quality for integration
            max_int {float} -- highest quality for integration

        Keyword Arguments:
            num_samples {int} -- Number of samples to approximate the integral.
                                 The greater the sample number, the more accurate the estimate. (default: {10000})

        Returns:
            float -- average bitrate saving
        """
        if d_roi[0] >= d_roi[1] or f1 is None or f2 is None:
            return np.nan

        samples = np.linspace(d_roi[0], d_roi[1], num=num_samples)

        y1 = f1(samples)
        y2 = f2(samples)
        y = (y2 - y1) / (y1 + 1)

        valid_mask = np.logical_not(np.isnan(y))
        x: NDArray = samples[valid_mask]
        y: NDArray = y[valid_mask]

        # return nan if no valid data for comparison
        if len(y) == 0:
            return np.nan

        int_diff = np.trapz(y, x=x)
        avg_diff = int_diff / (x.max() - x.min()) * 100
        return avg_diff
