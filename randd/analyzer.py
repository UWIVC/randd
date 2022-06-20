import numpy as np
from randd.model import GRD, EGRD
from dataclasses import dataclass
from randd.internal import Estimator
from numpy.typing import ArrayLike, NDArray
from typing import Callable, Optional, Tuple, Type


@dataclass
class Profile:
    r"""A dataclass to store output information.

    The class stores data to characterize the RD performance of a given codec.

    Args:
        name (str): name of the codec
        rd_func (Callable[[ArrayLike], NDArray]): Estimated RD function
        dr_func (Callable[[ArrayLike], NDArray]): Estimated DR function
        sampled_r (ArrayLike): Input encoding representations
        sampled_d (ArrayLike): Input distortion levels
    """
    name: str
    rd_func: Callable[[ArrayLike], NDArray]
    dr_func: Callable[[ArrayLike], NDArray]
    sampled_r: ArrayLike
    sampled_d: ArrayLike


@dataclass
class Summary:
    r"""A dataclass to store output information.

    The class contains the codec comparison results.

    Args:
        codec1 (Profile): Output profile that describes the RD characteristics of codec1.
        codec2 (Profile): Output profile that describes the RD characteristics of codec2.
        r_roi (Tuple[float, float]): Bitrate region used for codec comparison.
        d_roi (Tuple[float, float]): Distortion region used for codec comparison.
        quality_gain (float): Quality gain of codec2 compared to codec1.
        bitrate_saving (float): Bitrate saving of codec2 over codec1.
        log_scale (bool): Whether the comparison is performed in a log scale.
    """
    codec1: Profile
    codec2: Profile
    r_roi: Tuple[float, float]
    d_roi: Tuple[float, float]
    quality_gain: float
    bitrate_saving: float
    log_scale: bool


class Analyzer:
    r"""Rate-distortion analyzer.

    The core class of R&D. The class firstly estimates the RD curve from sparse samples on
    a RD function for each codec, and then computes the bitrate saving and quality gain
    based on the estimated curves.

    Args:
        model1 (Type[GRD], optional): The RD function model for the codec 1.
            Available optionals include :class:`randd.model.Linear`, :class:`randd.model.LogCubic`,
            :class:`randd.model.LogPCHIP`, and :class:`randd.model.EGRD`. Defaults to :class:`randd.model.EGRD`.
        model2 (Type[GRD], optional): The RD function model for the codec 2.
            Defaults to EGRD.
        r_roi (Optional[ArrayLike], optional): Bitrate region of interest for the quality gain computation.
            r_roi should be a tuple of float (r_min, r_max). If not specified, the analyzer will infer the range
            from the given samples. Defaults to ``None``.
        d_roi (Optional[ArrayLike], optional): Distortion region of interest for the bitrate saving computation.
            d_roi should be a tuple of (d_min, d_max). If not specified. the analyzer will infer the range from
            the given samples. Defaults to ``None``.
        ndim (int, optional): Number of input dimensions.
            R&D supports multi-dimensional generalized rate-distortion function estimation.
            For ndim > 1, the analyzer will first estimate the generalized rate-distortion surface, and then
            produce a rate-distortion curve on the convex hull. To be specific, for the representations with the same
            bitrate, we take the representation with the best quality for further analysis.
            Defaults to ``1``.
        d_measure (str, optional): The name of distortion measure. Defaults to ``psnr``.
        log_scale (bool, optional): Whether to perform the codec comparison in a logscale. Defaults to ``False``.
    """
    def __init__(
        self,
        model1: Type[GRD] = EGRD,
        model2: Type[GRD] = EGRD,
        r_roi: Optional[Tuple[float, float]] = None,
        d_roi: Optional[Tuple[float, float]] = None,
        ndim: int = 1,
        d_measure: str = 'psnr',
        log_scale: bool = False
    ) -> None:
        self.estimator1 = Estimator(model=model1, d_measure=d_measure, ndim=ndim)
        self.estimator2 = Estimator(model=model2, d_measure=d_measure, ndim=ndim)
        self.r_roi = r_roi
        self.d_roi = d_roi
        self.log_scale = log_scale

    def __call__(
        self,
        r1: ArrayLike, d1: ArrayLike,
        r2: ArrayLike, d2: ArrayLike,
        codec1: str = 'codec1',
        codec2: str = 'codec2',
    ) -> Tuple[float, float, Summary]:
        r"""Run codec comparison.

        Args:
            r1 (ArrayLike): The representations of the first encoder.
                For the most basic use case, r1 is a 1D array containing the encoding bitrates of a content
                for encoder 1. e.g., r1 = [100, 350, 754, 1890]
                For advance use case, r1 can be a 2D array of size NxD, where N is the number of encoding
                representations, and D is the number of encoding attributes. The first encoding attributes
                has to be the encoding bitrate, where as the rest of the dimension can be spatial resolution,
                frame rate, bit depth, and etc.
            d1 (ArrayLike): The corresponding distortions of the first encoder.
                A 1D array containing the distortion measures for r1.
            r2 (ArrayLike): The representations of the second encoder.
                Similar to r1, but for a competing encoder.
            d2 (ArrayLike): The corresponding distortions of the second encoder.
                A 1D array containing the distortion measures for r2.
            codec1 (str, optional): name of the codec 1.
                Defaults to `codec1`.
            codec2 (str, optional): name of the codec 2.
                Defaults to `codec2`.

        Returns:
            Tuple[float, float, Summary]: quality gain, bitrate saving, :class:`randd.analyzer.Summary`
        """
        r1, d1, r2, d2 = self._validate_input(r1, d1, r2, d2)

        # get the rate-distortion/distortion-rate functions for each encoder
        rd1, dr1 = self.estimator1(r1, d1)
        rd2, dr2 = self.estimator2(r2, d2)

        # get roi for rate and distortion
        r_roi = self._r_roi(r1, r2)
        d_roi = self._d_roi(rd1, rd2, r_roi)

        # calculate average quality difference
        d_gain = self._dist_gain(rd1, rd2, r_roi)
        r_save = self._rate_save(dr1, dr2, d_roi)

        # build the comparison result object
        codec1 = Profile(**{
            'rd_func': rd1, 'dr_func': dr1,
            'sampled_r': r1, 'sampled_d': d1,
            'name': codec1,
        })

        codec2 = Profile(**{
            'rd_func': rd2, 'dr_func': dr2,
            'sampled_r': r2, 'sampled_d': d2,
            'name': codec2,
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
        r1: ArrayLike, d1: ArrayLike,
        r2: ArrayLike, d2: ArrayLike,
    ) -> None:
        if len(r1) == 0 or len(r2) == 0 or len(d1) == 0 or len(d2) == 0:
            raise ValueError('Invalid input. Input array cannot be empty.')

        if len(r1) != len(d1):
            raise ValueError(f'Input dimension mismatch: r1 length {len(r1)} != d1 length {len(d1)}.')

        if len(r2) != len(d2):
            raise ValueError(f'Input dimension mismatch: r2 length {len(r2)} != d2 length {len(d2)}.')

        r1 = np.asarray(r1)
        d1 = np.asarray(d1)
        r2 = np.asarray(r2)
        d2 = np.asarray(d2)

        d1 = np.squeeze(d1)
        d2 = np.squeeze(d2)

        if r1.ndim > 2:
            raise ValueError('Invalid r1. The dimension of r1 cannot be greater than 2.')

        if r2.ndim > 2:
            raise ValueError('Invalid r2. The dimension of r2 cannot be greater than 2.')

        if d1.ndim > 1:
            raise ValueError('Invalid d1. d1 has to be a 1-D array/list.')

        if d2.ndim > 1:
            raise ValueError('Invalid d2. d2 has to be a 1-D array/list.')

        if r1.ndim == 1:
            r1 = np.expand_dims(r1, axis=1)

        if r2.ndim == 1:
            r2 = np.expand_dims(r2, axis=1)

        return r1, d1, r2, d2

    def _r_roi(self, r1: NDArray, r2: NDArray) -> Tuple[float, float]:
        if self.r_roi:
            return tuple(self.r_roi)
        else:
            y1 = r1 if r1.ndim == 1 else r1[:, 0]
            y2 = r2 if r2.ndim == 1 else r2[:, 0]
            min_r = max(np.amin(y1), np.amin(y2))
            max_r = min(np.amax(y1), np.amax(y2))

        return min_r, max_r

    def _d_roi(
        self,
        rd1: Callable[[ArrayLike], NDArray],
        rd2: Callable[[ArrayLike], NDArray],
        r_roi: Tuple[float, float]
    ) -> Tuple[float, float]:
        if self.d_roi:
            return tuple(self.d_roi)
        else:
            min_r, max_r = r_roi
            min_d = np.nanmax([rd1(min_r), rd2(min_r)])
            max_d = np.nanmax([rd1(max_r), rd2(min_r)])

        return min_d, max_d

    def _dist_gain(
        self,
        f1: Callable[[ArrayLike], NDArray],
        f2: Callable[[ArrayLike], NDArray],
        r_roi: Tuple[float, float],
        num_samples: int = 10000
    ) -> float:
        r"""Calculate average quality gain based on trapz method. Suitable for any funcitons.

        Args:
            f1 (Callable[[ArrayLike], NDArray]): RD function for reference codec.
            f2 (Callable[[ArrayLike], NDArray]): RD function for comparing codec.
            r_roi (Tuple[float, float]): Bitrate region of interest to compute quality gain.
            num_samples (int): Number of samples used in the computation of intergral.

        Returns:
            float: average quality difference
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
        r"""Calculate average bitrate-saving based on trapz method. Suitable for any funcitons.

        Args:
            f1 (Callable[[ArrayLike], NDArray]): DR function for reference codec.
            f2 (Callable[[ArrayLike], NDArray]): DR function for comparing codec.
            d_roi (Tuple[float, float]): Distortion region of interest to compute bitrate saving.
            num_samples (int): Number of samples used in the computation of intergral.

        Returns:
            float: average bitrate saving
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
