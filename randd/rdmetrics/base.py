from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Callable, Optional, Tuple, Union


@dataclass
class CodecRDPerformance:
    name: str
    rq_func: Callable[[ArrayLike], NDArray]
    qr_func: Callable[[ArrayLike], NDArray]
    sampled_r: ArrayLike
    sampled_q: ArrayLike


@dataclass
class CodecComparisonResult:
    codec1: CodecRDPerformance
    codec2: CodecRDPerformance
    r_range_to_compare: Tuple[float, float]
    q_range_to_compare: Tuple[float, float]
    quality_gain: float
    bitrate_saving: float
    log_scale: bool


class CodecComparisonBase(metaclass=ABCMeta):
    def __init__(
        self,
        estimator: Callable[[ArrayLike, ArrayLike], Tuple[Callable[[ArrayLike], NDArray], Callable[[ArrayLike], NDArray]]],
        estimator2: Optional[Callable[[ArrayLike, ArrayLike],
                             Tuple[Callable[[ArrayLike], NDArray], Callable[[ArrayLike], NDArray]]]] = None,
        log_scale: bool = False
    ) -> None:
        self.estimator1 = estimator

        if estimator2 is None:
            self.estimator2 = self.estimator1
        else:
            self.estimator2 = estimator2

        self.log_scale = log_scale

    def estimate_quality_gain(
        self,
        p1: Callable[[ArrayLike], NDArray],
        p2: Callable[[ArrayLike], NDArray],
        min_r: float, max_r: float,
        num_samples: int = 10000
    ) -> float:
        """Calculate average quality gain based on trapz method. Suitable for any funcitons.

        Arguments:
            p1 {Callable[[ArrayLike], NDArray]} -- rate_quality function for reference codec
            p2 {Callable[[ArrayLike], NDArray]} -- rate_quality function for comparing codec
            min_int {float} -- lowest bitrate for integration
            max_int {float} -- highest bitrate for integration

        Keyword Arguments:
            log_scale {bool} -- Indicate whether the integration is taken in log space. (default: {False})
            num_samples {int} -- Number of samples to approximate the integral.
                                 The greater the sample number, the more accurate the estimate. (default: {10000})

        Returns:
            float -- average quality difference
        """
        if min_r >= max_r or p1 is None or p2 is None:
            return np.nan
        if self.log_scale:
            min_r, max_r = np.log10(min_r), np.log10(max_r)
            x = np.linspace(min_r, max_r, num=num_samples)
            samples = np.power(10.0, x)
        else:
            x = np.linspace(min_r, max_r, num=num_samples)
            samples = x

        y1 = p1(samples)
        y2 = p2(samples)
        y = y2 - y1

        valid_mask = np.logical_not(np.isnan(y))
        x: NDArray = x[valid_mask]
        y: NDArray = y[valid_mask]
        if len(y) > 1:
            int_diff = np.trapz(y, x=x)

            # find avg quality difference
            avg_diff = int_diff/(x.max() - x.min())
        else:
            avg_diff = np.nan

        return avg_diff

    def estimate_bitrate_saving(
        self,
        p1: Callable[[ArrayLike], NDArray],
        p2: Callable[[ArrayLike], NDArray],
        min_q: float, max_q: float,
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
        if min_q >= max_q or p1 is None or p2 is None:
            return np.nan
        samples = np.linspace(min_q, max_q, num=num_samples)
        y1 = p1(samples)
        y2 = p2(samples)

        y = (y2 - y1) / (y1 + 1)
        valid_mask = np.logical_not(np.isnan(y))
        x: NDArray = samples[valid_mask]
        y: NDArray = y[valid_mask]

        if len(y) > 1:
            int_diff = np.trapz(y, x=x)
            avg_diff = int_diff / (x.max() - x.min()) * 100
        else:
            avg_diff = np.nan

        return avg_diff

    @abstractmethod
    def _get_default_range_r(
        self,
        reps1: ArrayLike,
        reps2: ArrayLike
    ) -> Tuple[float, float]:
        """Determine the default comparing bitrate range

        This function should be different for 1D and 2D
        cases.

        Args:
            reps1 (ArrayLike): representations of sampled
                data points for the first decoder.
            reps2 (ArrayLike): representations of sampled
                data points for the second decoder.

        Returns:
            Tuple[float, float]: the obtained default
                comparing bitrate range
        """
        pass

    def __call__(
        self,
        reps1: ArrayLike, q1: ArrayLike,
        reps2: ArrayLike, q2: ArrayLike,
        range_r: Optional[Tuple[float, float]] = None,
        range_q: Optional[Tuple[float, float]] = None,
        codec_name1: str = 'codec1',
        codec_name2: str = 'codec2',
        log_scale: bool = False,
        detailed_result: bool = False
    ) -> Union[Tuple[float, float], Tuple[float, float, CodecComparisonResult]]:

        if len(reps1) == 0 or len(reps2) == 0 or len(reps1) != len(q1) or len(reps2) != len(q2):
            raise ValueError('Video codecs cannot be compared with incomplete data!')

        # calculate quality difference

        # get the rate-quality/quality-rate functions for each encoder
        rq_func1, qr_func1 = self.estimator1(reps1, q1)
        rq_func2, qr_func2 = self.estimator2(reps2, q2)

        # get bitrate integration range
        if range_r is None:
            min_r, max_r = self._get_default_range_r(reps1, reps2)
        else:
            min_r, max_r = range_r

        # calculate average quality difference
        quality_gain = self.estimate_quality_gain(rq_func1, rq_func2, min_r, max_r)

        # get valid quality range for comparison
        min_q = np.nanmax([rq_func1(min_r), rq_func2(min_r)])
        max_q = np.nanmin([rq_func1(max_r), rq_func2(max_r)])
        if range_q is not None:
            min_q, max_q = np.nanmax([range_q[0], min_q]), np.nanmin([range_q[1], max_q])

        # calculate average bitrate saving
        bitrate_saving = self.estimate_bitrate_saving(qr_func1, qr_func2, min_q, max_q)

        if not detailed_result:
            return quality_gain, bitrate_saving

        # build the comparison result object
        codec1 = CodecRDPerformance(**{
            'name': codec_name1,
            'rq_func': rq_func1,
            'qr_func': qr_func1,
            'sampled_r': reps1,
            'sampled_q': q1,
        })
        codec2 = CodecRDPerformance(**{
            'name': codec_name2,
            'rq_func': rq_func2,
            'qr_func': qr_func2,
            'sampled_r': reps2,
            'sampled_q': q2,
        })

        comparison_result = CodecComparisonResult(
            codec1=codec1,
            codec2=codec2,
            r_range_to_compare=(min_r, max_r),
            q_range_to_compare=(min_q, max_q),
            quality_gain=quality_gain,
            bitrate_saving=bitrate_saving,
            log_scale=log_scale
        )

        return quality_gain, bitrate_saving, comparison_result


class CodecComparisonBase1d(CodecComparisonBase):
    def _get_default_range_r(self, reps1: ArrayLike, reps2: ArrayLike) -> Tuple[float, float]:
        """Determine the default valid comparing bitrate range for the 1D case.

        In this case, each entry of reps1 or reps2 represents
        the bitrate of a data point. Therefore, the valid comparing
        range for most video codec comparison algorithm is the intersection of
        the bitrate ranges covered by the data points for the
        two codecs.

        Args:
            reps1 (ArrayLike): A 1D vector containing datapoint
                bitrates.
            reps2 (ArrayLike): A 1D vector containing datapoint
                bitrates.

        Returns:
            Tuple[float, float]: default valid bitrate range
        """
        min_r = max(np.amin(reps1), np.amin(reps2))
        max_r = min(np.amax(reps1), np.amax(reps2))
        return min_r, max_r


class CodecComparisonBase2d(CodecComparisonBase):
    def _get_default_range_r(self, reps1: ArrayLike, reps2: ArrayLike) -> Tuple[float, float]:
        """Determine the default valid comparing bitrate range for the 2D case.

        In this case, each row of reps1 or reps2 represents
        the (bitrate, resolution) of a representation.
        Therefore, the valid comparing range for most video codec comparison
        algorithm is the intersection of the bitrate
        ranges covered by the data points for the two codecs.

        Args:
            reps1 (ArrayLike): A Nx2 array. Each row contains a
                datapoint's (bitrate, resolution).
            reps2 (ArrayLike): A Nx2 array. Each row contains a
                datapoint's (bitrate, resolution).

        Returns:
            Tuple[float, float]: default valid bitrate range
        """
        min_r = max(np.amin(reps1[:, 0]), np.amin(reps2[:, 0]))
        max_r = min(np.amax(reps1[:, 0]), np.amax(reps2[:, 0]))
        return min_r, max_r
