from randd.rdmetrics.base import CodecComparisonBase1d, CodecComparisonBase2d
from randd.estimators import (
    EgrdEstimator, SsimplusEgrdEstimator1d, SsimplusEgrdEstimator2d,
    VmafEgrdEstimator1d, VmafEgrdEstimator2d, PsnrEgrdEstimator1d, PsnrEgrdEstimator2d)
from numpy.typing import NDArray


class EgrdCodecComparison1d(CodecComparisonBase1d):
    def __init__(
        self,
        reps: NDArray,
        f0: NDArray,
        basis: NDArray,
        log_scale: bool = False
    ) -> None:
        super().__init__(
            estimator=EgrdEstimator(
                reps=reps,
                f0=f0,
                basis=basis
            ),
            log_scale=log_scale
        )


class SsimplusEgrdCodecComparison1d(CodecComparisonBase1d):
    def __init__(
        self,
        log_scale: bool = False
    ) -> None:
        super().__init__(
            estimator=SsimplusEgrdEstimator1d(),
            log_scale=log_scale
        )


class VmafEgrdCodecComparison1d(CodecComparisonBase1d):
    def __init__(
        self,
        log_scale: bool = False
    ) -> None:
        super().__init__(
            estimator=VmafEgrdEstimator1d(),
            log_scale=log_scale
        )


class PsnrEgrdCodecComparison1d(CodecComparisonBase1d):
    def __init__(
        self,
        log_scale: bool = False
    ) -> None:
        super().__init__(
            estimator=PsnrEgrdEstimator1d(),
            log_scale=log_scale
        )


class EgrdCodecComparison2d(CodecComparisonBase2d):
    def __init__(
        self,
        reps: NDArray,
        f0: NDArray,
        basis: NDArray,
        log_scale: bool = False
    ) -> None:
        super().__init__(
            estimator=EgrdEstimator(
                reps=reps,
                f0=f0,
                basis=basis
            ),
            log_scale=log_scale
        )


class SsimplusEgrdCodecComparison2d(CodecComparisonBase2d):
    def __init__(
        self,
        log_scale: bool = False
    ) -> None:
        super().__init__(
            estimator=SsimplusEgrdEstimator2d(),
            log_scale=log_scale
        )


class VmafEgrdCodecComparison2d(CodecComparisonBase2d):
    def __init__(
        self,
        log_scale: bool = False
    ) -> None:
        super().__init__(
            estimator=VmafEgrdEstimator2d(),
            log_scale=log_scale
        )


class PsnrEgrdCodecComparison2d(CodecComparisonBase2d):
    def __init__(
        self,
        log_scale: bool = False
    ) -> None:
        super().__init__(
            estimator=PsnrEgrdEstimator2d(),
            log_scale=log_scale
        )
