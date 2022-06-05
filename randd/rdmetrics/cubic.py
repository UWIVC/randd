from randd.rdmetrics.base import CodecComparisonBase1d, CodecComparisonBase2d
from randd.estimators import LogCubicEstimator1d, LogCubicEstimator2d


class LogCubicCodecComparison1d(CodecComparisonBase1d):
    def __init__(
        self,
        log_scale: bool = True
    ) -> None:
        super().__init__(
            estimator=LogCubicEstimator1d(),
            log_scale=log_scale
        )


class LogCubicCodecComparison2d(CodecComparisonBase2d):
    def __init__(
        self,
        log_scale: bool = True
    ) -> None:
        super().__init__(
            estimator=LogCubicEstimator2d(),
            log_scale=log_scale
        )
