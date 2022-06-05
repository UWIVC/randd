from randd.rdmetrics.base import CodecComparisonBase1d, CodecComparisonBase2d
from randd.estimators import LogPchipEstimator1d, LogPchipEstimator2d


class LogPchipCodecComparison1d(CodecComparisonBase1d):
    def __init__(
        self,
        log_scale: bool = True
    ) -> None:
        super().__init__(
            estimator=LogPchipEstimator1d(),
            log_scale=log_scale
        )


class LogPchipCodecComparison2d(CodecComparisonBase2d):
    def __init__(
        self,
        log_scale: bool = True
    ) -> None:
        super().__init__(
            estimator=LogPchipEstimator2d(),
            log_scale=log_scale
        )
