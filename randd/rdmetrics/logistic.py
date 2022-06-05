from randd.rdmetrics.base import CodecComparisonBase1d, CodecComparisonBase2d
from randd.estimators import LogLogisticEstimator1d, LogLogisticEstimator2d


class LogLogisticCodecComparison1d(CodecComparisonBase1d):
    def __init__(
        self,
        log_scale: bool = True
    ) -> None:
        super().__init__(
            estimator=LogLogisticEstimator1d(),
            log_scale=log_scale
        )


class LogLogisticCodecComparison2d(CodecComparisonBase2d):
    def __init__(
        self,
        log_scale: bool = True
    ) -> None:
        super().__init__(
            estimator=LogLogisticEstimator2d(),
            log_scale=log_scale
        )
