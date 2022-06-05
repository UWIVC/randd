from randd.rdmetrics.base import CodecComparisonBase1d, CodecComparisonBase2d
from randd.estimators import LinearEstimator1d, LinearEstimator2d


class LinearCodecComparison1d(CodecComparisonBase1d):
    def __init__(
        self,
        log_scale: bool = False
    ) -> None:
        super().__init__(
            estimator=LinearEstimator1d(),
            log_scale=log_scale
        )


class LinearCodecComparison2d(CodecComparisonBase2d):
    def __init__(
        self,
        log_scale: bool = False
    ) -> None:
        super().__init__(
            estimator=LinearEstimator2d(),
            log_scale=log_scale
        )
