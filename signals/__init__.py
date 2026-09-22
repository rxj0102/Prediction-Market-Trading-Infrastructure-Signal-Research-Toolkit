"""Module 2: mispricing signal research and forecast calibration."""

from signals.calibration import (
    BrierDecomposition,
    CalibrationBin,
    brier_decomposition,
    brier_score,
    calibration_curve,
    expected_calibration_error,
    log_loss,
)
from signals.model import LogisticModel
from signals.research import (
    MarketSignal,
    SignalConfig,
    SignalModel,
    SignalReport,
    evaluate,
    split_markets,
)

__all__ = [
    "BrierDecomposition",
    "CalibrationBin",
    "LogisticModel",
    "MarketSignal",
    "SignalConfig",
    "SignalModel",
    "SignalReport",
    "brier_decomposition",
    "brier_score",
    "calibration_curve",
    "evaluate",
    "expected_calibration_error",
    "log_loss",
    "split_markets",
]
