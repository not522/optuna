from typing import TYPE_CHECKING

from optuna.pruners._base import BasePruner
from optuna.pruners._hyperband import HyperbandPruner
from optuna.pruners._median import MedianPruner
from optuna.pruners._nop import NopPruner
from optuna.pruners._patient import PatientPruner
from optuna.pruners._percentile import PercentilePruner
from optuna.pruners._successive_halving import SuccessiveHalvingPruner
from optuna.pruners._threshold import ThresholdPruner
from optuna.pruners._wilcoxon import WilcoxonPruner


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


__all__ = [
    "BasePruner",
    "HyperbandPruner",
    "MedianPruner",
    "NopPruner",
    "PatientPruner",
    "PercentilePruner",
    "SuccessiveHalvingPruner",
    "ThresholdPruner",
    "WilcoxonPruner",
]


def _filter_study(study: "Study", trial: "FrozenTrial") -> "Study":
    return study
