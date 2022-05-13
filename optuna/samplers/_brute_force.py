import copy
import decimal

from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._base import BaseSampler
from optuna.trial import TrialState


class BruteForceSampler(BaseSampler):
    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    def sample_independent(self, study, trial, param_name, param_distribution):
        if isinstance(param_distribution, FloatDistribution):
            assert param_distribution.step is not None
            low = decimal.Decimal(str(param_distribution.low))
            high = decimal.Decimal(str(param_distribution.high))
            step = decimal.Decimal(str(param_distribution.step))
            value = low + step
            while value <= high:
                params = copy.deepcopy(trial.params)
                params[param_name] = float(value)
                study.enqueue_trial(params)
                value += step
            return param_distribution.low
        elif isinstance(param_distribution, IntDistribution):
            low = param_distribution.low
            high = param_distribution.high
            step = param_distribution.step
            for value in range(low + step, high + 1, step):
                params = copy.deepcopy(trial.params)
                params[param_name] = value
                study.enqueue_trial(params)
            return low
        elif isinstance(param_distribution, CategoricalDistribution):
            for value in param_distribution.choices[1:]:
                params = copy.deepcopy(trial.params)
                params[param_name] = value
                study.enqueue_trial(params)
            return param_distribution.choices[0]
        else:
            raise RuntimeError()

    def after_trial(self, study, trial, state, values):
        if len(study.get_trials(deepcopy=False, states=(TrialState.WAITING,))) == 0:
            study.stop()
