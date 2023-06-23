from __future__ import annotations

import abc
from typing import List
from typing import NamedTuple

import numpy as np

from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe import _truncnorm


class _BatchedDistributions(abc.ABC):
    @abc.abstractmethod
    def sample(
        self, rng: np.random.RandomState, batch_size: int, active_indices: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _BatchedCategoricalDistributions(_BatchedDistributions):
    def __init__(self, weights: np.ndarray) -> None:
        self._weights = weights

    def sample(
        self, rng: np.random.RandomState, batch_size: int, active_indices: np.ndarray
    ) -> np.ndarray:
        active_weights = self._weights[active_indices, :]
        rnd_quantile = rng.rand(batch_size)
        cum_probs = np.cumsum(active_weights, axis=-1)
        assert np.isclose(cum_probs[:, -1], 1).all()
        cum_probs[:, -1] = 1  # Avoid numerical errors.
        return np.sum(cum_probs < rnd_quantile[:, None], axis=-1)

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        return np.log(
            np.take_along_axis(
                self._weights[None, :, :], x[:, None, None].astype(np.int64), axis=-1
            )
        )[:, :, 0]


class _BatchedTruncNormDistributions(_BatchedDistributions):
    def __init__(self, mu: np.ndarray, sigma: np.ndarray, low: float, high: float, log: bool, search_space: FloatDistribution | IntDistribution) -> None:
        self._mu = mu
        self._sigma = sigma
        self._low = low  # Currently, low and high do not change per trial.
        self._high = high
        self._log = log
        self._search_space = search_space

    def sample(
        self, rng: np.random.RandomState, batch_size: int, active_indices: np.ndarray
    ) -> np.ndarray:
        active_mus = self._mu[active_indices]
        active_sigmas = self._sigma[active_indices]
        samples = _truncnorm.rvs(
            a=(self._low - active_mus) / active_sigmas,
            b=(self._high - active_mus) / active_sigmas,
            loc=active_mus,
            scale=active_sigmas,
            random_state=rng,
        )

        if self._log:
            samples = np.exp(samples)

        if isinstance(self._search_space, IntDistribution):
            # TODO(contramundum53): Remove this line after fixing log-Int hack.
            samples = np.clip(
                self._search_space.low + np.round((samples - self._search_space.low) / self._search_space.step) * self._search_space.step,
                self._search_space.low,
                self._search_space.high,
            )

        return samples

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        if self._log:
            x = np.log(x)

        return _truncnorm.logpdf(
            x=x[:, None],
            a=(self._low - self._mu[None, :]) / self._sigma[None, :],
            b=(self._high - self._mu[None, :]) / self._sigma[None, :],
            loc=self._mu[None, :],
            scale=self._sigma[None, :],
        )


class _BatchedDiscreteTruncNormDistributions(_BatchedDistributions):
    def __init__(
            self, mu: np.ndarray, sigma: np.ndarray, low: float, high: float, step: float, log: bool
    ) -> None:
        self._mu = mu
        self._sigma = sigma
        self._low = low  # Currently, low, high and step do not change per trial.
        self._high = high
        self._step = step
        self._log = log

    def sample(
        self, rng: np.random.RandomState, batch_size: int, active_indices: np.ndarray
    ) -> np.ndarray:
        active_mus = self._mu[active_indices]
        active_sigmas = self._sigma[active_indices]
        samples = _truncnorm.rvs(
            a=(self._low - self._step / 2 - active_mus) / active_sigmas,
            b=(self._high + self._step / 2 - active_mus) / active_sigmas,
            loc=active_mus,
            scale=active_sigmas,
            random_state=rng,
        )
        samples = np.clip(
            self._low + np.round((samples - self._low) / self._step) * self._step,
            self._low,
            self._high,
        )

        if self._log:
            samples = np.exp(samples)

        return samples

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        if self._log:
            x = np.log(x)

        lower_limit = self._low - self._step / 2
        upper_limit = self._high + self._step / 2
        x_lower = np.maximum(x - self._step / 2, lower_limit)
        x_upper = np.minimum(x + self._step / 2, upper_limit)
        log_gauss_mass = _truncnorm._log_gauss_mass(
            (x_lower[:, None] - self._mu[None, :]) / self._sigma[None, :],
            (x_upper[:, None] - self._mu[None, :]) / self._sigma[None, :],
        )
        log_p_accept = _truncnorm._log_gauss_mass(
            (self._low - self._step / 2 - self._mu[None, :]) / self._sigma[None, :],
            (self._high + self._step / 2 - self._mu[None, :]) / self._sigma[None, :],
        )
        return log_gauss_mass - log_p_accept


class _MixtureOfProductDistribution(NamedTuple):
    weights: np.ndarray
    distributions: List[_BatchedDistributions]

    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)

        ret = np.empty((batch_size, len(self.distributions)), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            ret[:, i] = d.sample(rng, batch_size, active_indices)

        return ret

    def log_pdf(self, x: dict[str, np.ndarray]) -> np.ndarray:
        n_vars = len(x)
        batch_size = next(iter(x.values())).size
        log_pdfs = np.empty((batch_size, len(self.weights), n_vars), dtype=np.float64)
        for i, (d, xi) in enumerate(zip(self.distributions, x.values())):
            log_pdfs[:, :, i] = d.log_pdf(xi)
        weighted_log_pdf = np.sum(log_pdfs, axis=-1) + np.log(self.weights[None, :])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
