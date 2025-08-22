from __future__ import annotations

from typing import NamedTuple
from typing import Union

import numpy as np

from optuna.samplers._tpe import _truncnorm


class _BatchedCategoricalDistributions(NamedTuple):
    weights: np.ndarray


class _BatchedTruncNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low and high do not change per trial.
    high: float


class _BatchedTruncLogNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low and high do not change per trial.
    high: float


class _BatchedDiscreteTruncNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low, high and step do not change per trial.
    high: float
    step: float


class _BatchedDiscreteTruncLogNormDistributions(NamedTuple):
    mu: np.ndarray
    sigma: np.ndarray
    low: float  # Currently, low, high and step do not change per trial.
    high: float
    step: float


_BatchedDistributions = Union[
    _BatchedCategoricalDistributions,
    _BatchedTruncNormDistributions,
    _BatchedDiscreteTruncNormDistributions,
]


def _unique_inverse_2d(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function is a quicker version of:
        np.unique(np.concatenate([a[:, None], b[:, None]], axis=-1), return_inverse=True).
    """
    assert a.shape == b.shape and len(a.shape) == 1
    order = np.argsort(b)
    # Stable sorting is required for the tie breaking.
    order = order[np.argsort(a[order], kind="stable")]
    a_order = a[order]
    b_order = b[order]
    is_first_occurrence = np.empty_like(a, dtype=bool)
    is_first_occurrence[0] = True
    is_first_occurrence[1:] = (a_order[1:] != a_order[:-1]) | (b_order[1:] != b_order[:-1])
    inv = np.empty(a_order.size, dtype=int)
    inv[order] = np.cumsum(is_first_occurrence) - 1
    return a_order[is_first_occurrence], b_order[is_first_occurrence], inv


def _log_gauss_mass_unique(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    This function reduces the log Gaussian probability mass computation by avoiding the
    duplicated evaluations using the np.unique_inverse(...) equivalent operation.
    """
    a_uniq, b_uniq, inv = _unique_inverse_2d(a.ravel(), b.ravel())
    return _truncnorm._log_gauss_mass(a_uniq, b_uniq)[inv].reshape(a.shape)


class _MixtureOfProductDistribution(NamedTuple):
    weights: np.ndarray
    distributions: list[_BatchedDistributions]

    def sample(self, rng: np.random.RandomState, batch_size: int) -> np.ndarray:
        active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)
        ret = np.empty((batch_size, len(self.distributions)), dtype=float)
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                active_weights = d.weights[active_indices, :]
                rnd_quantile = rng.rand(batch_size)
                cum_probs = np.cumsum(active_weights, axis=-1)
                assert np.isclose(cum_probs[:, -1], 1).all()
                cum_probs[:, -1] = 1  # Avoid numerical errors.
                ret[:, i] = np.sum(cum_probs < rnd_quantile[:, np.newaxis], axis=-1)
            elif isinstance(d, _BatchedTruncNormDistributions):
                ret[:, i] = _truncnorm.rvs(
                    a=(d.low - d.mu[active_indices]) / d.sigma[active_indices],
                    b=(d.high - d.mu[active_indices]) / d.sigma[active_indices],
                    loc=d.mu[active_indices],
                    scale=d.sigma[active_indices],
                    random_state=rng,
                ).T
            elif isinstance(d, _BatchedTruncLogNormDistributions):
                ret[:, i] = np.exp(
                    _truncnorm.rvs(
                        a=(np.log(d.low) - d.mu[active_indices]) / d.sigma[active_indices],
                        b=(np.log(d.high) - d.mu[active_indices]) / d.sigma[active_indices],
                        loc=d.mu[active_indices],
                        scale=d.sigma[active_indices],
                        random_state=rng,
                    ).T
                )
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                ret[:, i] = _truncnorm.rvs(
                    a=(d.low - d.step / 2 - d.mu[active_indices]) / d.sigma[active_indices],
                    b=(d.high + d.step / 2 - d.mu[active_indices]) / d.sigma[active_indices],
                    loc=d.mu[active_indices],
                    scale=d.sigma[active_indices],
                    random_state=rng,
                ).T
                ret[:, i] = np.clip(
                    d.low + np.round((ret[:, i] - d.low) / d.step) * d.step, d.low, d.high
                )
            elif isinstance(d, _BatchedDiscreteTruncLogNormDistributions):
                ret[:, i] = np.exp(
                    _truncnorm.rvs(
                        a=(np.log(d.low - d.step / 2) - d.mu[active_indices])
                        / d.sigma[active_indices],
                        b=(np.log(d.high + d.step / 2) - d.mu[active_indices])
                        / d.sigma[active_indices],
                        loc=d.mu[active_indices],
                        scale=d.sigma[active_indices],
                        random_state=rng,
                    ).T
                )
                ret[:, i] = np.clip(
                    d.low + np.round((ret[:, i] - d.low) / d.step) * d.step, d.low, d.high
                )
            else:
                assert False

        return ret

    def log_pdf(self, x: np.ndarray) -> np.ndarray:
        weighted_log_pdf = np.zeros((len(x), len(self.weights)), dtype=np.float64)
        for i, d in enumerate(self.distributions):
            if isinstance(d, _BatchedCategoricalDistributions):
                xi = x[:, i, np.newaxis, np.newaxis].astype(np.int64)
                weighted_log_pdf += np.log(np.take_along_axis(d.weights[np.newaxis], xi, axis=-1))[
                    ..., 0
                ]
            elif isinstance(d, _BatchedTruncNormDistributions):
                weighted_log_pdf += _truncnorm.logpdf(
                    x[:, np.newaxis, i],
                    a=(d.low - d.mu) / d.sigma,
                    b=(d.high - d.mu) / d.sigma,
                    loc=d.mu,
                    scale=d.sigma,
                )
            elif isinstance(d, _BatchedTruncLogNormDistributions):
                weighted_log_pdf += _truncnorm.logpdf(
                    np.log(x[:, np.newaxis, i]),
                    a=(np.log(d.low) - d.mu) / d.sigma,
                    b=(np.log(d.high) - d.mu) / d.sigma,
                    loc=d.mu,
                    scale=d.sigma,
                )
            elif isinstance(d, _BatchedDiscreteTruncNormDistributions):
                xi_uniq, xi_inv = np.unique(x[:, i], return_inverse=True)
                mu_uniq, sigma_uniq, mu_sigma_inv = _unique_inverse_2d(d.mu, d.sigma)
                weighted_log_pdf += _log_gauss_mass_unique(
                    ((xi_uniq - d.step / 2)[:, np.newaxis] - mu_uniq) / sigma_uniq,
                    ((xi_uniq + d.step / 2)[:, np.newaxis] - mu_uniq) / sigma_uniq,
                )[np.ix_(xi_inv, mu_sigma_inv)]
                # Very unlikely to observe duplications below, so we skip the unique operation.
                weighted_log_pdf -= _truncnorm._log_gauss_mass(
                    (d.low - d.step / 2 - mu_uniq) / sigma_uniq,
                    (d.high + d.step / 2 - mu_uniq) / sigma_uniq,
                )[mu_sigma_inv]
            elif isinstance(d, _BatchedDiscreteTruncLogNormDistributions):
                weighted_log_pdf += _truncnorm.logpdf(
                    np.log(x[:, np.newaxis, i]),
                    a=(np.log(d.low - d.step / 2) - d.mu) / d.sigma,
                    b=(np.log(d.high + d.step / 2) - d.mu) / d.sigma,
                    loc=d.mu,
                    scale=d.sigma,
                )
            else:
                assert False

        weighted_log_pdf += np.log(self.weights[np.newaxis])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_
