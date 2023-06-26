from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Optional

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions


EPS = 1e-12


class _ParzenEstimatorParameters(
    NamedTuple(
        "_ParzenEstimatorParameters",
        [
            ("consider_prior", bool),
            ("prior_weight", Optional[float]),
            ("weights", Callable[[int], np.ndarray]),
        ],
    )
):
    pass


class _ParzenEstimator:
    def __init__(
        self,
        observations: Dict[str, np.ndarray],
        search_space: Dict[str, BaseDistribution],
        parameters: _ParzenEstimatorParameters,
        distribution_factories: dict[
            type[BaseDistribution],
            Callable[
                [
                    np.ndarray,
                    BaseDistribution,
                    dict[str, BaseDistribution],
                ],
                _BatchedDistributions,
            ],
        ],
        predetermined_weights: Optional[np.ndarray] = None,
    ) -> None:
        if parameters.consider_prior:
            if parameters.prior_weight is None:
                raise ValueError("Prior weight must be specified when consider_prior==True.")
            elif parameters.prior_weight <= 0:
                raise ValueError("Prior weight must be positive.")

        self._search_space = search_space
        self._parameters = parameters

        n_observations = next(iter(observations.values())).size

        assert predetermined_weights is None or n_observations == len(predetermined_weights)
        weights = (
            predetermined_weights
            if predetermined_weights is not None
            else self._call_weights_func(parameters.weights, n_observations)
        )

        if n_observations == 0:
            weights = np.array([1.0])
        elif parameters.consider_prior:
            assert parameters.prior_weight is not None
            weights = np.append(weights, [parameters.prior_weight])
        weights /= weights.sum()
        self._weights = weights

        self._distributions = [
            distribution_factories[type(distribution)](
                observations[param], distribution, search_space
            )
            for i, (param, distribution) in enumerate(search_space.items())
        ]

    def sample(self, rng: np.random.RandomState, size: int) -> Dict[str, np.ndarray]:
        active_indices = rng.choice(len(self._weights), p=self._weights, size=size)

        samples = {}
        for param, d in zip(self._search_space, self._distributions):
            samples[param] = d.sample(rng, size, active_indices)

        return samples

    def log_pdf(self, samples_dict: Dict[str, np.ndarray]) -> np.ndarray:
        n_vars = len(samples_dict)
        batch_size = next(iter(samples_dict.values())).size
        log_pdfs = np.empty((batch_size, len(self._weights), n_vars), dtype=np.float64)
        for i, (d, xi) in enumerate(zip(self._distributions, samples_dict.values())):
            log_pdfs[:, :, i] = d.log_pdf(xi)
        weighted_log_pdf = np.sum(log_pdfs, axis=-1) + np.log(self._weights[None, :])
        max_ = weighted_log_pdf.max(axis=1)
        # We need to avoid (-inf) - (-inf) when the probability is zero.
        max_[np.isneginf(max_)] = 0
        with np.errstate(divide="ignore"):  # Suppress warning in log(0).
            return np.log(np.exp(weighted_log_pdf - max_[:, None]).sum(axis=1)) + max_

    @staticmethod
    def _call_weights_func(weights_func: Callable[[int], np.ndarray], n: int) -> np.ndarray:
        w = np.array(weights_func(n))[:n]
        if np.any(w < 0):
            raise ValueError(
                f"The `weights` function is not allowed to return negative values {w}. "
                + f"The argument of the `weights` function is {n}."
            )
        if len(w) > 0 and np.sum(w) <= 0:
            raise ValueError(
                f"The `weight` function is not allowed to return all-zero values {w}."
                + f" The argument of the `weights` function is {n}."
            )
        if not np.all(np.isfinite(w)):
            raise ValueError(
                "The `weights`function is not allowed to return infinite or NaN values "
                + f"{w}. The argument of the `weights` function is {n}."
            )

        # TODO(HideakiImamura) Raise `ValueError` if the weight function returns an ndarray of
        # unexpected size.
        return w
