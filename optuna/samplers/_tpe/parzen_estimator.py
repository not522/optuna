from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Optional

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers._tpe.probability_distributions import _BatchedCategoricalDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDiscreteTruncNormDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedDistributions
from optuna.samplers._tpe.probability_distributions import _BatchedTruncNormDistributions


EPS = 1e-12


class _ParzenEstimatorParameters(
    NamedTuple(
        "_ParzenEstimatorParameters",
        [
            ("consider_prior", bool),
            ("prior_weight", Optional[float]),
            ("consider_magic_clip", bool),
            ("consider_endpoints", bool),
            ("weights", Callable[[int], np.ndarray]),
            ("multivariate", bool),
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
            self._calculate_distributions(observations[param], search_space[param])
            for i, param in enumerate(search_space)
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

    def _calculate_distributions(
        self,
        observations: np.ndarray,
        search_space: BaseDistribution,
    ) -> _BatchedDistributions:
        if isinstance(search_space, CategoricalDistribution):
            return _CategoricalDistributionsFactory(
                self._parameters.consider_prior, self._parameters.prior_weight
            )(observations, search_space)
        else:
            assert isinstance(search_space, (FloatDistribution, IntDistribution))
            return _NumericalDistributionsFactory(
                self._parameters.consider_prior,
                self._parameters.multivariate,
                self._parameters.consider_endpoints,
                self._parameters.consider_magic_clip,
                len(self._search_space),
            )(observations, search_space)


class _CategoricalDistributionsFactory:
    def __init__(self, consider_prior, prior_weight) -> None:
        self._consider_prior = consider_prior
        self._prior_weight = prior_weight

    def __call__(
        self,
        observations: np.ndarray,
        search_space: CategoricalDistribution,
    ) -> _BatchedDistributions:
        consider_prior = self._consider_prior or len(observations) == 0

        assert self._prior_weight is not None
        weights = np.full(
            shape=(len(observations) + consider_prior, len(search_space.choices)),
            fill_value=self._prior_weight / (len(observations) + consider_prior),
        )

        weights[np.arange(len(observations)), observations.astype(int)] += 1
        weights /= weights.sum(axis=1, keepdims=True)
        return _BatchedCategoricalDistributions(weights)


class _NumericalDistributionsFactory:
    def __init__(
        self,
        consider_prior: bool,
        multivariate: bool,
        consider_endpoints: bool,
        consider_magic_clip: bool,
        n_params: int,
    ) -> None:
        self._consider_prior = consider_prior
        self._multivariate = multivariate
        self._consider_endpoints = consider_endpoints
        self._consider_magic_clip = consider_magic_clip
        self._n_params = n_params

    def __call__(
        self,
        observations: np.ndarray,
        search_space: FloatDistribution | IntDistribution,
    ) -> _BatchedDistributions:
        if search_space.log:
            observations = np.log(observations)
            low = np.log(search_space.low)
            high = np.log(search_space.high)
        else:
            low = search_space.low
            high = search_space.high
        step = search_space.step

        # TODO(contramundum53): This is a hack and should be fixed.
        if step is not None and search_space.log:
            low = np.log(search_space.low - step / 2)
            high = np.log(search_space.high + step / 2)
            step = None

        step_or_0 = step or 0

        mus = observations
        consider_prior = self._consider_prior or len(observations) == 0

        def compute_sigmas() -> np.ndarray:
            if self._multivariate:
                SIGMA0_MAGNITUDE = 0.2
                sigma = (
                    SIGMA0_MAGNITUDE
                    * max(len(observations), 1) ** (-1.0 / (self._n_params + 4))
                    * (high - low + step_or_0)
                )
                sigmas = np.full(shape=(len(observations),), fill_value=sigma)
            else:
                # TODO(contramundum53): Remove dependency on prior_mu
                prior_mu = 0.5 * (low + high)
                mus_with_prior = np.append(mus, prior_mu) if consider_prior else mus

                sorted_indices = np.argsort(mus_with_prior)
                sorted_mus = mus_with_prior[sorted_indices]
                sorted_mus_with_endpoints = np.empty(len(mus_with_prior) + 2, dtype=float)
                sorted_mus_with_endpoints[0] = low - step_or_0 / 2
                sorted_mus_with_endpoints[1:-1] = sorted_mus
                sorted_mus_with_endpoints[-1] = high + step_or_0 / 2

                sorted_sigmas = np.maximum(
                    sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2],
                    sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1],
                )

                if not self._consider_endpoints and sorted_mus_with_endpoints.shape[0] >= 4:
                    sorted_sigmas[0] = sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]
                    sorted_sigmas[-1] = (
                        sorted_mus_with_endpoints[-2] - sorted_mus_with_endpoints[-3]
                    )

                sigmas = sorted_sigmas[np.argsort(sorted_indices)][: len(observations)]

            # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
            maxsigma = 1.0 * (high - low + step_or_0)
            if self._consider_magic_clip:
                # TODO(contramundum53): Remove dependency of minsigma on consider_prior.
                minsigma = (
                    1.0
                    * (high - low + step_or_0)
                    / min(100.0, (1.0 + len(observations) + consider_prior))
                )
            else:
                minsigma = EPS
            return np.asarray(np.clip(sigmas, minsigma, maxsigma))

        sigmas = compute_sigmas()

        if consider_prior:
            prior_mu = 0.5 * (low + high)
            prior_sigma = 1.0 * (high - low + step_or_0)
            mus = np.append(mus, [prior_mu])
            sigmas = np.append(sigmas, [prior_sigma])

        if step is None:
            return _BatchedTruncNormDistributions(
                mus, sigmas, low, high, search_space.log, search_space
            )
        else:
            return _BatchedDiscreteTruncNormDistributions(
                mus, sigmas, low, high, step, search_space.log
            )
