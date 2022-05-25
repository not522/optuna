import math
import sys
from typing import Callable
from typing import Dict
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np
import scipy.special as special

from optuna import distributions
from optuna.distributions import BaseDistribution


EPS = 1e-12
SIGMA0_MAGNITUDE = 0.2

_DISTRIBUTION_CLASSES = (
    distributions.CategoricalDistribution,
    distributions.FloatDistribution,
    distributions.IntDistribution,
)


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


TRUNCNORM_TAIL_X = 30


def _ndtr(a):
    x = a / 2 ** 0.5
    z = abs(x)

    if z < 1 / 2 ** 0.5:
        y = 0.5 + 0.5 * math.erf(x)
    else:
        y = 0.5 * math.erfc(z)
        if x > 0:
            y = 1.0 - y

    return y


def _norm_logcdf(a):
    if a > 6:
        return -_ndtr(-a)
    if a > -20:
        return math.log(_ndtr(a))

    log_LHS = -0.5 * a ** 2 - math.log(-a) - 0.5 * math.log(2 * math.pi)
    last_total = 0
    right_hand_side = 1
    numerator = 1
    denom_factor = 1
    denom_cons = 1 / a ** 2
    sign = 1
    i = 0

    while abs(last_total - right_hand_side) > sys.float_info.epsilon:
        i += 1
        last_total = right_hand_side
        sign = -sign
        denom_factor *= denom_cons
        numerator *= 2 * i - 1
        right_hand_side += sign * numerator * denom_factor

    return log_LHS + math.log(right_hand_side)


def _norm_sf(x):
    return _ndtr(-x)


def _norm_logsf(x):
    return _norm_logcdf(-x)


def _truncnorm_get_delta_scalar(a, b):
    if (a > TRUNCNORM_TAIL_X) or (b < -TRUNCNORM_TAIL_X):
        return 0
    if a > 0:
        delta = _norm_sf(a) - _norm_sf(b)
    else:
        delta = _ndtr(b) - _ndtr(a)
    delta = max(delta, 0)
    return delta


def bisect(f, a, b, c):
    fa = f(a)
    fb = f(b)
    for _ in range(100):
        m = (a + b) / 2
        fm = f(m)
        if (fa < c and fm < c) or (fa > c and fm > c):
            a = m
            fa = fm
        else:
            b = m
            fb = fm
    return m


def _truncnorm_ppf_scalar(q, a, b):
    shp = np.shape(q)
    q = np.atleast_1d(q)
    out = np.zeros(np.shape(q))
    condle0, condge1 = (q <= 0), (q >= 1)
    if np.any(condle0):
        out[condle0] = a
    if np.any(condge1):
        out[condge1] = b
    delta = _truncnorm_get_delta_scalar(a, b)
    cond_inner = ~condle0 & ~condge1
    if np.any(cond_inner):
        qinner = q[cond_inner]
        if delta > 0:
            if a > 0:
                sa, sb = _norm_sf(a), _norm_sf(b)
                np.place(out, cond_inner, bisect(_norm_sf, a, b, qinner * sb + sa * (1.0 - qinner)))
            else:
                na, nb = _ndtr(a), _ndtr(b)
                np.place(out, cond_inner, bisect(_ndtr, a, b, qinner * nb + na * (1.0 - qinner)))
        else:
            if b < 0:
                # Solve
                # norm_logcdf(x)
                #      = norm_logcdf(a) + log1p(q * (expm1(norm_logcdf(b)
                #                                    - norm_logcdf(a)))
                #      = nla + log1p(q * expm1(nlb - nla))
                #      = nlb + log(q) + log1p((1-q) * exp(nla - nlb)/q)
                nla, nlb = _norm_logcdf(a), _norm_logcdf(b)
                values = nlb + np.log(q[cond_inner])
                C = np.exp(nla - nlb)
                if C:
                    one_minus_q = (1 - q)[cond_inner]
                    values += np.log1p(one_minus_q * C / q[cond_inner])
                x = [bisect(_norm_logcdf, a, b, c) for c in values]
                np.place(out, cond_inner, x)
            else:
                # Solve
                # norm_logsf(x)
                #      = norm_logsf(b) + log1p((1-q) * (expm1(norm_logsf(a)
                #                                       - norm_logsf(b)))
                #      = slb + log1p((1-q)[cond_inner] * expm1(sla - slb))
                #      = sla + log(1-q) + log1p(q * np.exp(slb - sla)/(1-q))
                sla, slb = _norm_logsf(a), _norm_logsf(b)
                one_minus_q = (1 - q)[cond_inner]
                values = sla + np.log(one_minus_q)
                C = np.exp(slb - sla)
                if C:
                    values += np.log1p(q[cond_inner] * C / one_minus_q)
                x = [bisect(_norm_logsf, a, b, c) for c in values]
                np.place(out, cond_inner, x)
        out[out < a] = a
        out[out > b] = b
    return (out[0] if (shp == ()) else out)


def _ppf(q, a, b):
    if np.isscalar(a) and np.isscalar(b):
        return _truncnorm_ppf_scalar(q, a, b)
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    if a.size == 1 and b.size == 1:
        return _truncnorm_ppf_scalar(q, a.item(), b.item())

    out = None
    it = np.nditer([q, a, b, out], [],
                   [['readonly'], ['readonly'], ['readonly'],
                    ['writeonly', 'allocate']])
    for (_q, _a, _b, _x) in it:
        _x[...] = _truncnorm_ppf_scalar(_q, _a, _b)
    return it.operands[3]


class _ParzenEstimator:
    def __init__(
        self,
        observations: Dict[str, np.ndarray],
        search_space: Dict[str, BaseDistribution],
        parameters: _ParzenEstimatorParameters,
        predetermined_weights: Optional[np.ndarray] = None,
    ) -> None:

        self._search_space = search_space
        self._parameters = parameters
        self._n_observations = next(iter(observations.values())).size
        if predetermined_weights is not None:
            assert self._n_observations == len(predetermined_weights)
        self._weights = self._calculate_weights(predetermined_weights)

        self._low: Dict[str, Optional[float]] = {}
        self._high: Dict[str, Optional[float]] = {}
        self._q: Dict[str, Optional[float]] = {}
        for param_name, dist in search_space.items():
            if isinstance(dist, distributions.CategoricalDistribution):
                low = high = q = None
            else:
                low, high, q = self._calculate_parzen_bounds(dist)
            self._low[param_name] = low
            self._high[param_name] = high
            self._q[param_name] = q

        # `_low`, `_high`, `_q` are needed for transformation.
        observations = self._transform_to_uniform(observations)

        # Transformed `observations` might be needed for following operations.
        self._sigmas0 = self._precompute_sigmas0(observations)

        self._mus: Dict[str, Optional[np.ndarray]] = {}
        self._sigmas: Dict[str, Optional[np.ndarray]] = {}
        self._categorical_weights: Dict[str, Optional[np.ndarray]] = {}
        categorical_weights: Optional[np.ndarray]
        for param_name, dist in search_space.items():
            param_observations = observations[param_name]
            if isinstance(dist, distributions.CategoricalDistribution):
                mus = sigmas = None
                categorical_weights = self._calculate_categorical_params(
                    param_observations, param_name
                )
            else:
                mus, sigmas = self._calculate_numerical_params(param_observations, param_name)
                categorical_weights = None
            self._mus[param_name] = mus
            self._sigmas[param_name] = sigmas
            self._categorical_weights[param_name] = categorical_weights

    def sample(self, rng: np.random.RandomState, size: int) -> Dict[str, np.ndarray]:

        samples_dict = {}
        active = rng.choice(len(self._weights), size, p=self._weights)

        for param_name, dist in self._search_space.items():

            if isinstance(dist, distributions.CategoricalDistribution):
                categorical_weights = self._categorical_weights[param_name]
                assert categorical_weights is not None
                weights = categorical_weights[active, :]
                samples = _ParzenEstimator._sample_from_categorical_dist(rng, weights)

            else:
                # We restore parameters of parzen estimators.
                low = self._low[param_name]
                high = self._high[param_name]
                mus = self._mus[param_name]
                sigmas = self._sigmas[param_name]
                assert low is not None
                assert high is not None
                assert mus is not None
                assert sigmas is not None

                # We sample from truncnorm.
                trunc_low = (low - mus[active]) / sigmas[active]
                trunc_high = (high - mus[active]) / sigmas[active]
                samples = np.full((), fill_value=high + 1.0, dtype=np.float64)
                while (samples >= high).any():
                    a, b = trunc_low, trunc_high
                    size = (size,)
                    random_state = rng
                    out = np.empty(size)

                    it = np.nditer([a, b], flags=['multi_index'], op_flags=[['readonly'], ['readonly']])
                    while not it.finished:
                        idx = (it.multi_index[0],)
                        U = random_state.uniform(low=0, high=1, size=1)
                        out[idx] = _ppf(U, it[0], it[1])
                        it.iternext()

                    rvs = out * sigmas[active] + mus[active]

                    samples = np.where(samples < high, samples, rvs)
                print(samples)
                print(sum(samples))
            samples_dict[param_name] = samples
        samples_dict = self._transform_from_uniform(samples_dict)
        return samples_dict

    def log_pdf(self, samples_dict: Dict[str, np.ndarray]) -> np.ndarray:

        samples_dict = self._transform_to_uniform(samples_dict)
        n_observations = len(self._weights)
        n_samples = next(iter(samples_dict.values())).size
        if n_samples == 0:
            return np.asarray([], dtype=float)

        # When the search space is one CategoricalDistribution, we use the faster processing,
        # whose computation result is equivalent to the general one.
        if len(self._search_space.items()) == 1:
            param_name, dist = list(self._search_space.items())[0]
            if isinstance(dist, distributions.CategoricalDistribution):
                samples = samples_dict[param_name]
                categorical_weights = self._categorical_weights[param_name]
                assert categorical_weights is not None
                ret = np.log(np.inner(categorical_weights.T, self._weights))[samples]
                return ret

        # We compute log pdf (component_log_pdf)
        # for each sample in samples_dict (of size n_samples)
        # for each component of `_MultivariateParzenEstimator` (of size n_observations).
        component_log_pdf = np.zeros((n_samples, n_observations))
        for param_name, dist in self._search_space.items():
            samples = samples_dict[param_name]
            if isinstance(dist, distributions.CategoricalDistribution):
                categorical_weights = self._categorical_weights[param_name]
                assert categorical_weights is not None
                log_pdf = np.log(categorical_weights.T[samples, :])
            else:
                # We restore parameters of parzen estimators.
                low = np.asarray(self._low[param_name])
                high = np.asarray(self._high[param_name])
                q = self._q[param_name]
                mus = self._mus[param_name]
                sigmas = self._sigmas[param_name]
                assert low is not None
                assert high is not None
                assert mus is not None
                assert sigmas is not None

                cdf_func = _ParzenEstimator._normal_cdf
                p_accept = cdf_func(high, mus, sigmas) - cdf_func(low, mus, sigmas)
                if q is None:
                    distance = samples[:, None] - mus
                    mahalanobis = distance / np.maximum(sigmas, EPS)
                    z = np.sqrt(2 * np.pi) * sigmas
                    coefficient = 1 / z / p_accept
                    log_pdf = -0.5 * mahalanobis**2 + np.log(coefficient)
                else:
                    upper_bound = np.minimum(samples + q / 2.0, high)
                    lower_bound = np.maximum(samples - q / 2.0, low)
                    cdf = cdf_func(upper_bound[:, None], mus[None], sigmas[None]) - cdf_func(
                        lower_bound[:, None], mus[None], sigmas[None]
                    )
                    log_pdf = np.log(cdf + EPS) - np.log(p_accept + EPS)
            component_log_pdf += log_pdf
        weighted_log_pdf = component_log_pdf + np.log(self._weights)
        max_ = weighted_log_pdf.max(axis=1)
        with np.errstate(divide="ignore"):
            return np.log(np.exp(weighted_log_pdf - max_[:, np.newaxis]).sum(axis=1)) + max_

    def _calculate_weights(self, predetermined_weights: Optional[np.ndarray]) -> np.ndarray:

        # We decide the weights.
        consider_prior = self._parameters.consider_prior
        prior_weight = self._parameters.prior_weight
        weights_func = self._parameters.weights
        n_observations = self._n_observations

        if n_observations == 0:
            consider_prior = True

        if predetermined_weights is None:
            w = weights_func(n_observations)[:n_observations]
        else:
            w = predetermined_weights[:n_observations]

        if consider_prior:
            # TODO(HideakiImamura) Raise `ValueError` if the weight function returns an ndarray of
            # unexpected size.
            weights = np.zeros(n_observations + 1)
            weights[:-1] = w
            weights[-1] = prior_weight
        else:
            weights = w
        weights /= weights.sum()
        return weights

    def _calculate_parzen_bounds(
        self, distribution: BaseDistribution
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:

        # We calculate low and high.
        if isinstance(distribution, distributions.FloatDistribution):
            if distribution.log:
                low = np.log(distribution.low)
                high = np.log(distribution.high)
                q = None
            elif distribution.step is not None:
                q = distribution.step
                low = distribution.low - 0.5 * q
                high = distribution.high + 0.5 * q
            else:
                low = distribution.low
                high = distribution.high
                q = None
        elif isinstance(distribution, distributions.IntDistribution):
            if distribution.log:
                low = np.log(distribution.low - 0.5)
                high = np.log(distribution.high + 0.5)
                q = None
            else:
                q = distribution.step
                low = distribution.low - 0.5 * q
                high = distribution.high + 0.5 * q
        else:
            distribution_list = [
                distributions.CategoricalDistribution.__name__,
                distributions.FloatDistribution.__name__,
                distributions.IntDistribution.__name__,
            ]
            raise NotImplementedError(
                "The distribution {} is not implemented. "
                "The parameter distribution should be one of the {}".format(
                    distribution, distribution_list
                )
            )

        assert low < high

        return low, high, q

    def _transform_to_uniform(self, samples_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:

        transformed = {}
        for param_name, samples in samples_dict.items():
            distribution = self._search_space[param_name]

            assert isinstance(distribution, _DISTRIBUTION_CLASSES)
            if isinstance(
                distribution,
                (distributions.FloatDistribution, distributions.IntDistribution),
            ):
                if distribution.log:
                    samples = np.log(samples)

            transformed[param_name] = samples
        return transformed

    def _transform_from_uniform(
        self, samples_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:

        transformed = {}
        for param_name, samples in samples_dict.items():
            distribution = self._search_space[param_name]

            assert isinstance(distribution, _DISTRIBUTION_CLASSES)
            if isinstance(distribution, distributions.FloatDistribution):
                if distribution.log:
                    transformed[param_name] = np.exp(samples)
                elif distribution.step is not None:
                    q = self._q[param_name]
                    assert q is not None
                    samples = np.round((samples - distribution.low) / q) * q + distribution.low
                    transformed[param_name] = np.asarray(
                        np.clip(samples, distribution.low, distribution.high)
                    )
                else:
                    transformed[param_name] = samples
            elif isinstance(distribution, distributions.IntDistribution):
                if distribution.log:
                    samples = np.round(np.exp(samples))
                    transformed[param_name] = np.asarray(
                        np.clip(samples, distribution.low, distribution.high)
                    )
                else:
                    q = self._q[param_name]
                    assert q is not None
                    samples = np.round((samples - distribution.low) / q) * q + distribution.low
                    transformed[param_name] = np.asarray(
                        np.clip(samples, distribution.low, distribution.high)
                    )
            elif isinstance(distribution, distributions.CategoricalDistribution):
                transformed[param_name] = samples

        return transformed

    def _precompute_sigmas0(self, observations: Dict[str, np.ndarray]) -> Optional[float]:

        n_observations = next(iter(observations.values())).size
        n_observations = max(n_observations, 1)
        n_params = len(observations)

        # If it is univariate, there is no need to precompute sigmas0, so this method returns None.
        if not self._parameters.multivariate:
            return None

        # We use Scott's rule for bandwidth selection if the number of parameters > 1.
        # This rule was used in the BOHB paper.
        # TODO(kstoneriv3): The constant factor SIGMA0_MAGNITUDE=0.2 might not be optimal.
        return SIGMA0_MAGNITUDE * n_observations ** (-1.0 / (n_params + 4))

    def _calculate_categorical_params(
        self, observations: np.ndarray, param_name: str
    ) -> np.ndarray:

        # TODO(kstoneriv3): This the bandwidth selection rule might not be optimal.
        observations = observations.astype(int)
        n_observations = self._n_observations
        consider_prior = self._parameters.consider_prior
        prior_weight = self._parameters.prior_weight
        distribution = self._search_space[param_name]
        assert isinstance(distribution, distributions.CategoricalDistribution)
        choices = distribution.choices

        if n_observations == 0:
            consider_prior = True

        if consider_prior:
            shape = (n_observations + 1, len(choices))
            assert prior_weight is not None
            value = prior_weight / (n_observations + 1)
        else:
            shape = (n_observations, len(choices))
            assert prior_weight is not None
            value = prior_weight / n_observations
        weights = np.full(shape, fill_value=value)
        weights[np.arange(n_observations), observations] += 1
        weights /= weights.sum(axis=1, keepdims=True)
        return weights

    def _calculate_numerical_params(
        self, observations: np.ndarray, param_name: str
    ) -> Tuple[np.ndarray, np.ndarray]:

        n_observations = self._n_observations
        consider_prior = self._parameters.consider_prior
        consider_endpoints = self._parameters.consider_endpoints
        consider_magic_clip = self._parameters.consider_magic_clip
        multivariate = self._parameters.multivariate
        sigmas0 = self._sigmas0
        low = self._low[param_name]
        high = self._high[param_name]
        assert low is not None
        assert high is not None
        assert len(observations) == self._n_observations

        if n_observations == 0:
            consider_prior = True

        prior_mu = 0.5 * (low + high)
        prior_sigma = 1.0 * (high - low)

        if consider_prior:
            mus = np.empty(n_observations + 1)
            mus[:n_observations] = observations
            mus[n_observations] = prior_mu
            sigmas = np.empty(n_observations + 1)
        else:
            mus = observations
            sigmas = np.empty(n_observations)

        if multivariate:
            assert sigmas0 is not None
            sigmas[:] = sigmas0 * (high - low)
        else:
            assert sigmas0 is None
            sorted_indices = np.argsort(mus)
            sorted_mus = mus[sorted_indices]
            sorted_mus_with_endpoints = np.empty(len(mus) + 2, dtype=float)
            sorted_mus_with_endpoints[0] = low
            sorted_mus_with_endpoints[1:-1] = sorted_mus
            sorted_mus_with_endpoints[-1] = high

            sorted_sigmas = np.maximum(
                sorted_mus_with_endpoints[1:-1] - sorted_mus_with_endpoints[0:-2],
                sorted_mus_with_endpoints[2:] - sorted_mus_with_endpoints[1:-1],
            )

            if not consider_endpoints and sorted_mus_with_endpoints.shape[0] >= 4:
                sorted_sigmas[0] = sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]
                sorted_sigmas[-1] = sorted_mus_with_endpoints[-2] - sorted_mus_with_endpoints[-3]

            sigmas[:] = sorted_sigmas[np.argsort(sorted_indices)]

        # We adjust the range of the 'sigmas' according to the 'consider_magic_clip' flag.
        maxsigma = 1.0 * (high - low)
        if consider_magic_clip:
            minsigma = 1.0 * (high - low) / min(100.0, (1.0 + len(mus)))
        else:
            minsigma = EPS
        sigmas = np.asarray(np.clip(sigmas, minsigma, maxsigma))

        if consider_prior:
            sigmas[n_observations] = prior_sigma

        return mus, sigmas

    @staticmethod
    def _normal_cdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:

        mu, sigma = map(np.asarray, (mu, sigma))
        denominator = x - mu
        numerator = np.maximum(np.sqrt(2) * sigma, EPS)
        z = denominator / numerator
        return 0.5 * (1 + np.vectorize(math.erf)(z))

    @staticmethod
    def _sample_from_categorical_dist(
        rng: np.random.RandomState, probabilities: np.ndarray
    ) -> np.ndarray:

        n_samples = probabilities.shape[0]
        rnd_quantile = rng.rand(n_samples)
        cum_probs = np.cumsum(probabilities, axis=1)
        return np.sum(cum_probs < rnd_quantile[..., None], axis=1)
