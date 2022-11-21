# This file contains the codes from SciPy project.
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import functools
import math
import sys
from typing import Callable
from typing import Optional
from typing import Union

import numpy as np


vectorize = lambda f: np.vectorize(f, cache=True)    
_norm_pdf_C = math.sqrt(2 * math.pi)
_norm_pdf_logC = math.log(_norm_pdf_C)


def _log_sum(log_p: float, log_q: float) -> float:
    if log_p > log_q:
        log_p, log_q = log_q, log_p
    return math.log1p(math.exp(log_p - log_q)) + log_q


def _log_diff(log_p: float, log_q: float) -> float:
    # returns log(q - p).
    # assuming that log_q is always greater than log_q
    return math.log1p(-math.exp(log_q - log_p)) + log_p


@functools.lru_cache
def _ndtr(a: float) -> float:
    x = a / 2**0.5

    if x < -1 / 2**0.5:
        y = 0.5 * math.erfc(-x)
    elif x < 1 / 2**0.5:
        y = 0.5 + 0.5 * math.erf(x)
    else:
        y = 1.0 - 0.5 * math.erfc(x)

    return y


@functools.lru_cache
def _log_ndtr(a: float) -> float:
    if a > 6:
        return -_ndtr(-a)
    if a > -20:
        return math.log(_ndtr(a))

    log_LHS = -0.5 * a**2 - math.log(-a) - 0.5 * math.log(2 * math.pi)
    last_total = 0.0
    right_hand_side = 1.0
    numerator = 1.0
    denom_factor = 1.0
    denom_cons = 1 / a**2
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


def _norm_logpdf(x: float) -> float:
    return -(x**2) / 2.0 - _norm_pdf_logC


@functools.lru_cache
def _log_gauss_mass(a: float, b: float) -> float:
    """Log of Gaussian probability mass within an interval"""

    # Calculations in right tail are inaccurate, so we'll exploit the
    # symmetry and work only in the left tail

    if b <= 0:
        return _log_diff(_log_ndtr(b), _log_ndtr(a))
    elif a > 0:
        return _log_diff(_log_ndtr(-a), _log_ndtr(-b))
    else:
        # Previously, this was implemented as:
        # left_mass = mass_case_left(a, 0)
        # right_mass = mass_case_right(0, b)
        # return _log_sum(left_mass, right_mass)
        # Catastrophic cancellation occurs as np.exp(log_mass) approaches 1.
        # Correct for this with an alternative formulation.
        # We're not concerned with underflow here: if only one term
        # underflows, it was insignificant; if both terms underflow,
        # the result can't accurately be represented in logspace anyway
        # because sc.log1p(x) ~ x for small x.
        return math.log1p(-_ndtr(a) - _ndtr(-b))


def _bisect(f: Callable[[float], float], a: float, b: float, c: float) -> float:
    if f(a) > c:
        a, b = b, a
    # TODO(amylase): Justify this constant
    for _ in range(100):
        m = (a + b) / 2
        if f(m) < c:
            a = m
        else:
            b = m
    return m


def _ndtri_exp(y: float) -> float:
    # TODO(amylase): Justify this constant
    return _bisect(_log_ndtr, -100, +100, y)


@vectorize
def ppf(q: float, a: float, b: float) -> float:
    if a == b:
        return np.nan
    if q == 0:
        return a
    if q == 1:
        return b

    if a < 0:
        log_Phi_x = _log_sum(_log_ndtr(a), math.log(q) + _log_gauss_mass(a, b))
        return _ndtri_exp(log_Phi_x)
    else:
        log_Phi_x = _log_sum(_log_ndtr(-b), math.log1p(-q) + _log_gauss_mass(a, b))
        return -_ndtri_exp(log_Phi_x)


def rvs(
    a: np.ndarray,
    b: np.ndarray,
    loc: Union[np.ndarray, float] = 0,
    scale: Union[np.ndarray, float] = 1,
    size: int = 1,
    random_state: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    random_state = random_state or np.random.RandomState()
    percentiles = random_state.uniform(low=0, high=1, size=size)
    return ppf(percentiles, a, b) * scale + loc


@vectorize
def logpdf(x: float, a: float, b: float, loc: float = 0, scale: float = 1) -> float:
    x = (x - loc) / scale
    if a == b:
        return np.nan
    if x < a or b < x:
        return -np.inf
    return _norm_logpdf(x) - _log_gauss_mass(a, b)


def _logcdf(x: float, a: float, b: float) -> float:
    logcdf = _log_gauss_mass(a, x) - _log_gauss_mass(a, b)
    if logcdf > -0.1:  # avoid catastrophic cancellation
        log_survival = _log_gauss_mass(x, b) - _log_gauss_mass(a, b)
        logcdf = math.log1p(-math.exp(log_survival))
    return logcdf


@vectorize
def logcdf(x: float, a: float, b: float, loc: float = 0, scale: float = 1) -> float:
    if a == b:
        return math.nan
    x = (x - loc) / scale
    if x <= a:
        return -math.inf
    if x >= b:
        return 0
    return _logcdf(x, a, b)


@vectorize
def cdf(x: float, a: float, b: float, loc: float = 0, scale: float = 1) -> float:
    if a == b:
        return math.nan
    x = (x - loc) / scale
    if x <= a:
        return 0
    if x >= b:
        return 1
    return math.exp(_logcdf(x, a, b))
