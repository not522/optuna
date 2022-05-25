import math
import sys
from typing import Callable


TRUNCNORM_TAIL_X = 30


def _ndtr(a: float) -> float:
    x = a / 2 ** 0.5
    z = abs(x)

    if z < 1 / 2 ** 0.5:
        y = 0.5 + 0.5 * math.erf(x)
    else:
        y = 0.5 * math.erfc(z)
        if x > 0:
            y = 1.0 - y

    return y


def _log_ndtr(a: float) -> float:
    if a > 6:
        return -_ndtr(-a)
    if a > -20:
        return math.log(_ndtr(a))

    log_LHS = -0.5 * a ** 2 - math.log(-a) - 0.5 * math.log(2 * math.pi)
    last_total = 0.0
    right_hand_side = 1.0
    numerator = 1.0
    denom_factor = 1.0
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


def _truncnorm_delta(a: float, b: float) -> float:
    if (a > TRUNCNORM_TAIL_X) or (b < -TRUNCNORM_TAIL_X):
        return 0
    return _ndtr(b) - _ndtr(a)


def _bisect(f: Callable[[float], float], a: float, b: float, c: float) -> float:
    if f(a) > c:
        a, b = b, a
    for _ in range(100):
        m = (a + b) / 2
        if f(m) < c:
            a = m
        else:
            b = m
    return m


def truncnorm_ppf(q: float, a: float, b: float) -> float:
    if _truncnorm_delta(a, b) > 0:
        na, nb = _ndtr(a), _ndtr(b)
        return _bisect(_ndtr, a, b, q * nb + na * (1 - q))
    else:
        if b < 0:
            sign = 1
        else:
            sign = -1

        # Solve
        # norm_logcdf(x)
        #      = norm_logcdf(a) + log1p(q * (expm1(norm_logcdf(b)
        #                                    - norm_logcdf(a)))
        #      = nla + log1p(q * expm1(nlb - nla))
        #      = nlb + log(q) + log1p((1-q) * exp(nla - nlb)/q)
        nla, nlb = _log_ndtr(sign * a), _log_ndtr(sign * b)
        values = nlb + math.log(q)
        values += math.log1p((1 - q) * math.exp(nla - nlb) / q)
        return sign * _bisect(_log_ndtr, sign * a, sign * b, values)
