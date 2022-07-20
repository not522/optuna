from typing import Callable
from typing import Tuple
from typing import Union

from _pytest.mark.structures import ParameterSet
import pytest


def mark_skipif_unavailable_class(cls: Union[Callable, Tuple[Callable, ...]]) -> ParameterSet:
    if not isinstance(cls, tuple):
        cls = (cls,)

    try:
        cls[0]()
        skip = False
        reason = ""
    except Exception as e:
        skip = True
        reason = str(e)

    return pytest.param(*cls, marks=pytest.mark.skipif(skip, reason=reason))
