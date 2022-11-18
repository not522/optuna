import sys
from typing import Iterable
from unittest.mock import patch

import pytest

import optuna
from optuna._imports import try_import
from optuna.testing.pruners import DeterministicPruner


with try_import():
    from ignite.engine import Engine

# TODO(not522): Remove version constraints if TensorFlow supports Python 3.11
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.version_info >= (3, 11), reason="Python 3.11 is not supported."),
]


def test_pytorch_ignite_pruning_handler() -> None:
    def update(engine: Engine, batch: Iterable) -> None:

        pass

    trainer = Engine(update)
    evaluator = Engine(update)

    # The pruner is activated.
    study = optuna.create_study(pruner=DeterministicPruner(True))
    trial = study.ask()

    handler = optuna.integration.PyTorchIgnitePruningHandler(trial, "accuracy", trainer)
    with patch.object(trainer, "state", epoch=3):
        with patch.object(evaluator, "state", metrics={"accuracy": 1}):
            with pytest.raises(optuna.TrialPruned):
                handler(evaluator)
            assert study.trials[0].intermediate_values == {3: 1}

    # The pruner is not activated.
    study = optuna.create_study(pruner=DeterministicPruner(False))
    trial = study.ask()

    handler = optuna.integration.PyTorchIgnitePruningHandler(trial, "accuracy", trainer)
    with patch.object(trainer, "state", epoch=5):
        with patch.object(evaluator, "state", metrics={"accuracy": 2}):
            handler(evaluator)
            assert study.trials[0].intermediate_values == {5: 2}
