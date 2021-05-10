import pytest
from pytest_benchmark.fixture import BenchmarkFixture

import optuna
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from optuna.testing.storage import StorageSupplier


SAMPLER_MODES = [
    "random",
    "tpe",
    "cmaes",
]


def create_sampler(sampler_mode: str) -> BaseSampler:
    if sampler_mode == "random":
        return RandomSampler()
    elif sampler_mode == "tpe":
        return TPESampler()
    elif sampler_mode == "cmaes":
        return CmaEsSampler()
    else:
        assert False


def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_int("y", -100, 100)
    return x ** 2 + y ** 2


def optimize(storage_mode: str, sampler_mode: str, n_trials: int) -> None:
    with StorageSupplier(storage_mode) as storage:
        sampler = create_sampler(sampler_mode)
        study = optuna.create_study(storage=storage, sampler=sampler)
        study.optimize(objective, n_trials=n_trials)


@pytest.mark.parametrize(
    "storage_mode, sampler_mode, n_trials",
    (
        ("inmemory", "random", 1000),
        ("inmemory", "random", 10000),
        ("inmemory", "tpe", 1000),
        ("inmemory", "cmaes", 1000),
        ("sqlite", "random", 1000),
        ("cache", "random", 1000),
        ("redis", "random", 1000),
    )
)
def test_bench_optimize(benchmark: BenchmarkFixture, storage_mode: str, sampler_mode: str, n_trials: int) -> None:
    benchmark.pedantic(optimize, args=(storage_mode, sampler_mode, n_trials), iterations=1, rounds=1)
