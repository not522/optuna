import optuna
from optuna.samplers import BruteForceSampler


def objective(trial):
    c = trial.suggest_categorical("c", ["float", "int"])
    if c == "float":
        return trial.suggest_float("x", 1, 3, step=0.5)
    elif c == "int":
        a = trial.suggest_int("a", 1, 3)
        b = trial.suggest_int("b", a, 3)
        return a + b


def main():
    study = optuna.create_study(sampler=BruteForceSampler())
    study.optimize(objective)


if __name__ == "__main__":
    main()
