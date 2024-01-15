import optuna


def objective(trial):
    v = {}
    for i in range(1, 1001):
        v[i] = 1 / i
    for i in range(100):
        trial.set_user_attr(f"attr_{i}", v)
    return trial.suggest_float("x", -10, 10) ** 2

# storage = "sqlite:///tmp.db"
# storage = "mysql+pymysql://user:test@127.0.0.1/optunatest"
# storage = "postgresql+psycopg2://user:test@127.0.0.1/optunatest"
study = optuna.create_study(storage=storage, study_name="test")
study.optimize(objective, n_trials=10)
