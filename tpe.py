import optuna


optuna.logging.set_verbosity(optuna.logging.WARNING)

N_STUDY = 100
N_TRIAL = 100
N_BATCH = 2

for n_dim in range(2, 11):
    with open(f"master_{n_dim}.txt", "w") as f:
        for seed in range(N_STUDY):
            sampler = optuna.samplers.TPESampler(
                seed=seed,
                multivariate=True,
                constant_liar=True,
            )
            study = optuna.create_study(sampler=sampler)
            for i in range(0, N_TRIAL, N_BATCH):
                trials = [study.ask() for _ in range(N_BATCH)]
                values = [0 for trial in trials]
                for d in range(n_dim):
                    for j, trial in enumerate(trials):
                        values[j] += trial.suggest_float(f"x_{d}", -10, 10) ** 2
                for trial, value in zip(trials, values):
                    study.tell(trial, value)
            f.write(f"{study.best_value}\n")
