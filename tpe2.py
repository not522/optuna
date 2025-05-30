import optuna
import matplotlib.pyplot as plt

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_STUDY = 100
N_TRIAL = 200
N_BATCH = 10

y_min = [0, 0, 0.01, 0.05, 0.2, 1, 2, 5, 10, 15, 25]

for n_dim in range(2, 11):
    studies = [[], []]
    for k, constant_liar in enumerate((False, True)):
        for seed in range(N_STUDY):
            sampler = optuna.samplers.TPESampler(
                seed=seed,
                multivariate=True,
                constant_liar=constant_liar,
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
            studies[k].append(study)
    ax = optuna.visualization.matplotlib.plot_optimization_history(studies, error_bar=True)
    plt.ylim(y_min[n_dim], 50 * n_dim)
    plt.yscale("log")
    plt.legend()
    plt.title(f"{n_dim}-D Sphare")
    plt.savefig(f"{n_dim}.png")
