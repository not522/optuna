"""
.. _rdb:

Saving/Resuming Study with RDB Backend
==========================================

An RDB backend enables persistent experiments (i.e., to save and resume a study) as well as access to history of studies.
In addition, we can run multi-node optimization tasks with this feature, which is described in :ref:`distributed`.

In this section, let's try simple examples running on a local environment with SQLite DB.

.. note::
    You can also utilize other RDB backends, e.g., PostgreSQL or MySQL, by setting the storage argument to the DB's URL.
    Please refer to `SQLAlchemy's document <https://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls>`__ for how to set up the URL.


New Study
---------

We can create a persistent study by calling :func:`~optuna.study.create_study` function as follows.
An SQLite file ``example.db`` is automatically initialized with a new study record.
"""

import logging
import sys

import optuna

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(study_name=study_name, storage=storage_name)

###################################################################################################
# To run a study, call :func:`~optuna.study.Study.optimize` method passing an objective function.


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study.optimize(objective, n_trials=3)

###################################################################################################
# Resume Study
# ------------
#
# To resume a study, instantiate a :class:`~optuna.study.Study` object
# passing the study name ``example-study`` and the DB URL ``sqlite:///example-study.db``.


study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=3)

###################################################################################################
# Note that the storage doesn't store the state of the instance of :mod:`~optuna.samplers`
# and :mod:`~optuna.pruners`.
# When we resume a study with a sampler whose ``seed`` argument is specified for
# reproducibility, you need to restore the sampler with using ``pickle`` as follows::
#
#     import pickle
#
#     # Save the sampler with pickle to be loaded later.
#     with open("sampler.pkl", "wb") as fout:
#         pickle.dump(study.sampler, fout)
#
#     restored_sampler = pickle.load(open("sampler.pkl", "rb"))
#     study = optuna.create_study(
#         study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler
#     )
#     study.optimize(objective, n_trials=3)
#

###################################################################################################
# Experimental History
# --------------------
#
# Note that this section requires the installation of `Pandas <https://pandas.pydata.org/>`__:
#
# .. code-block:: bash
#
#     $ pip install pandas
#
# We can access histories of studies and trials via the :class:`~optuna.study.Study` class.
# For example, we can get all trials of ``example-study`` as:

study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

###################################################################################################
# The method :func:`~optuna.study.Study.trials_dataframe` returns a pandas dataframe like:

print(df)

###################################################################################################
# A :class:`~optuna.study.Study` object also provides properties
# such as :attr:`~optuna.study.Study.trials`, :attr:`~optuna.study.Study.best_value`,
# :attr:`~optuna.study.Study.best_params` (see also :ref:`first`).


print("Best params: ", study.best_params)
print("Best value: ", study.best_value)
print("Best Trial: ", study.best_trial)
print("Trials: ", study.trials)
