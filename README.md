# Backdoor attacks and corresponding defenses in continual federated learning settings.
Official implementation of [Towards a Defense against Backdoor Attacks in Continual Federated Learning]

## Prerequisites

* Python 3.9

A `requirements.txt` is provided as a fallback for use with `pip` or Anaconda.

## File description

* `backdoor_attack_and_defense.py`: Implementation of backdoor attacks and the backbone-shadow defense framework.
* `fed_aggregate.py`: Implementation of clients' training process in federated learning systems with potential attacks.
* `fedaggregate_RFA.py`: Implementation of RFA, a robust secure aggregation oracle based on the geometric median.
* `spectre.py`: Implementation of SPECTRE-based filters.
* `robust_estimator.py`: Implementation of robust estimation.

## Running an experiment

For experiments with backbone-shadow defense framework, you could run:

`python backdoor_attack_and_defense.py`

The hyperparameters are detailed in `backdoor_attack_and_defense.py`.

The files related to experiment are stored in the directory `output/`.