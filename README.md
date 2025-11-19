# MyBandit

For me to learn how to implement multi-armed bandits.

## Install
```
python -m venv env
pip install -r requirements.txt
```

## Experiment
If you want to run an algorithm to experiment:
```
python main.py --mode classic --algo epsilon_greedy --epsilon 0.2 --rounds 1000
```

If you want to run a comparison of different algorithms:
```
python main.py --mode classic --algo comparison --rounds 10000
```

| Name           | Type    | Outcome  |
|----------------|---------|----------|
| mode           | str     | classic or contextual     |
| algo           | str     | algo name (e.g. `epsilon_greedy`) or `comparison`     |
| n_arms         | int     | number of arms for bandits        |
| n_features     | int     | number of features for contextual bandit   |
| rounds         | int     | number of rounds to run algorithms for    |
| epsilon        | float   | exploration/exploitation outcome for `epsilon_greedy` |
| alpha          | float   | exploration parameter for `LinUCB`     |
| v              | float   | variance parameter for `LinThompson`   |
| lr             | float   | learning rate for contextual `epsilon_greedy` |
| gap_strength   | float   | how similar/different arms should be (contextual comparison)      |
| noise_std      | float   | standard deviation for noise in rewards (contextual)    |
| reward_type    | str     | `gaussian` or `bernoulli` (only for contextual)    |
| verbose        | flag    | printing verbose output during comparison      |
| sweep          | flag    | run a sweep of hyperparameters (contextual comparison) |
| drift          | flag    | whether to add drift (non-stationarity) to the environment |
| time_dependent | flag    | whether the drift should change over time |
| drift_rate     | float   | drift magnitude |
| drift_frequency| int     | faster drift oscillation |
| discount       | flag    | whether to discount previous updates to adapt to drift |
| k              | int     | number of arms to pull (combinatorial bandit) |

## Things to implement:
- cascade combinatorial bandits
- adverserial bandits
- warmup
