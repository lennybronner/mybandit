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

## Things to implement:
- contextual bandits
- non-stationarity
- combinatorial bandits
- (maybe adverserial bandits)