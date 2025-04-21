import numpy as np
from bandits.base import BaseBandit

class ContextualEpsilonGreedyBandit(BaseBandit):
    """
    Contextual Epsilon-Greedy Bandit
    
    This bandit uses an epsilon-greedy strategy to select arms based on the context.
    It maintains a linear model for each arm and updates the model based on the received rewards.
    With probability \epsilon it explores by selecting a random arm, and with probability 1-\epsilon it exploits
    by selecting the arm with the highest predicted reward based on the context.
    The model updates theta parameters using online linear regression.
    """
    def __init__(self, n_arms, n_features, epsilon=0.1, lr=0.01):
        super().__init__(n_arms)
        self.n_features = n_features
        self.epsilon = epsilon
        self.lr = lr
        self.theta = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, **kwargs):
        x = kwargs.get('context', None)
        if x is None:
            raise ValueError("Context must be provided for ContextualEpsilonGreedy.")
        
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)
        
        scores = [x @ self.theta[a] for a in range(self.n_arms)]
        return np.argmax(scores)
    
    def update(self, arm, reward, **kwargs):
        x = kwargs.get('context', None)
        if x is None:
            raise ValueError("Context must be provided for ContextualEpsilonGreedy.")
        
        # online linear regression update
        pred = x @ self.theta[arm]
        error = reward - pred
        self.theta[arm] += self.lr * error * x

    def reset(self):
        self.theta = [np.zeros(self.n_features) for _ in range(self.n_arms)]