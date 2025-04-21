import numpy as np
from bandits.base import BaseBandit

class LogisticBandit(BaseBandit):
    """
    Logistic Bandit with linear model for contextual bandits.
    This bandit uses a logistic function to model the probability of selecting each arm based on the context.
    The model is updated using gradient ascent based on the received reward.
    The reward is assumed to be binary (0 or 1).
    """
    def __init__(self, n_arms, n_features, lr=0.1, discount=1.0):
        super().__init__(n_arms)
        self.n_features = n_features
        self.lr = lr
        self.theta = [np.zeros(n_features) for _ in range(n_arms)]
        self.discount = discount

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def select_arm(self, **kwargs):
        context = kwargs.get('context', None)
        if context is None:
            raise ValueError("Context must be provided for Logistic Bandit.")
        
        scores = [self.sigmoid(context @ self.theta[a]) for a in range(self.n_arms)]
        return np.argmax(scores)
    
    def update(self, arm, reward, **kwargs):
        context = kwargs.get('context', None)
        if context is None:
            raise ValueError("Context must be provided for Logistic Bandit.")
        
        pred = self.sigmoid(context @ self.theta[arm])
        error = reward - pred
        self.theta[arm] *= self.discount
        self.theta[arm] += self.lr * error * context

    def reset(self):
        self.theta = [np.zeros(self.n_features) for _ in range(self.n_arms)]
