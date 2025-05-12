import numpy as np
from bandits.base import BaseBandit

class CombinatorialLinUCBBandit(BaseBandit):
    def __init__(self, n_arms, n_features, k, alpha=1.0, **kwargs):
        super().__init__(n_arms)
        self.n_features = n_features
        self.k = k
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, **kwargs):
        x = kwargs.get('context', None)
        if x is None:
            raise ValueError("Context must be provided for CombinatorialLinUCB.")
        
        scores = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            uncertainty = self.alpha * np.sqrt(x.T @ A_inv @ x)
            scores[a] = x.T @ theta + uncertainty
        
        top_k = np.argsort(scores)[-self.k:]
        return list(top_k)

    def update(self, arms, rewards, **kwargs):
        x = kwargs.get('context', None)
        if x is None:
            raise ValueError("Context must be provided for CombinatorialLinUCB.")
        
        for arm, reward in zip(arms, rewards):
            self.A[arm] += np.outer(x, x)
            self.b[arm] += reward * x

    def reset(self):
        self.A = [np.identity(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.n_features) for _ in range(self.n_arms)]