import numpy as np
from bandits.base import BaseBandit

class CombinatorialThompsonSamplingBandit(BaseBandit):
    def __init__(self, n_arms, n_features, k, v=1.0):
        super().__init__(n_arms)
        self.n_features = n_features
        self.k = k
        self.v = v
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select_arm(self, **kwargs):
        x = kwargs.get('context', None)
        if x is None:
            raise ValueError("Context must be provided for Combinatorial Thompson Sampling.")
        
        scores = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            mu = A_inv @ self.b[a]
            cov = self.v ** 2 * A_inv

            theta_sample = np.random.multivariate_normal(mu, cov)
            scores[a] = x.T @ theta_sample
        
        top_k = np.argsort(scores)[-self.k:]
        return list(top_k)
    
    def update(self, arms, rewards, **kwargs):
        x = kwargs.get('context', None)
        if x is None:
            raise ValueError("Context must be provided for Combinatorial Thompson Sampling.")
        
        for arm, reward in zip(arms, rewards):
            self.A[arm] += np.outer(x, x)
            self.b[arm] += reward * x
            
    def reset(self):
        self.A = [np.identity(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.n_features) for _ in range(self.n_arms)]