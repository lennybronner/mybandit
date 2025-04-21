import numpy as np
from bandits.base import BaseBandit

class LinUCBBandit(BaseBandit):
    """
    LinUCB algorithm for contextual bandits.
    
    This implementation uses a linear model to predict the expected reward
    based on the context. The algorithm maintains a set of parameters for
    each arm, which are updated based on the observed rewards and contexts.

    - The reward is linearly dependent on context x
    - Each arm has its own weight vector theta_a
    - We use an upper confidence bound to choose arms:
        score_a = x^T * theta_a + alpha * sqrt(x^T * A_a^-1 * x)

        where
        - A_a is the design matrix for arm a (starts as identity)
        - b_a is the reward vector for arm a (updated after each round)
        - \hat{\theta}_a = A_a^-1 * b_a
    """

    def __init__(self, n_arms, n_features, alpha=1.0):
        super().__init__(n_arms)
        self.n_features = n_features
        self.alpha = alpha
        self.A = [np.identity(n_features) for _ in range(n_arms)] # A = X^T X + I
        self.b = [np.zeros(n_features) for _ in range(n_arms)] # b = X^T y

    def select_arm(self, **kwargs):
        x = kwargs.get('context', None)
        if x is None:
            raise ValueError("Context must be provided for LinUCB.")
        
        scores = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            uncertainty = self.alpha * np.sqrt(x.T @ A_inv @ x)
            scores[a] = x.T @ theta + uncertainty

        return np.argmax(scores)
    
    def update(self, arm, reward, **kwargs):
        x = kwargs.get('context', None)
        if x is None:
            raise ValueError("Context must be provided for LinUCB.")
        
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def reset(self):
        self.A = [np.identity(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.n_features) for _ in range(self.n_arms)]