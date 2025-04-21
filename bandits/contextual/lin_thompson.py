import numpy as np
from bandits.base import BaseBandit

class LinThompsonBandit(BaseBandit):
    """
    LinThompson algorithm for contextual bandits.
    
    This implementation uses a linear model to predict the expected reward
    based on the context. The algorithm maintains a set of parameters for
    each arm, which are updated based on the observed rewards and contexts.

    Instead of computing an upper confidence bound (like LinUCB), LinTS samples a random
    weight vector \tilde{\theta}_a from the posterior distribution for each arm, then
    selects the arm that looks best for the current context

    Each arm a maintains:
        - A_a: X_a^T X_a + I
        - b_a: X_a^T y_a

    Then:
        \tilde{\theta}_a \\sim N(\\hat{\theta}_a, v^2 A_a^{-1})
        score_a = x^T \tilde{\theta}_a
    where
        - \\hat{\theta}_a = A_a^{-1} b_a
        - v^2 is a variance scaling parameter (controls exploration)
    """

    def __init__(self, n_arms, n_features, v=1.0, discount=1.0):
        super().__init__(n_arms)
        self.n_features = n_features
        self.v = v

        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

        self.discount = discount

    def select_arm(self, **kwargs):
        x = kwargs.get('context', None)
        if x is None:
            raise ValueError("Context must be provided for LinThompson.")
        
        scores = np.zeros(self.n_arms)

        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            mu = A_inv @ self.b[a]
            cov = self.v ** 2 * A_inv

            theta_sample = np.random.multivariate_normal(mu, cov)
            scores[a] = x.T @ theta_sample

        return np.argmax(scores)
    
    def update(self, arm, reward, **kwargs):
        x = kwargs.get('context', None)
        if x is None:
            raise ValueError("Context must be provided for LinThompson.")
        
        self.A[arm] *= self.discount
        self.b[arm] *= self.discount
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x

    def reset(self):
        self.A = [np.identity(self.n_features) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.n_features) for _ in range(self.n_arms)]