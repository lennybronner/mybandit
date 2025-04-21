import numpy as np

class ContextualBanditEnv:
    def __init__(self, n_arms, n_features, noise_std=0.1):
        self.n_arms = n_arms
        self.n_features = n_features
        self.noise_std = noise_std
        self.theta = np.random.randn(n_arms, n_features)

    def get_context(self):
        return np.random.randn(self.n_features)
    
    def pull(self, arm, context):
        mean_reward = np.dot(self.theta[arm], context)
        reward = mean_reward + np.random.normal(0, self.noise_std)
        return reward
