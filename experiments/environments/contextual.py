import numpy as np

class ContextualBanditEnv:
    def __init__(self, n_arms, n_features, noise_std=0.1, reward_type="gaussian"):
        self.n_arms = n_arms
        self.n_features = n_features
        self.noise_std = noise_std
        self.theta = np.random.randn(n_arms, n_features)
        self.reward_type = reward_type

    def get_context(self):
        return np.random.randn(self.n_features)
    
    def pull(self, arm, context):
        mean_reward = np.dot(self.theta[arm], context)
        if self.reward_type == "gaussian":
            reward = np.random.normal(mean_reward, self.noise_std)
        elif self.reward_type == "bernoulli":
            prob = 1 / (1 + np.exp(-mean_reward)) # sigmoid
            reward = int(np.random.rand() < prob) # Bernoulli sample
        return reward
