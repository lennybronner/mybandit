import numpy as np

class ContextualBanditEnv:
    def __init__(self, n_arms, n_features, noise_std=0.1, reward_type="gaussian", **kwargs):
        self.n_arms = n_arms
        self.n_features = n_features
        self.noise_std = noise_std
        
        self.gap_strength = kwargs.get('gap_strength', 1)
        theta_0 = np.random.randn(n_features)
        self.theta = theta_0 + np.random.randn(n_arms, n_features) * self.gap_strength
        
        self.reward_type = reward_type
        
        self.t = 0
        self.drift = kwargs.get('drift', False)
        self.time_dependent = kwargs.get('time_dependent', False)
        self.drift_rate = kwargs.get('drift_rate', 0.01)
        self.drift_frequency = kwargs.get('drift_frequency', 100)

    def get_context(self):
        return np.random.randn(self.n_features)
    
    def pull(self, arm, context):
        if self.time_dependent:
            self.t += 1
        if self.drift:
            drift = self.drift_rate * np.sin(self.t / self.drift_frequency)
            self.theta[arm] += drift * np.random.randn(self.n_features)
        mean_reward = np.dot(self.theta[arm], context)
        
        if self.reward_type == "gaussian":
            reward = np.random.normal(mean_reward, self.noise_std)
        elif self.reward_type == "bernoulli":
            prob = 1 / (1 + np.exp(-mean_reward)) # sigmoid
            reward = int(np.random.rand() < prob) # Bernoulli sample
        return reward
