import numpy as np

class CombinatorialBanditEnv:
    def __init__(self, n_arms, n_features, k, noise_std=0.1, reward_type="gaussian", **kwargs):
        self.n_arms = n_arms
        self.n_features = n_features
        self.k = k

        self.noise_std = noise_std
        self.reward_type = reward_type
        self.gap_strength = kwargs.get('gap_strength', 1)

        theta_0 = np.random.randn(n_features)
        self.theta = theta_0 + np.random.randn(n_arms, n_features) * self.gap_strength

    def get_context(self):
        return np.random.randn(self.n_features)
    
    def pull(self, arms, context):

        if self.reward_type == "bernoulli":
            logits = [context @ self.theta[arm] for arm in arms]
            exp_logits = np.exp(logits)
            no_click_logit = np.exp(0.0)
            exp_logits_with_baseline = np.append(exp_logits, no_click_logit)
            probs = exp_logits_with_baseline / np.sum(exp_logits_with_baseline)
            clicked = np.random.choice(len(arms) + 1, p=probs)
            if clicked < len(arms):
                rewards = [1 if i == clicked else 0 for i in range(len(arms))]
            else:
                rewards = [0] * len(arms)
        elif self.reward_type == "gaussian":
            rewards = []
            for arm in arms:
                mean = np.dot(self.theta[arm], context)
                reward = np.random.normal(mean, self.noise_std)
                rewards.append(reward)
        return rewards
