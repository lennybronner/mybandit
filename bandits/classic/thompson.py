import numpy as np
from bandits.base import BaseBandit

class ThompsonSamplingBandit(BaseBandit):
    """
    Thompson Sampling Bandit algorithm implementation.
    
    This algorithm selects arms based on the Thompson Sampling strategy,
    which uses Bayesian inference to update the probability distribution of each arm's reward.
    It samples from the posterior distribution of each arm and selects the arm with the highest sample.

    For each arm a, we maintain two parameters: alpha_a and beta_a.
    - alpha_a: Number of successes (rewards)
    - beta_a: Number of failures (non-rewards)

    At each round, we sample from the Beta distribution:
        sample_a ~ Beta(alpha_a, beta_a)
    and select the arm with the highest sample value.
    
    The Beta distribution is a conjugate prior for the Bernoulli distribution,
    which makes it suitable for modeling the success/failure of arms.
    
    The update rule is as follows:
        - If the arm gives a reward, we increment alpha_a
        - If the arm does not give a reward, we increment beta_a
    The algorithm is efficient and works well in practice, especially in non-stationary environments.
    """

    def __init__(self, n_arms):
        """
        Initialize the Thompson Sampling Bandit algorithm.

        Parameters:
        - n_arms: Number of arms.
        """
        super().__init__(n_arms)
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

    def select_arm(self, **kwargs):
        """
        Select an arm using the Thompson Sampling strategy.

        Returns:
        - arm: The selected arm.
        """
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm, reward, **kwargs):
        """
        Update the arm counts and rewards based on the received reward.

        Parameters:
        - arm: The selected arm.
        - reward: The received reward.
        """
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    @property
    def values(self):
        """
        Get the estimated values of each arm.

        Returns:
        - values: Estimated values of each arm.
        """
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def counts(self):
        """
        Get the counts of each arm.

        Returns:
        - counts: Counts of each arm.
        """
        return self.alpha + self.beta - 2 # 2 is subtracted because we initialized alpha and beta to 1

    def reset(self):
        """
        Reset the bandit algorithm.
        """
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)