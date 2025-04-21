import numpy as np
import random
from bandits.base import BaseBandit

class EpsilonGreedyBandit(BaseBandit):
    """
    Epsilon-Greedy Bandit algorithm implementation.

    This algorithm selects arms based on an epsilon-greedy strategy,
    where with probability epsilon it explores a random arm,
    and with probability 1-epsilon it exploits the arm that gives the best reward.
    """
    
    def __init__(self, n_arms, epsilon=0.1):
        """
        Initialize the Epsilon-Greedy Bandit algorithm.

        Parameters:
        - n_arms: Number of arms.
        - epsilon: Probability of exploration (default is 0.1).
        """
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)

    def select_arm(self, **kwargs):
        """
        Select an arm using the Epsilon-Greedy strategy.

        Returns:
        - arm: The selected arm.
        """
        if random.random() < self.epsilon:
            # Exploration: select a random arm
            arm = random.randint(0, self.n_arms - 1)
        else:
            # Exploitation: select the arm with the highest average reward
            # argmax returns the first ocurrence in case of a tie
            arm = np.argmax(self.values)
        return arm
    
    def update(self, arm, reward, **kwargs):
        """
        Update the arm counts and rewards based on the received reward.

        Parameters:
        - arm: The selected arm.
        - reward: The received reward.
        """
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def reset(self):
        """
        Reset the bandit algorithm.
        """
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)