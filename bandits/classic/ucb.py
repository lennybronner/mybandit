import numpy as np
from bandits.base import BaseBandit

class UCB1Bandit(BaseBandit):
    """
    Upper Confidence Bound (UCB1) Bandit algorithm implementation.
    This algorithm selects arms based on the UCB strategy,
    which balances exploration and exploitation by considering the uncertainty in the estimated values.

    It picks the arm with the highest upper confidence bound:
        UCB_a = \hat{\mu}_a + \sqrt{\frac{2 \log t}{N_a}}

    where:
    - \hat{\mu}_a is the estimated value of arm a
    - N_a is the number of times arm a has been pulled
    - t is the current round number

    If an arm is underexplored (ie. N_a is small), the UCB value will be high.
    As the arm is pulled more often, the UCB value will decrease.
    """
    
    def __init__(self, n_arms):
        """
        Initialize the UCB1 Bandit algorithm.

        Parameters:
        - n_arms: Number of arms.
        """
        super().__init__(n_arms)
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_pulls = 0

    def select_arm(self, **kwargs):
        """
        Select an arm using the Upper Confidence Bound (UCB) strategy.

        Returns:
        - arm: The selected arm.
        """
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                return a # play each arm once first


        ucb_values = self.values + np.sqrt((2 * np.log(self.total_pulls)) / self.counts)
        return np.argmax(ucb_values)
    
    def update(self, arm, reward, **kwargs):
        """
        Update the arm counts and rewards based on the received reward.

        Parameters:
        - arm: The selected arm.
        - reward: The received reward.
        """
        self.total_pulls += 1
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

    def reset(self):
        """
        Reset the bandit algorithm.
        """
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.total_pulls = 0