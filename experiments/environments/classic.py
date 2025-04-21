import numpy as np

class ClassicBanditEnv:
    def __init__(self, arm_probs):
        self.arm_probs = arm_probs
        self.n_arms = len(arm_probs)

    def pull(self, arm):
        return np.random.random() < self.arm_probs[arm]
