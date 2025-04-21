class BaseBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms

    def select_arm(self, **kwargs):
        """
        Select an arm to pull based on the current policy.
        
        Parameters:
        - kwargs: Additional parameters for the selection process (e.g., context).
        
        Returns:
        - arm: The index of the selected arm.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def update(self, arm, reward, **kwargs):
        """
        Update the internal state of the bandit based on the pulled arm and received reward.
        
        Parameters:
        - arm: The index of the pulled arm.
        - reward: The received reward.
        - kwargs: Additional parameters for the update process (e.g., context).
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    
    def reset(self):
        pass