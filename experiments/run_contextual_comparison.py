import numpy as np

from bandits.contextual.linucb import LinUCBBandit
from bandits.contextual.lin_thompson import LinThompsonBandit
from experiments.environments.contextual import ContextualBanditEnv
from experiments.utils.plot import plot_multiple_cumulative_rewards, plot_multiple_regrets

def run_all(rounds=2000, n_arms=3, n_features=5, noise_std=0.2, gap_strength=0.05):
    # Shared environment with fixed true weights
    env = ContextualBanditEnv(n_arms=n_arms, n_features=n_features, noise_std=noise_std)
    
    # Clone the environment’s true thetas so we can compute optimal rewards
    env.theta = env.theta[0] + np.random.randn(n_arms, n_features) * gap_strength
    true_thetas = env.theta.copy()
    contexts = [env.get_context() for _ in range(rounds)]
    optimal_rewards = [max(x @ theta for theta in true_thetas) for x in contexts]

    bandits = {
        "linucb": LinUCBBandit(n_arms, n_features, alpha=0.5),
        "linthompson": LinThompsonBandit(n_arms, n_features, v=0.5)
    }

    results = {}

    for name, bandit in bandits.items():
        rewards = []
        bandit.reset()
        for context in contexts: # of length rounds
            arm = bandit.select_arm(context=context)
            reward = env.pull(arm, context)
            bandit.update(arm, reward, context=context)

            rewards.append(reward)

        results[name] = rewards

    plot_multiple_cumulative_rewards(results)
    plot_multiple_regrets(results, optimal_rewards)