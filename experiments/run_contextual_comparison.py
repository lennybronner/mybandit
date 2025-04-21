import numpy as np

from bandits.contextual.linucb import LinUCBBandit
from bandits.contextual.lin_thompson import LinThompsonBandit
from bandits.classic.ucb import UCB1Bandit
from bandits.classic.thompson import ThompsonSamplingBandit
from experiments.environments.contextual import ContextualBanditEnv
from experiments.utils.plot import plot_multiple_cumulative_rewards, plot_multiple_regrets, plot_multiple_running_ctr

def run_all(rounds=2000, n_arms=3, n_features=5, noise_std=0.2, gap_strength=0.05, reward_type="gaussian", verbose=False):
    # Shared environment with fixed true weights
    env = ContextualBanditEnv(n_arms=n_arms, n_features=n_features, noise_std=noise_std, reward_type=reward_type)
    
    # Clone the environment’s true thetas so we can compute optimal rewards
    env.theta = env.theta[0] + np.random.randn(n_arms, n_features) * gap_strength
    true_thetas = env.theta.copy()
    contexts = [env.get_context() for _ in range(rounds)]
    optimal_rewards = [max(x @ theta for theta in true_thetas) for x in contexts]
    bandits = {
        "linucb": LinUCBBandit(n_arms, n_features, alpha=0.5),
        "linthompson": LinThompsonBandit(n_arms, n_features, v=0.5),
        "ucb": UCB1Bandit(n_arms), # Classic UCB1 for comparison, the environment is still contextual
        "thompson": ThompsonSamplingBandit(n_arms), # Classic Thompson for comparison, the environment is still contextual
    }

    results = {}
    for name, bandit in bandits.items():
        rewards = []
        bandit.reset()
        for t, context in enumerate(contexts): # of length rounds

            arm = bandit.select_arm(context=context)
            reward = env.pull(arm, context)
            bandit.update(arm, reward, context=context)
            
            opt = optimal_rewards[t]
            if verbose:
                regret = opt - reward
                print(f"[{name}] Round {t}: arm={arm}, reward={reward:.2f}, optimal={opt:.2f}, regret={regret:.2f}")
            rewards.append(reward)

        results[name] = rewards

    plot_multiple_cumulative_rewards(results)
    plot_multiple_regrets(results, optimal_rewards)
    if reward_type == "bernoulli":
        plot_multiple_running_ctr(results)