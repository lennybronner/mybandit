import numpy as np

from bandits.contextual.linucb import LinUCBBandit
from bandits.contextual.lin_thompson import LinThompsonBandit
from bandits.classic.ucb import UCB1Bandit
from bandits.classic.thompson import ThompsonSamplingBandit
from experiments.environments.contextual import ContextualBanditEnv
from experiments.utils.plot import plot_multiple_cumulative_rewards, plot_multiple_regrets, plot_multiple_running_ctr

def run_bandit_on_contexts(name, bandit, contexts, env, optimal_rewards, verbose):
    rewards = []
    bandit.reset()
    for t, x in enumerate(contexts):
        arm = bandit.select_arm(context=x)
        reward = env.pull(arm, x)
        bandit.update(arm, reward, context=x)
        rewards.append(reward)
        opt = optimal_rewards[t]
        if verbose:
            regret = opt - reward
            print(f"[{name}] Round {t}: arm={arm}, reward={reward:.2f}, optimal={opt:.2f}, regret={regret:.2f}")
    return rewards

def run_all(rounds=2000, n_arms=3, n_features=5, noise_std=0.2, gap_strength=0.05, reward_type="gaussian", verbose=False, sweep=False):
    alpha_values = [0.5]
    v_values = [0.5]
    if sweep:
        alpha_values = [0.1, 0.5, 1.0, 2.0]
        v_values = [0.1, 0.5, 1.0, 2.0]

    # Shared environment with fixed true weights
    env = ContextualBanditEnv(n_arms=n_arms, n_features=n_features, noise_std=noise_std, reward_type=reward_type)
    
    # Clone the environment’s true thetas so we can compute optimal rewards
    env.theta = env.theta[0] + np.random.randn(n_arms, n_features) * gap_strength
    true_thetas = env.theta.copy()
    contexts = [env.get_context() for _ in range(rounds)]
    optimal_rewards = [max(x @ theta for theta in true_thetas) for x in contexts]
    
    bandits = {
        f"linucb (alpha={alpha_value})": LinUCBBandit(n_arms, n_features, alpha=alpha_value) for alpha_value in alpha_values
    }
    bandits.update({
        f"linthompson (v={v_value})": LinThompsonBandit(n_arms, n_features, v=v_value) for v_value in v_values
    })
    # Add classic bandits for comparison
    bandits.update({
        "ucb": UCB1Bandit(n_arms), # Classic UCB1 for comparison, the environment is still contextual
        "thompson": ThompsonSamplingBandit(n_arms), # Classic Thompson for comparison, the environment is still contextual
    })

    results = {}
    for name, bandit in bandits.items():
        rewards = run_bandit_on_contexts(name, bandit, contexts, env, optimal_rewards, verbose)
        print(f"[{name}] Total reward: {sum(rewards)}, Average rewards: {np.mean(rewards):.4f}")
        results[name] = rewards

    plot_multiple_cumulative_rewards(results)
    plot_multiple_regrets(results, optimal_rewards)
    if reward_type == "bernoulli":
        plot_multiple_running_ctr(results)

