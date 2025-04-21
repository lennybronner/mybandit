from bandits.classic.epsilon_greedy import EpsilonGreedyBandit
from bandits.classic.ucb import UCB1Bandit
from bandits.classic.thompson import ThompsonSamplingBandit
from experiments.environments.classic import ClassicBanditEnv
from experiments.utils.plot import plot_multiple_cumulative_rewards, plot_multiple_regrets, plot_multiple_running_ctr

import numpy as np

def run_all(rounds=10000, n_arms=3, verbose=False):
    arm_probs = np.random.uniform(0.3, 0.9, size=n_arms).round(2)
    print(f"Arm probabilities: {arm_probs}")
    env = ClassicBanditEnv(arm_probs)
    optimal_reward = max(arm_probs)

    results = {}

    bandits = {
        "epsilon": EpsilonGreedyBandit(len(arm_probs), epsilon=0.1),
        "ucb": UCB1Bandit(len(arm_probs)),
        "thompson": ThompsonSamplingBandit(len(arm_probs)),
    }

    for name, bandit in bandits.items():
        rewards = []
        for t in range(rounds):
            arm = bandit.select_arm()
            reward = env.pull(arm)
            bandit.update(arm, reward)
            if verbose:
                print(f"[{name}] Round {t}: arm={arm}, reward={reward:.2f}")
            rewards.append(reward)
        results[name] = rewards

    plot_multiple_cumulative_rewards(results)
    plot_multiple_regrets(results, optimal_reward)
    plot_multiple_running_ctr(results)
