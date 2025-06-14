from bandits.classic.epsilon_greedy import EpsilonGreedyBandit
from bandits.classic.ucb import UCB1Bandit
from bandits.classic.thompson import ThompsonSamplingBandit
from experiments.environments.classic import ClassicBanditEnv
from experiments.utils.plot import plot_cumulative_reward, plot_arm_selection, plot_running_average_reward, plot_arm_selection_over_time

import numpy as np

def run(rounds=1000, n_arms=3, algo="epsilon_greedy", **kwargs):
    """
    Run the Epsilon-Greedy Bandit algorithm on a classic bandit environment.

    Parameters:
    - n_arms: Number of arms (default is 3).
    - rounds: Number of rounds to run (default is 1000).
    - epsilon: Probability of exploration (default is 0.1).
    """
    arm_probs = np.random.uniform(0.3, 0.9, size=n_arms).round(2)

    env = ClassicBanditEnv(arm_probs)
    if algo == "epsilon_greedy":
        epsilon = kwargs.get("epsilon", 0.1)
        bandit = EpsilonGreedyBandit(n_arms=n_arms, epsilon=epsilon)
    elif algo == "ucb":
        bandit = UCB1Bandit(n_arms=n_arms)
    elif algo == "thompson":
        bandit = ThompsonSamplingBandit(n_arms=n_arms)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    rewards = []
    arm_counts_over_time = np.zeros((n_arms, rounds))

    for t in range(rounds):
        arm = bandit.select_arm()
        reward = env.pull(arm)
        bandit.update(arm, reward)
        rewards.append(reward)
        arm_counts_over_time[arm, t] = 1

    print("Arm Probabilities", arm_probs)
    print(f"\n=== Classic Bandit Results ===")
    print("Estimated values:", bandit.values)
    print("Arm counts:", bandit.counts)
    print("Total reward:", sum(rewards))
    print(f"Average reward: {sum(rewards) / rounds:.4f}")

    plot_cumulative_reward(rewards, title=f"Cumulative Reward - {algo.capitalize()}")
    plot_arm_selection(bandit.counts, title=f"Arm Selection - {algo.capitalize()}")
    plot_running_average_reward(rewards, title=f"Running Average Reward - {algo.capitalize()}")
    plot_arm_selection_over_time(arm_counts_over_time, title=f"Arm Selection Over Time - {algo.capitalize()}", rate=False)
    plot_arm_selection_over_time(arm_counts_over_time, title=f"Arm Selection Over Time - {algo.capitalize()}", rate=True)