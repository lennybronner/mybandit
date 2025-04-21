from bandits.classic.epsilon_greedy import EpsilonGreedyBandit
from bandits.classic.ucb import UCB1Bandit
from bandits.classic.thompson import ThompsonSamplingBandit
from experiments.environments.classic import ClassicBanditEnv
from experiments.utils.plot import plot_multiple_cumulative_rewards, plot_multiple_regrets

def run_all(rounds=10000):
    arm_probs = [0.2, 0.5, 0.8]
    envs = {algo: ClassicBanditEnv(arm_probs) for algo in ["epsilon", "ucb", "thompson"]}
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
            reward = envs[name].pull(arm)
            bandit.update(arm, reward)
            rewards.append(reward)
        results[name] = rewards

    plot_multiple_cumulative_rewards(results)
    plot_multiple_regrets(results, optimal_reward)
