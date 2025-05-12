from bandits.combinatorial.combinatorial_linucb import CombinatorialLinUCBBandit
from bandits.combinatorial.combinatorial_thompson import CombinatorialThompsonSamplingBandit
from experiments.environments.combinatorial import CombinatorialBanditEnv
from experiments.utils.plot import plot_cumulative_reward, plot_arm_selection, plot_running_average_reward

def run(rounds=1000, algo='linucb', n_arms=3, n_features=5, k=2, **kwargs):
    env = CombinatorialBanditEnv(n_arms=n_arms, n_features=n_features, k=k, **kwargs)

    if algo == 'linucb':
        alpha = kwargs.get('alpha', 1.0)
        bandit = CombinatorialLinUCBBandit(n_arms=n_arms, n_features=n_features, k=k, alpha=alpha)
    elif algo == 'thompson':
        v = kwargs.get('v', 1.0)
        bandit = CombinatorialThompsonSamplingBandit(n_arms=n_arms, n_features=n_features, k=k, v=v)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    all_rewards = []
    counts = [0] * n_arms

    for _ in range(rounds):
        context = env.get_context()
        arms = bandit.select_arm(context=context)
        rewards = env.pull(arms, context)
        bandit.update(arms, rewards, context=context)

        all_rewards.append(rewards)
        for arm in arms:
            counts[arm] += 1

    total_reward = sum([sum(rewards) for rewards in all_rewards])
    print(f"\n=== Combinatorial Bandit Results ===")
    print(f"Total reward: {total_reward}")
    print(f"Average reward: {total_reward/rounds:.4f}")
    print(f"Arm counts: {counts}")

    plot_cumulative_reward(all_rewards, title="Cumulative Reward")
    plot_arm_selection(counts, title="Arm Selection")
    plot_running_average_reward(all_rewards, title="Running CTR")