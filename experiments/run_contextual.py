from bandits.contextual.linucb import LinUCBBandit
from bandits.contextual.lin_thompson import LinThompsonBandit
from experiments.environments.contextual import ContextualBanditEnv
from experiments.utils.plot import plot_cumulative_reward, plot_arm_selection

def run(rounds=1000, algo='linucb', n_arms=3, n_features=5, **kwargs):
    env = ContextualBanditEnv(n_arms=n_arms, n_features=n_features)

    if algo == 'linucb':
        alpha = kwargs.get('alpha', 1.0)
        bandit = LinUCBBandit(n_arms=n_arms, n_features=n_features, alpha=alpha)
    elif algo == 'linthompson':
        v = kwargs.get('v', 1.0)
        bandit = LinThompsonBandit(n_arms=n_arms, n_features=n_features, v=v)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    rewards = []
    counts = [0] * n_arms

    for _ in range(rounds):
        context = env.get_context()
        arm = bandit.select_arm(context=context)
        reward = env.pull(arm, context)
        bandit.update(arm, reward, context=context)

        rewards.append(reward)
        counts[arm] += 1

    print(f"\n=== Contextual Bandit Results ===")
    print(f"Total reward: {sum(rewards)}")
    print(f"Average reward: {sum(rewards)/rounds:.4f}")
    print(f"Arm counts: {counts}")

    plot_cumulative_reward(rewards, title="Cumulative Reward")
    plot_arm_selection(counts, title="Arm Selection")