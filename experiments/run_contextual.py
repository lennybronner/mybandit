from bandits.contextual.linucb import LinUCBBandit
from bandits.contextual.lin_thompson import LinThompsonBandit
from bandits.contextual.contextual_epsilon_greedy import ContextualEpsilonGreedyBandit
from bandits.contextual.logistic_bandit import LogisticBandit
from experiments.environments.contextual import ContextualBanditEnv
from experiments.utils.plot import plot_cumulative_reward, plot_arm_selection, plot_running_ctr

def run(rounds=1000, algo='linucb', n_arms=3, n_features=5, noise_std=0.2, reward_type="gaussian", **kwargs):
    env = ContextualBanditEnv(n_arms=n_arms, n_features=n_features, noise_std=noise_std, reward_type=reward_type)

    if algo == 'linucb':
        alpha = kwargs.get('alpha', 1.0)
        bandit = LinUCBBandit(n_arms=n_arms, n_features=n_features, alpha=alpha)
    elif algo == 'linthompson':
        v = kwargs.get('v', 1.0)
        bandit = LinThompsonBandit(n_arms=n_arms, n_features=n_features, v=v)
    elif algo == 'epsilon_greedy':
        epsilon = kwargs.get('epsilon', 0.1)
        lr = kwargs.get('lr', 0.01)
        bandit = ContextualEpsilonGreedyBandit(n_arms=n_arms, n_features=n_features, epsilon=epsilon, lr=lr)
    elif algo == "logistic":
        lr = kwargs.get('lr', 0.01)
        bandit = LogisticBandit(n_arms=n_arms, n_features=n_features, lr=lr)
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
    if reward_type == "bernoulli":
        plot_running_ctr(rewards, title="Running CTR")