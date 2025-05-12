import numpy as np

from bandits.contextual.linucb import LinUCBBandit
from bandits.contextual.lin_thompson import LinThompsonBandit
from bandits.contextual.contextual_epsilon_greedy import ContextualEpsilonGreedyBandit
from bandits.contextual.logistic_bandit import LogisticBandit
from experiments.environments.contextual import ContextualBanditEnv
from experiments.utils.plot import plot_multiple_cumulative_rewards, plot_multiple_regrets, plot_multiple_running_average_rewards

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

def generate_bandit_specs(alpha_values, v_values, epsilon_values, lr_values, discount_values):
    bandit_specs = []

    # LinUCB variants
    for alpha in alpha_values:
        for discount in discount_values:
            bandit_specs.append({
                "name": f"linucb (alpha={alpha}, gamma={discount})",
                "cls": LinUCBBandit,
                "params": {"alpha": alpha, "discount": discount}
            })

    # LinThompson variants
    for v in v_values:
        for discount in discount_values:
            bandit_specs.append({
                "name": f"linthompson (v={v}, gamma={discount})",
                "cls": LinThompsonBandit,
                "params": {"v": v, "discount": discount}
            })

    # Epsilon-Greedy variants
    for epsilon in epsilon_values:
        for discount in discount_values:
            bandit_specs.append({
                "name": f"epsilon_greedy (epsilon={epsilon}, gamma={discount})",
                "cls": ContextualEpsilonGreedyBandit,
                "params": {"epsilon": epsilon, "lr": lr_values[0], "discount": discount}
            })

    # Logistic variants
    for lr in lr_values:
        for discount in discount_values:
            bandit_specs.append({
                "name": f"logistic (lr={lr}, gamma={discount})",
                "cls": LogisticBandit,
                "params": {"lr": lr, "discount": discount}
            })
    
    return bandit_specs

def run_all(rounds=2000, n_arms=3, n_features=5, verbose=False, sweep=False, **kwargs):
    alpha_values = [kwargs.get("alpha", 1.0)]
    v_values = [kwargs.get("v", 1.0)]
    epsilon_values = [kwargs.get("epsilon", 0.1)]
    lr_values = [kwargs.get("lr", 0.01)]
    discount_values = [kwargs.get("discount", 1.0)]

    if sweep:
        # alpha_values = [0.1, 0.5, 1.0, 2.0]
        # v_values = [0.1, 0.5, 1.0, 2.0]
        # epsilon_values = [0.01, 0.1, 0.2, 0.5]
        discount_values = [0.5, 0.9, 1.0]

    # Shared environment with fixed true weights
    env = ContextualBanditEnv(n_arms=n_arms, n_features=n_features, **kwargs)
    
    # Clone the environment’s true thetas so we can compute optimal rewards
    true_thetas = env.theta.copy()
    contexts = [env.get_context() for _ in range(rounds)]
    optimal_rewards = [max(x @ theta for theta in true_thetas) for x in contexts]


    bandit_specs = generate_bandit_specs(alpha_values, v_values, epsilon_values, lr_values, discount_values)
    bandits = {spec["name"]: spec["cls"](n_arms, n_features, **spec["params"]) for spec in bandit_specs}

    results = {}
    for name, bandit in bandits.items():
        rewards = run_bandit_on_contexts(name, bandit, contexts, env, optimal_rewards, verbose)
        print(f"[{name}] Total reward: {sum(rewards)}, Average rewards: {np.mean(rewards):.4f}")
        results[name] = rewards

    plot_multiple_cumulative_rewards(results)
    plot_multiple_regrets(results, optimal_rewards)
    plot_multiple_running_average_rewards(results)

