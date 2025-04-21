import matplotlib.pyplot as plt
import numpy as np

def plot_cumulative_reward(rewards, title="Cumulative Reward"):
    cumulative = np.cumsum(rewards)
    plt.figure(figsize=(10, 4))
    plt.plot(cumulative, label="Cumulative Reward")
    plt.xlabel("Rounds")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_arm_selection(counts, title="Arm Selection Counts"):
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(counts)), counts)
    plt.xlabel("Arm")
    plt.ylabel("Number of Times Pulled")
    plt.title(title)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_running_average_reward(rewards, title="Running Averge Reward"):
    avg = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    plt.figure(figsize=(8, 4))
    plt.plot(avg)
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multiple_cumulative_rewards(results):
    plt.figure(figsize=(10, 5))
    for label, rewards in results.items():
        plt.plot(np.cumsum(rewards), label=label)
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multiple_regrets(results, optimal_reward):
    plt.figure(figsize=(10, 5))
    for label, rewards in results.items():
        regret = np.cumsum(np.maximum(0, optimal_reward - np.array(rewards)))
        plt.plot(regret, label=label)
    plt.xlabel("Rounds")
    plt.ylabel("Regret")
    plt.title("Regret Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_multiple_running_average_rewards(rewards):
    plt.figure(figsize=(10, 5))
    for label, reward in rewards.items():
        avg = np.cumsum(reward) / (np.arange(len(reward)) + 1)
        plt.plot(avg, label=label)
    plt.xlabel("Rounds")
    plt.ylabel("Average Reward")
    plt.title("Running Average Reward Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()