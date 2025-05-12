import argparse
from experiments import run_classic
from experiments import run_classic_comparison 
from experiments import run_contextual
from experiments import run_contextual_comparison
from experiments import run_combinatorial

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a bandit experiment.")
    parser.add_argument("--mode", type=str, choices=["classic", "contextual", "combinatorial"], default="classic",
                        help="Type of bandit experiment to run.")
    parser.add_argument("--algo", type=str, default="epsilon_greedy",
                        help="Which bandit algorithm to use (or comparison)")
    parser.add_argument("--n_arms", type=int, default=3,
                        help="Number of arms")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Exploration rate (only for epsilon-greedy)")
    parser.add_argument("--n_features", type=int, default=5,
                        help="Number of features (only for contextual bandits)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Exploration parameter (only for LinUCB)")
    parser.add_argument("--v", type=float, default=1.0,
                        help="Variance parameter (only for LinThompson)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (only for contextual epsilon-greedy)")
    parser.add_argument("--gap_strength", type=float, default=1,
                        help="How big of a gap between the arms (only for contextual comparison)")
    parser.add_argument("--noise_std", type=float, default=0.2,
                        help="Standard deviation of noise in rewards (only for contextual)")
    parser.add_argument("--reward_type", type=str, choices=["bernoulli", "gaussian"], default="gaussian",
                        help="Type of reward distribution (only for contextual bandits)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")
    parser.add_argument("--rounds", type=int, default=1000,
                        help="Number of rounds to simulate")
    parser.add_argument("--sweep", action="store_true",
                        help="Run a sweep of hyperparameters (only for contextual comparison)")
    parser.add_argument("--drift", action="store_true",
                        help="Enable drift in the environment (only for contextual bandits)")
    parser.add_argument("--time_dependent", action="store_true",
                        help="Enable time-dependent drift (only for contextual bandits)")
    parser.add_argument("--drift_rate", type=float, default=0.01,
                        help="Rate of drift in the environment (only for contextual bandits)")
    parser.add_argument("--drift_frequency", type=int, default=100,
                        help="Frequency of drift in the environment (only for contextual bandits)")
    parser.add_argument("--discount", type=float, default=1.0,
                        help="Discount parameters to adapt to drift (only for contextual bandits)")
    parser.add_argument("--k", type=int, default=2,
                        help="Number of arms to pull in combinatorial bandits (only for combinatorial bandits)")
    args = parser.parse_args()

    environment_context = {
        "noise_std": args.noise_std,
        "reward_type": args.reward_type,
        "gap_strength": args.gap_strength,
        "drift": args.drift,
        "time_dependent": args.time_dependent,
        "drift_rate": args.drift_rate,
        "drift_frequency": args.drift_frequency,
    }

    model_context = {
        "alpha": args.alpha,
        "v": args.v,
        "epsilon": args.epsilon,
        "lr": args.lr,
        "discount": args.discount,
    }

    if args.algo == "comparison":
        if args.mode == "classic":
            run_classic_comparison.run_all(rounds=args.rounds, n_arms=args.n_arms, verbose=args.verbose, sweep=args.sweep, **model_context)
        elif args.mode == "contextual":
            run_contextual_comparison.run_all(rounds=args.rounds, n_arms=args.n_arms, n_features=args.n_features,
                                              verbose=args.verbose, sweep=args.sweep,
                                              **environment_context, **model_context)
    else:
        if args.mode == "classic":
            run_classic.run(rounds=args.rounds, algo=args.algo, n_arms=args.n_arms, **model_context)
        elif args.mode == "contextual":
            run_contextual.run(rounds=args.rounds, algo=args.algo, n_arms=args.n_arms, n_features=args.n_features, **environment_context, **model_context)
        elif args.mode == "combinatorial":
            run_combinatorial.run(rounds=args.rounds, algo=args.algo, n_arms=args.n_arms, n_features=args.n_features, k=args.k, **environment_context, **model_context)