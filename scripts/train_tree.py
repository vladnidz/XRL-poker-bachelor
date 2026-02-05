"""
Script 3: Train CART decision tree on CFR+ generated data.

Usage:
    python scripts/train_tree.py [--depth N] [--depth-search]
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy_engine.decision_tree_policy import DecisionTreePolicy
from config import (
    MODEL_DIR, DATA_DIR, TREE_DEFAULT_DEPTH, TREE_MAX_DEPTHS,
    TREE_RANDOM_STATE, ACTION_NAMES
)


def main():
    parser = argparse.ArgumentParser(description="Train decision tree policy")
    parser.add_argument("--data", type=str,
                        default=os.path.join(DATA_DIR, "training_data.npz"),
                        help="Path to training data")
    parser.add_argument("--depth", type=int, default=TREE_DEFAULT_DEPTH,
                        help="Max tree depth")
    parser.add_argument("--depth-search", action="store_true",
                        help="Try multiple depths and compare")
    parser.add_argument("--output", type=str,
                        default=os.path.join(MODEL_DIR, "decision_tree.joblib"),
                        help="Output path for trained tree")
    args = parser.parse_args()

    # Load data
    print(f"Loading training data from {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    X, y = data["X"], data["y"]

    # Load feature names if saved
    if "feature_names" in data:
        feature_names = data["feature_names"].tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {feature_names}")
    print(f"Actions: {dict(zip(*np.unique(y, return_counts=True)))}")

    if args.depth_search:
        print("\n=== Depth Search ===")
        policy = DecisionTreePolicy(random_state=TREE_RANDOM_STATE)
        results = policy.depth_search(X, y, depths=TREE_MAX_DEPTHS,
                                       feature_names=feature_names)

        print("\n=== Summary ===")
        print(f"{'Depth':>6} {'Train Acc':>10} {'CV Acc':>10} {'Leaves':>8}")
        for r in results:
            print(f"{r['max_depth_param']:>6} "
                  f"{r['train_accuracy']:>10.4f} "
                  f"{r['cv_accuracy_mean']:>10.4f} "
                  f"{r['n_leaves']:>8}")

    # Train final model
    print(f"\n=== Training final tree (depth={args.depth}) ===")
    policy = DecisionTreePolicy(max_depth=args.depth,
                                 random_state=TREE_RANDOM_STATE)
    metrics = policy.train(X, y, feature_names=feature_names,
                           action_names=ACTION_NAMES)

    # Save
    policy.save(args.output)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
