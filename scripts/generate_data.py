"""
Script 2: Generate training data from the converged MCCFR strategy.

Uses equity-based features with hand bucketing for interpretability.

Usage:
    python scripts/generate_data.py [--cfr-model PATH] [--samples N] [--traverse]
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy_engine.data_generator import DataGenerator
from src.game_environment.holdem_features import build_features, get_feature_names
from config import MODEL_DIR, DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Generate training data from MCCFR")
    parser.add_argument("--cfr-model", type=str,
                        default=os.path.join(MODEL_DIR, "cfr_final.pkl"),
                        help="Path to MCCFR checkpoint")
    parser.add_argument("--samples", type=int, default=100000,
                        help="Number of game rollouts to sample")
    parser.add_argument("--traverse", action="store_true",
                        help="Use full game tree traversal instead of sampling")
    parser.add_argument("--output", type=str,
                        default=os.path.join(DATA_DIR, "training_data.npz"),
                        help="Output path for training data")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Loading MCCFR model from {args.cfr_model}...")
    import pickle
    with open(args.cfr_model, 'rb') as f:
        data = pickle.load(f)

    solver = data['solver']
    # Support both old and new save formats
    if 'game' in data:
        game = data['game']
    elif 'game_string' in data:
        import pyspiel
        game = pyspiel.load_game(data['game_string'])
    else:
        game = solver._game
    avg_policy = solver.average_policy()

    print(f"Game: {game}")
    print(f"MCCFR iterations: {data['iterations_done']}")

    feature_names = get_feature_names()
    print(f"Features ({len(feature_names)}): {feature_names}")

    generator = DataGenerator(
        game, avg_policy,
        feature_builder=build_features
    )

    if args.traverse:
        print("Generating data via full game tree traversal...")
        X, y = generator.generate_by_traversal()
    else:
        print(f"Generating {args.samples} samples with equity-based features...")
        X, y = generator.generate(num_samples=args.samples)

    print(f"Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Action distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    np.savez(args.output, X=X, y=y, feature_names=np.array(feature_names))
    print(f"Training data saved to {args.output}")


if __name__ == "__main__":
    main()
