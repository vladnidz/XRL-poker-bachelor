"""
Script 1: Train MCCFR strategy using OpenSpiel.

Usage:
    python scripts/train_cfr.py [--iterations N]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspiel
from src.strategy_engine.cfr_trainer import CFRTrainer
from config import MCCFR_ITERATIONS, MCCFR_CHECKPOINT_EVERY, MODEL_DIR, GAME_STRING


def main():
    parser = argparse.ArgumentParser(description="Train MCCFR poker strategy")
    parser.add_argument("--iterations", type=int, default=MCCFR_ITERATIONS)
    parser.add_argument("--checkpoint-every", type=int,
                        default=MCCFR_CHECKPOINT_EVERY)
    args = parser.parse_args()

    game = pyspiel.load_game(GAME_STRING)
    print(f"Loaded: {game}")

    trainer = CFRTrainer(game)
    trainer.train(
        num_iterations=args.iterations,
        checkpoint_every=args.checkpoint_every,
        save_dir=MODEL_DIR,
    )

    print("\nMCCFR training complete.")
    print(f"Models saved to: {MODEL_DIR}/")


if __name__ == "__main__":
    main()
