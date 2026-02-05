"""
Script 1: Train CFR+ strategy using OpenSpiel.

Usage:
    python scripts/train_cfr.py [--iterations N] [--game holdem|leduc|kuhn]
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspiel
from src.strategy_engine.cfr_trainer import CFRTrainer
from config import CFR_ITERATIONS, CFR_CHECKPOINT_EVERY, MODEL_DIR, MINI_HULH_GAME_STRING


def main():
    parser = argparse.ArgumentParser(description="Train CFR+ poker strategy")
    parser.add_argument("--iterations", type=int, default=CFR_ITERATIONS,
                        help="Number of CFR+ iterations")
    parser.add_argument("--game", type=str, default="holdem",
                        choices=["holdem", "leduc", "kuhn"],
                        help="Game variant")
    parser.add_argument("--checkpoint-every", type=int,
                        default=CFR_CHECKPOINT_EVERY)
    args = parser.parse_args()

    # Load game
    if args.game == "holdem":
        game = pyspiel.load_game(MINI_HULH_GAME_STRING)
        print("Loaded: Mini HULH (3 ranks x 2 suits, 2 hole cards, 1 board)")
    elif args.game == "leduc":
        game = pyspiel.load_game("leduc_poker")
        print("Loaded: Leduc Hold'em (6-card poker with community card)")
    else:
        game = pyspiel.load_game("kuhn_poker")
        print("Loaded: Kuhn Poker (3-card game)")

    # Train
    trainer = CFRTrainer(game)
    trainer.train(
        num_iterations=args.iterations,
        checkpoint_every=args.checkpoint_every,
        save_dir=MODEL_DIR,
    )

    print("\nCFR+ training complete.")
    print(f"Models saved to: {MODEL_DIR}/")


if __name__ == "__main__":
    main()
