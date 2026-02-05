"""
Full training + evaluation pipeline.

Usage:
    python scripts/run_pipeline.py [--game holdem|leduc|kuhn] [--iterations N] [--depth N]
"""

import sys
import os
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Full XRL poker pipeline")
    parser.add_argument("--game", type=str, default="holdem",
                        choices=["holdem", "leduc", "kuhn"])
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=10000)
    args = parser.parse_args()

    # Step 1: Train CFR+
    run(f"python scripts/train_cfr.py --game {args.game} "
        f"--iterations {args.iterations} --checkpoint-every {max(1, args.iterations // 5)}")

    # Step 2: Generate training data (full traversal for exact coverage)
    run("python scripts/generate_data.py --traverse")

    # Step 3: Train decision tree
    run(f"python scripts/train_tree.py --depth {args.depth} --depth-search")

    # Step 4: Evaluate
    run(f"python scripts/evaluate.py --num-games {args.eval_games}")

    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")
    print("\nTo launch the UI:")
    print("  streamlit run src/ui/app.py")


if __name__ == "__main__":
    main()
