"""MCCFR trainer using OpenSpiel's external sampling implementation.

Monte Carlo CFR samples the game tree instead of traversing it fully,
making it memory-efficient for large games (e.g., 24-card Hold'em).
"""

import os
import pickle
import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr as mccfr
from tqdm import tqdm


class CFRTrainer:
    """Trains a strategy via Monte Carlo CFR (external sampling)."""

    def __init__(self, game):
        self.game = game
        self.solver = None
        self.iterations_done = 0

    def train(self, num_iterations=200000, checkpoint_every=50000,
              save_dir="models"):
        os.makedirs(save_dir, exist_ok=True)

        if self.solver is None:
            self.solver = mccfr.ExternalSamplingSolver(self.game)

        print(f"Starting MCCFR training for {num_iterations} iterations...")

        for i in tqdm(range(1, num_iterations + 1)):
            self.solver.iteration()
            self.iterations_done += 1

            if i % checkpoint_every == 0:
                print(f"  Iteration {i}: checkpoint saved")
                path = os.path.join(save_dir, f"mccfr_checkpoint_{i}.pkl")
                self.save(path)

        final_path = os.path.join(save_dir, "cfr_final.pkl")
        self.save(final_path)
        print(f"Training complete. Final model saved to {final_path}")

        return self.get_average_policy()

    def get_average_policy(self):
        if self.solver is None:
            raise RuntimeError("Solver not initialized. Call train() first.")
        return self.solver.average_policy()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'solver': self.solver,
                'game': self.game,
                'iterations_done': self.iterations_done,
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.solver = data['solver']
        self.iterations_done = data['iterations_done']
        print(f"Loaded MCCFR checkpoint: {self.iterations_done} iterations")
