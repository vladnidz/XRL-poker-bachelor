"""CFR+ trainer using OpenSpiel's built-in implementation."""

import os
import pickle
import pyspiel
from open_spiel.python.algorithms.cfr import CFRPlusSolver as _CFRPlusSolver
from tqdm import tqdm


class CFRTrainer:
    """Trains a CFR+ strategy for a given OpenSpiel game."""

    def __init__(self, game):
        """
        Args:
            game: pyspiel.Game object
        """
        self.game = game
        self.solver = None
        self.iterations_done = 0

    def train(self, num_iterations=100000, checkpoint_every=10000,
              save_dir="models"):
        """
        Run CFR+ for the specified number of iterations.

        Args:
            num_iterations: total training iterations
            checkpoint_every: save checkpoint interval
            save_dir: directory to save checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)

        if self.solver is None:
            self.solver = _CFRPlusSolver(self.game)

        print(f"Starting CFR+ training for {num_iterations} iterations...")

        for i in tqdm(range(1, num_iterations + 1)):
            self.solver.evaluate_and_update_policy()
            self.iterations_done += 1

            if i % checkpoint_every == 0:
                exploitability = self._compute_exploitability()
                print(f"  Iteration {i}: exploitability = {exploitability:.6f}")

                path = os.path.join(save_dir, f"cfr_checkpoint_{i}.pkl")
                self.save(path)

        # Save final
        final_path = os.path.join(save_dir, "cfr_final.pkl")
        self.save(final_path)
        print(f"Training complete. Final model saved to {final_path}")

        return self.get_average_policy()

    def _compute_exploitability(self):
        """Compute Nash distance (exploitability) of current average policy.
        Skipped for universal_poker (too expensive); only computed for small games.
        """
        game_type = self.game.get_type()
        # universal_poker exploitability is extremely slow; skip it
        if "universal_poker" in str(self.game):
            return float("nan")
        try:
            from open_spiel.python.algorithms import exploitability
            avg_policy = self.get_average_policy()
            return exploitability.exploitability(self.game, avg_policy)
        except Exception:
            return float("nan")

    def get_average_policy(self):
        """Return the average policy (converged strategy)."""
        if self.solver is None:
            raise RuntimeError("Solver not initialized. Call train() first.")
        return self.solver.average_policy()

    def get_action_probabilities(self, info_state_string, legal_actions):
        """
        Get action probabilities from the average policy for a given state.

        Args:
            info_state_string: information state string
            legal_actions: list of legal action indices

        Returns:
            dict: {action: probability}
        """
        avg_policy = self.get_average_policy()
        policy_dict = dict(avg_policy.action_probability_array)

        probs = {}
        for action in legal_actions:
            probs[action] = policy_dict.get(action, 0.0)

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {a: p / total for a, p in probs.items()}
        else:
            uniform = 1.0 / len(legal_actions)
            probs = {a: uniform for a in legal_actions}

        return probs

    def save(self, path):
        """Save trainer state to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'solver': self.solver,
                'iterations_done': self.iterations_done,
            }, f)

    def load(self, path):
        """Load trainer state from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.solver = data['solver']
        self.iterations_done = data['iterations_done']
        print(f"Loaded CFR+ checkpoint: {self.iterations_done} iterations")
