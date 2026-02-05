"""Generate training data (state-action pairs) from a CFR+ strategy."""

import numpy as np
import pyspiel
from tqdm import tqdm


class DataGenerator:
    """
    Collects (feature_vector, action) pairs from a converged CFR+ strategy
    for supervised learning (policy distillation).
    """

    def __init__(self, game, average_policy, feature_builder=None):
        """
        Args:
            game: pyspiel.Game object
            average_policy: converged CFR+ average policy (TabularPolicy)
            feature_builder: callable(state, player) -> feature_vector
                             If None, uses info state tensor
        """
        self.game = game
        self.policy = average_policy
        self.feature_builder = feature_builder

    def generate(self, num_samples=100000):
        """
        Generate training data by sampling game plays from the CFR+ policy.
        At each decision point, record the state features and sample an action
        from the mixed strategy distribution.

        Args:
            num_samples: number of game rollouts to sample

        Returns:
            (X, y): numpy arrays of features and action labels
        """
        return self._generate_by_sampling(num_samples)

    def generate_by_traversal(self):
        """
        Traverse the full game tree and collect state-action pairs.
        For mixed strategies, samples action proportional to policy probs.
        Only feasible for small games (Kuhn, small poker variants).

        Returns:
            (X, y): numpy arrays of features and action labels
        """
        X = []
        y = []
        seen_info_states = {}

        def _traverse(state):
            if state.is_terminal():
                return

            if state.is_chance_node():
                for action, prob in state.chance_outcomes():
                    _traverse(state.child(action))
                return

            current_player = state.current_player()
            info_state = state.information_state_string(current_player)
            legal_actions = state.legal_actions(current_player)

            # Get policy probabilities using the State object
            action_probs = self.policy.action_probabilities(state)

            # Build feature vector
            if self.feature_builder:
                features = self.feature_builder(state, current_player)
            else:
                features = np.array(
                    state.information_state_tensor(current_player),
                    dtype=np.float32
                )

            # For each info state, add one sample per action weighted by prob
            if info_state not in seen_info_states:
                seen_info_states[info_state] = True
                for action in legal_actions:
                    prob = action_probs.get(action, 0.0)
                    if prob > 0.01:  # Only include actions with meaningful probability
                        # Add multiple copies proportional to probability
                        count = max(1, int(round(prob * 100)))
                        for _ in range(count):
                            X.append(features.copy())
                            y.append(action)

            # Recurse into children
            for action in legal_actions:
                _traverse(state.child(action))

        _traverse(self.game.new_initial_state())

        if not X:
            raise RuntimeError("No training data generated. Check game/policy.")

        return np.array(X), np.array(y)

    def _generate_by_sampling(self, num_samples):
        """Generate training data by sampling random game rollouts."""
        X = []
        y = []
        rng = np.random.default_rng(42)

        for _ in tqdm(range(num_samples), desc="Generating samples"):
            state = self.game.new_initial_state()

            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    actions, probs = zip(*outcomes)
                    action = rng.choice(actions, p=probs)
                    state.apply_action(action)
                    continue

                current_player = state.current_player()
                legal_actions = state.legal_actions(current_player)

                # Get policy from TabularPolicy using State object
                action_probs = self.policy.action_probabilities(state)

                # Build feature vector
                if self.feature_builder:
                    features = self.feature_builder(state, current_player)
                else:
                    features = np.array(
                        state.information_state_tensor(current_player),
                        dtype=np.float32
                    )

                # Sample action from the policy distribution
                probs_list = [action_probs.get(a, 0.0) for a in legal_actions]
                total = sum(probs_list)
                if total > 0:
                    probs_list = [p / total for p in probs_list]
                else:
                    probs_list = [1.0 / len(legal_actions)] * len(legal_actions)

                sampled_action = rng.choice(legal_actions, p=probs_list)

                # Record this decision point
                X.append(features)
                y.append(sampled_action)

                # Follow the sampled action
                state.apply_action(sampled_action)

        return np.array(X), np.array(y)
