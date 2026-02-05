"""
Equity-based state representation for poker decisions.

Follows the feature design from Bertsimas & Paskov (2022):
- Current equity
- Future equity deciles (per remaining street)
- Pot odds
- Stack-to-pot ratio
- Position (0=out of position, 1=in position)
- Betting history (numeric encoding of opponent actions per round)
"""

import numpy as np
from .equity_calculator import EquityCalculator


class StateRepresentation:
    """Builds interpretable feature vectors from poker game states."""

    # Betting round names
    ROUNDS = ["preflop", "flop", "turn", "river"]

    @staticmethod
    def build_feature_vector(hole_cards, board_cards, pot_size, bet_to_call,
                              stack_size, position, betting_history,
                              current_round, num_mc_samples=10000):
        """
        Build a full feature vector for the current decision point.

        Args:
            hole_cards: list of eval7.Card (player's 2 hole cards)
            board_cards: list of eval7.Card (0-5 community cards)
            pot_size: current pot size
            bet_to_call: amount needed to call
            stack_size: player's remaining stack
            position: 0 (OOP) or 1 (IP)
            betting_history: list of lists, opponent actions per round
            current_round: 0=preflop, 1=flop, 2=turn, 3=river

        Returns:
            numpy array: feature vector
        """
        features = []

        # Current equity
        equity = EquityCalculator.compute_equity(
            hole_cards, board_cards, num_samples=num_mc_samples
        )
        features.append(equity)

        # Future equity deciles for remaining streets
        if current_round == 0:  # Preflop
            # Future: flop (3 cards), turn (1 card), river (1 card)
            for future_count in [3, 1, 1]:
                dist = EquityCalculator.compute_future_equity_distribution(
                    hole_cards, board_cards,
                    future_cards_count=future_count,
                    num_board_samples=100,
                    num_equity_samples=200
                )
                features.extend(EquityCalculator.equity_deciles(dist))
        elif current_round == 1:  # Flop
            # Future: turn (1 card), river (1 card)
            for future_count in [1, 1]:
                dist = EquityCalculator.compute_future_equity_distribution(
                    hole_cards, board_cards,
                    future_cards_count=future_count,
                    num_board_samples=100,
                    num_equity_samples=200
                )
                features.extend(EquityCalculator.equity_deciles(dist))
        elif current_round == 2:  # Turn
            # Future: river (1 card)
            dist = EquityCalculator.compute_future_equity_distribution(
                hole_cards, board_cards,
                future_cards_count=1,
                num_board_samples=100,
                num_equity_samples=200
            )
            features.extend(EquityCalculator.equity_deciles(dist))
        # River: no future equity needed

        # Pot odds
        if pot_size > 0 and bet_to_call > 0:
            pot_odds = bet_to_call / (pot_size + bet_to_call)
        else:
            pot_odds = 0.0
        features.append(pot_odds)

        # Stack-to-pot ratio
        if pot_size > 0:
            spr = stack_size / pot_size
        else:
            spr = float(stack_size)
        features.append(spr)

        # Position
        features.append(float(position))

        # Betting history — 4 slots for opponent actions per round
        # Encode: 0=no action yet, 1=check/call, 2=bet/raise
        bet_hist = [0.0] * 4
        if betting_history:
            for i, actions in enumerate(betting_history[:4]):
                if actions:
                    bet_hist[i] = float(max(actions))
        features.extend(bet_hist)

        return np.array(features, dtype=np.float32)

    @staticmethod
    def build_simple_feature_vector(equity, pot_odds, stack_to_pot,
                                     position, betting_history=None):
        """
        Build a simplified feature vector (no future equity deciles).
        Useful for quick prototyping and testing.

        Args:
            equity: float in [0, 1]
            pot_odds: float in [0, 1]
            stack_to_pot: float >= 0
            position: 0 or 1
            betting_history: list of 4 floats

        Returns:
            numpy array: feature vector
        """
        features = [equity, pot_odds, stack_to_pot, float(position)]
        if betting_history:
            features.extend(betting_history[:4])
        else:
            features.extend([0.0] * 4)
        return np.array(features, dtype=np.float32)

    @staticmethod
    def get_feature_names(current_round):
        """Return feature names for the given betting round."""
        names = ["equity"]

        if current_round == 0:
            for street in ["flop", "turn", "river"]:
                names.extend([f"{street}_eq_d{i}" for i in range(1, 11)])
        elif current_round == 1:
            for street in ["turn", "river"]:
                names.extend([f"{street}_eq_d{i}" for i in range(1, 11)])
        elif current_round == 2:
            names.extend([f"river_eq_d{i}" for i in range(1, 11)])

        names.extend(["pot_odds", "stack_to_pot", "position"])
        names.extend([f"bet_history_{i}" for i in range(1, 5)])
        return names

    @staticmethod
    def get_simple_feature_names():
        """Return feature names for simplified representation."""
        return [
            "equity", "pot_odds", "stack_to_pot", "position",
            "bet_history_1", "bet_history_2", "bet_history_3", "bet_history_4"
        ]
