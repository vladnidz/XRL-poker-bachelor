"""
Equity-based feature builder for Leduc Hold'em.

Parses OpenSpiel Leduc game states and produces interpretable feature vectors
following the Bertsimas & Paskov (2022) methodology:
  - Current equity
  - Future equity quantiles (preflop only)
  - Pot odds
  - Position
  - Betting history
"""

import re
import numpy as np
from .leduc_equity import (
    compute_equity_preflop,
    compute_equity_postflop,
    compute_future_equity_distribution,
    equity_quantiles,
    rank_name,
)

# Number of future equity quantile features
N_QUANTILES = 5

FEATURE_NAMES = (
    ["equity"]
    + [f"future_eq_q{i+1}" for i in range(N_QUANTILES)]
    + ["pot_odds", "position", "opponent_action_r1", "opponent_action_r2"]
)


def get_feature_names():
    """Return list of feature names for the equity-based representation."""
    return list(FEATURE_NAMES)


def parse_leduc_state(state, player):
    """
    Extract game information from an OpenSpiel Leduc state.

    Returns dict with:
        my_card, board_card (or None), pot, money, round_num,
        position, round1_actions, round2_actions
    """
    info_str = state.information_state_string(player)

    # Parse private card: [Private: X]
    private_match = re.search(r'\[Private: (\d+)\]', info_str)
    my_card = int(private_match.group(1)) if private_match else None

    # Parse public card from game state (not info string, since info string
    # doesn't show the board card directly — we get it from the full state)
    state_str = str(state)

    # Cards line: "Cards (public player_0 player_1): BOARD P0 P1"
    cards_match = re.search(r'Cards.*?:\s*(-?\d+)\s+(\d+)\s+(\d+)', state_str)
    board_card = None
    if cards_match:
        board_val = int(cards_match.group(1))
        if board_val >= 0:  # -10000 means no board card yet
            board_card = board_val

    # Parse round
    round_match = re.search(r'\[Round (\d+)\]', info_str)
    round_num = int(round_match.group(1)) if round_match else 1

    # Parse pot
    pot_match = re.search(r'\[Pot: (\d+)\]', info_str)
    pot = int(pot_match.group(1)) if pot_match else 2

    # Parse money
    money_match = re.search(r'\[Money: (\d+) (\d+)\]', info_str)
    if money_match:
        money = [int(money_match.group(1)), int(money_match.group(2))]
    else:
        money = [99, 99]

    # Parse player (position)
    player_match = re.search(r'\[Player: (\d+)\]', info_str)
    acting_player = int(player_match.group(1)) if player_match else 0

    # Position: 0 = first to act (OOP), 1 = second to act (IP)
    position = 0 if acting_player == player else 1

    # Parse round actions: [Round1: X X ...] [Round2: X X ...]
    r1_match = re.search(r'\[Round1: ([\d ]*)\]', info_str)
    r2_match = re.search(r'\[Round2: ([\d ]*)\]', info_str)

    r1_actions = []
    if r1_match and r1_match.group(1).strip():
        r1_actions = [int(x) for x in r1_match.group(1).strip().split()]

    r2_actions = []
    if r2_match and r2_match.group(1).strip():
        r2_actions = [int(x) for x in r2_match.group(1).strip().split()]

    return {
        "my_card": my_card,
        "board_card": board_card,
        "pot": pot,
        "money": money,
        "round_num": round_num,
        "position": position,
        "round1_actions": r1_actions,
        "round2_actions": r2_actions,
    }


def opponent_aggression(actions):
    """
    Encode opponent's betting actions as a single numeric value.
    0 = no actions, 1 = passive (check/call only), 2 = aggressive (raised)
    """
    if not actions:
        return 0.0
    if any(a == 2 for a in actions):
        return 2.0  # Raised
    return 1.0  # Check/call only


def build_features(state, player):
    """
    Build an equity-based feature vector from a Leduc game state.

    Args:
        state: OpenSpiel State object
        player: player ID (0 or 1)

    Returns:
        numpy array of shape (10,) with interpretable features
    """
    info = parse_leduc_state(state, player)
    features = []

    # 1. Current equity
    if info["board_card"] is not None:
        equity = compute_equity_postflop(info["my_card"], info["board_card"])
    else:
        equity = compute_equity_preflop(info["my_card"])
    features.append(equity)

    # 2-6. Future equity quantiles (preflop only)
    if info["board_card"] is None:
        future_dist = compute_future_equity_distribution(info["my_card"])
        quantiles = equity_quantiles(future_dist, N_QUANTILES)
    else:
        quantiles = [0.0] * N_QUANTILES
    features.extend(quantiles)

    # 7. Pot odds
    # In Leduc: raise costs 2 in round 1, 4 in round 2
    # If facing a raise, bet_to_call is the raise amount
    legal = state.legal_actions(player)
    # Estimate bet to call from pot and money changes
    total_pot = info["pot"]
    if total_pot > 0:
        # Simplified: in limit poker, call cost is fixed per round
        raise_size = 2 if info["round_num"] == 1 else 4
        # Check if opponent raised (last action was 2)
        last_actions = info["round2_actions"] if info["round_num"] == 2 else info["round1_actions"]
        if last_actions and last_actions[-1] == 2:
            bet_to_call = raise_size
        else:
            bet_to_call = 0
        pot_odds = bet_to_call / (total_pot + bet_to_call) if (total_pot + bet_to_call) > 0 else 0.0
    else:
        pot_odds = 0.0
    features.append(pot_odds)

    # 8. Position
    features.append(float(info["position"]))

    # 9-10. Opponent betting history per round
    features.append(opponent_aggression(info["round1_actions"]))
    features.append(opponent_aggression(info["round2_actions"]))

    return np.array(features, dtype=np.float32)
