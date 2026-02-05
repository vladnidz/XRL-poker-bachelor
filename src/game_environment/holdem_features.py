"""
Equity-based feature builder for Mini Heads-Up Limit Hold'em.

Parses OpenSpiel universal_poker game states and produces interpretable
feature vectors following Bertsimas & Paskov (2022) methodology:
  - Current equity
  - Future equity deciles (preflop only)
  - Pot odds
  - Stack-to-pot ratio
  - Position
  - Betting history (per-round opponent aggression)
"""

import re
import numpy as np
from .holdem_equity import (
    parse_acpc_cards,
    compute_equity_preflop,
    compute_equity_postflop,
    compute_future_equity_distribution,
    equity_deciles,
    card_display,
)

# ---- Configuration (must match the game loaded in config.py) ----
NUM_RANKS = 3
NUM_SUITS = 2
N_DECILES = 10

FEATURE_NAMES = (
    ["equity"]
    + [f"future_eq_d{i+1}" for i in range(N_DECILES)]
    + ["pot_odds", "stack_to_pot", "position"]
    + ["opp_aggression_r1", "opp_aggression_r2"]
    + ["has_pocket_pair", "max_hole_rank", "board_pairs_hole"]
)


def get_feature_names():
    """Return list of feature names for the equity-based representation."""
    return list(FEATURE_NAMES)


# ---- State parsing ----

def parse_holdem_state(state, player):
    """
    Extract game information from an OpenSpiel universal_poker state.

    Info state format (example):
        [Round 0][Player: 0][Pot: 4][Money: 2147483646 2147483645]
        [Private: 4d4c][Public: ][Sequences: ]

    Postflop:
        [Round 1][Player: 0][Pot: 8][Money: ...]
        [Private: 4d4c][Public: 2d][Sequences: cc|]

    Returns dict with parsed game information.
    """
    info_str = state.information_state_string(player)

    # Round
    round_match = re.search(r'\[Round (\d+)\]', info_str)
    round_num = int(round_match.group(1)) if round_match else 0

    # Player currently acting
    player_match = re.search(r'\[Player: (\d+)\]', info_str)
    acting_player = int(player_match.group(1)) if player_match else 0

    # Pot
    pot_match = re.search(r'\[Pot: (\d+)\]', info_str)
    pot = int(pot_match.group(1)) if pot_match else 3

    # Money
    money_match = re.search(r'\[Money: (\d+) (\d+)\]', info_str)
    if money_match:
        money = [int(money_match.group(1)), int(money_match.group(2))]
    else:
        money = [1000, 1000]

    # Private cards (ACPC format, e.g. "4d4c")
    private_match = re.search(r'\[Private: ([^\]]+)\]', info_str)
    hole_cards = ()
    if private_match and private_match.group(1).strip():
        hole_cards = tuple(parse_acpc_cards(private_match.group(1).strip()))

    # Public cards (board)
    public_match = re.search(r'\[Public: ([^\]]*)\]', info_str)
    board_cards = []
    if public_match and public_match.group(1).strip():
        board_cards = parse_acpc_cards(public_match.group(1).strip())

    # Sequences (betting history): e.g. "cc|" or "crc|cb"
    seq_match = re.search(r'\[Sequences: ([^\]]*)\]', info_str)
    sequences_str = seq_match.group(1).strip() if seq_match else ""
    rounds_seq = sequences_str.split("|") if sequences_str else []

    # Position: 0 = OOP (first to act), 1 = IP (second to act)
    # In our config firstPlayer=1 (ACPC 1-indexed) = player 0 acts first
    position = 0 if acting_player == player else 1

    return {
        "hole_cards": hole_cards,
        "board_cards": board_cards,
        "round_num": round_num,
        "pot": pot,
        "money": money,
        "position": position,
        "acting_player": acting_player,
        "round_sequences": rounds_seq,
    }


# ---- Opponent aggression encoding ----

def opponent_aggression(sequence_str, player, round_idx, first_player=0):
    """
    Encode opponent's aggression from a round's action sequence.

    Actions alternate starting with first_player for that round.
    Returns: 0.0 = no actions, 1.0 = passive only, 2.0 = raised at least once.
    """
    if not sequence_str:
        return 0.0

    # Determine which positions in the sequence are opponent's
    opp_actions = []
    for i, ch in enumerate(sequence_str):
        # Actions alternate between players
        action_player = first_player if i % 2 == 0 else (1 - first_player)
        if action_player != player:
            opp_actions.append(ch)

    if not opp_actions:
        return 0.0
    if any(a == 'r' for a in opp_actions):
        return 2.0  # Aggressive (raised)
    return 1.0  # Passive (check/call only)


# ---- Feature builder ----

def build_features(state, player):
    """
    Build an equity-based feature vector from a universal_poker game state.

    Args:
        state: OpenSpiel State object
        player: player ID (0 or 1)

    Returns:
        numpy array of shape (16,) with interpretable features
    """
    info = parse_holdem_state(state, player)
    features = []

    # 1. Current equity
    if info["board_cards"]:
        board = info["board_cards"][0]
        equity = compute_equity_postflop(
            info["hole_cards"], board, NUM_RANKS, NUM_SUITS
        )
    else:
        equity = compute_equity_preflop(
            info["hole_cards"], NUM_RANKS, NUM_SUITS
        )
    features.append(equity)

    # 2-11. Future equity deciles (preflop only, zeros on postflop)
    if not info["board_cards"]:
        future_dist = compute_future_equity_distribution(
            info["hole_cards"], NUM_RANKS, NUM_SUITS
        )
        deciles = equity_deciles(future_dist, N_DECILES)
    else:
        deciles = [0.0] * N_DECILES
    features.extend(deciles)

    # 12. Pot odds
    total_pot = info["pot"]
    # In limit poker, call cost is the raise size for the current round
    raise_size = 2 if info["round_num"] == 0 else 4
    # Check if facing a bet: look at last action in current round's sequence
    current_seq = ""
    if info["round_sequences"]:
        idx = min(info["round_num"], len(info["round_sequences"]) - 1)
        current_seq = info["round_sequences"][idx] if idx < len(info["round_sequences"]) else ""
    if current_seq and current_seq[-1] == 'r':
        bet_to_call = raise_size
    else:
        bet_to_call = 0
    pot_odds = bet_to_call / (total_pot + bet_to_call) if (total_pot + bet_to_call) > 0 else 0.0
    features.append(pot_odds)

    # 13. Stack-to-pot ratio
    min_stack = min(info["money"])
    stack_to_pot = min_stack / total_pot if total_pot > 0 else 100.0
    # Cap for numerical stability (stacks are very large in universal_poker)
    stack_to_pot = min(stack_to_pot, 100.0)
    features.append(stack_to_pot)

    # 14. Position
    features.append(float(info["position"]))

    # 15-16. Opponent aggression per round
    for r in range(2):
        seq = info["round_sequences"][r] if r < len(info["round_sequences"]) else ""
        agg = opponent_aggression(seq, player, r, first_player=0)
        features.append(agg)

    # 17. Has pocket pair (both hole cards same rank)
    if len(info["hole_cards"]) == 2:
        has_pair = 1.0 if info["hole_cards"][0][0] == info["hole_cards"][1][0] else 0.0
    else:
        has_pair = 0.0
    features.append(has_pair)

    # 18. Max hole card rank (normalized to [0, 1])
    if info["hole_cards"]:
        max_rank = max(c[0] for c in info["hole_cards"])
        features.append(max_rank / max(NUM_RANKS - 1, 1))
    else:
        features.append(0.0)

    # 19. Board pairs one of my hole cards
    if info["board_cards"] and info["hole_cards"]:
        board_rank = info["board_cards"][0][0]
        hole_ranks = [c[0] for c in info["hole_cards"]]
        features.append(1.0 if board_rank in hole_ranks else 0.0)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32)
