"""
Exact equity calculator for Mini Heads-Up Limit Hold'em.

Game: reduced deck (configurable ranks x suits), 2 hole cards, 1 community card.
Hand evaluation: 3-card hands (2 hole + 1 board).
  - Straight (all 3 ranks consecutive) beats Pair.
  - Higher pair beats lower pair; same pair compare kickers.
  - Straight vs straight is always a tie (only one possible straight).

All equity calculations use exact enumeration (tractable for small decks).
"""

from itertools import combinations
from collections import Counter
import numpy as np

# ACPC notation helpers
ACPC_RANKS = "23456789TJQKA"
ACPC_SUITS = "cdhs"

# Display mapping: map low ACPC ranks to recognisable poker names
# For numRanks=3: 2->J, 3->Q, 4->K;  numRanks=4: 2->J, 3->Q, 4->K, 5->A
DISPLAY_RANKS = {0: "J", 1: "Q", 2: "K", 3: "A"}
DISPLAY_SUITS = {0: "\u2663", 1: "\u2666", 2: "\u2665", 3: "\u2660"}


# ---------------------------------------------------------------------------
# Card helpers
# ---------------------------------------------------------------------------

def parse_acpc_cards(card_str):
    """Parse ACPC card string like '4d4c' into list of (rank, suit) tuples."""
    cards = []
    for i in range(0, len(card_str), 2):
        rank_char = card_str[i]
        suit_char = card_str[i + 1]
        rank = ACPC_RANKS.index(rank_char)
        suit = ACPC_SUITS.index(suit_char)
        cards.append((rank, suit))
    return cards


def card_display(card, num_ranks=3):
    """Return human-readable card name, e.g. 'K♦'."""
    rank, suit = card
    return DISPLAY_RANKS.get(rank, str(rank)) + DISPLAY_SUITS.get(suit, "?")


def all_cards(num_ranks, num_suits):
    """Generate all cards in the deck as (rank, suit) tuples."""
    return [(r, s) for r in range(num_ranks) for s in range(num_suits)]


# ---------------------------------------------------------------------------
# Hand evaluation (3-card hands: 2 hole + 1 board)
# ---------------------------------------------------------------------------

def evaluate_hand(hole_cards, board_card):
    """
    Evaluate a 3-card poker hand (2 hole + 1 community).

    Returns a tuple that can be compared with > < == :
        (hand_type, tiebreaker1, tiebreaker2)
    where hand_type: 1 = straight, 0 = pair.

    With only 3 ranks in the game, 3 distinct ranks = straight (always 0-1-2).
    With more ranks, a straight requires 3 consecutive ranks.
    """
    ranks = sorted([hole_cards[0][0], hole_cards[1][0], board_card[0]])
    unique = set(ranks)

    if len(unique) == 3:
        # Check if consecutive
        if ranks[2] - ranks[0] == 2:
            # Straight — all straights tie (same set of ranks)
            return (1, ranks[2], 0)
        else:
            # Three different non-consecutive ranks → high card
            return (-1, ranks[2], ranks[1])
    elif len(unique) == 2:
        # Pair
        cnt = Counter(ranks)
        pair_rank = [r for r, c in cnt.items() if c == 2][0]
        kicker = [r for r, c in cnt.items() if c == 1][0]
        return (0, pair_rank, kicker)
    else:
        # Three of a kind (only possible with >= 3 suits per rank)
        return (2, ranks[0], 0)


def determine_winner(my_hole, opp_hole, board_card):
    """
    Returns: +1 if my hand wins, -1 if opponent wins, 0 if tie.
    """
    my_val = evaluate_hand(my_hole, board_card)
    opp_val = evaluate_hand(opp_hole, board_card)
    if my_val > opp_val:
        return 1
    elif my_val < opp_val:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Equity calculations
# ---------------------------------------------------------------------------

def compute_equity_preflop(my_hole, num_ranks=3, num_suits=2):
    """
    Exact preflop equity: enumerate all (opponent_hand, board_card) combos.

    Args:
        my_hole: tuple of 2 cards, each (rank, suit)
        num_ranks, num_suits: deck parameters

    Returns:
        float in [0, 1]
    """
    deck = all_cards(num_ranks, num_suits)
    remaining = [c for c in deck if c not in my_hole]

    wins = ties = total = 0

    for opp_hand in combinations(remaining, 2):
        leftover = [c for c in remaining if c not in opp_hand]
        for board_card in leftover:
            result = determine_winner(my_hole, opp_hand, board_card)
            if result == 1:
                wins += 1
            elif result == 0:
                ties += 1
            total += 1

    return (wins + 0.5 * ties) / total if total > 0 else 0.5


def compute_equity_postflop(my_hole, board_card, num_ranks=3, num_suits=2):
    """
    Exact postflop equity: enumerate all possible opponent hands.

    Args:
        my_hole: tuple of 2 cards
        board_card: single (rank, suit) card
        num_ranks, num_suits: deck parameters

    Returns:
        float in [0, 1]
    """
    deck = all_cards(num_ranks, num_suits)
    known = list(my_hole) + [board_card]
    remaining = [c for c in deck if c not in known]

    wins = ties = total = 0

    for opp_hand in combinations(remaining, 2):
        result = determine_winner(my_hole, opp_hand, board_card)
        if result == 1:
            wins += 1
        elif result == 0:
            ties += 1
        total += 1

    return (wins + 0.5 * ties) / total if total > 0 else 0.5


def compute_future_equity_distribution(my_hole, num_ranks=3, num_suits=2):
    """
    Preflop: for each possible board card, compute postflop equity.
    Returns sorted list of equity values.
    """
    deck = all_cards(num_ranks, num_suits)
    remaining = [c for c in deck if c not in my_hole]

    equities = []
    for board_card in remaining:
        eq = compute_equity_postflop(my_hole, board_card, num_ranks, num_suits)
        equities.append(eq)
    return sorted(equities)


def equity_deciles(distribution, n=10):
    """Convert an equity distribution into n quantile values."""
    if not distribution:
        return [0.5] * n
    arr = np.array(distribution)
    percentiles = np.linspace(100 / n, 100, n)
    return [float(np.percentile(arr, p)) for p in percentiles]
