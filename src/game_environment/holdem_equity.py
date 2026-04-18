"""
Equity calculator and hand bucketing for reduced-deck Hold'em.

Uses exact enumeration for equity on a reduced deck (configurable ranks x suits).
Hand buckets follow standard poker categorisation for interpretable features.
"""

import numpy as np
from itertools import combinations

# ACPC notation helpers
ACPC_RANKS = "23456789TJQKA"
ACPC_SUITS = "cdhs"

# Display mapping for 6-rank and 3-rank decks
RANK_NAMES_6 = {0: "9", 1: "10", 2: "J", 3: "Q", 4: "K", 5: "A"}
RANK_NAMES_3 = {0: "J", 1: "Q", 2: "K"}

DISPLAY_RANKS = RANK_NAMES_6
DISPLAY_SUITS = {0: "\u2663", 1: "\u2666", 2: "\u2665", 3: "\u2660"}

SUIT_NAMES = {0: "c", 1: "d", 2: "h", 3: "s"}


def card_display(rank, suit, num_ranks=6):
    """Human-readable card string."""
    rmap = RANK_NAMES_6 if num_ranks >= 6 else RANK_NAMES_3
    return rmap.get(rank, str(rank)) + DISPLAY_SUITS.get(suit, "?")


def parse_acpc_cards(acpc_str):
    """
    Parse ACPC card notation (e.g. '4d4c' or 'Kh9s') into (rank, suit) tuples.
    In universal_poker, ranks use ACPC_RANKS indexing offset by the game's min rank.
    """
    cards = []
    i = 0
    while i < len(acpc_str):
        rank_ch = acpc_str[i]
        suit_ch = acpc_str[i + 1]
        rank_idx = ACPC_RANKS.index(rank_ch)
        suit_idx = ACPC_SUITS.index(suit_ch)
        cards.append((rank_idx, suit_idx))
        i += 2
    return cards


def all_cards(num_ranks, num_suits):
    """Generate all (rank, suit) tuples for the deck."""
    return [(r, s) for r in range(num_ranks) for s in range(num_suits)]


# ---------------------------------------------------------------------------
# Hand comparison (3-card hands: 2 hole + 1 board)
# ---------------------------------------------------------------------------

def hand_strength(hole_cards, board_cards, num_ranks):
    """
    Evaluate a 3-card hand (2 hole + 1 board).
    Returns (category, tiebreakers) where higher = better.
    Categories: 3=straight, 2=pair, 1=high card
    """
    ranks = sorted([hole_cards[0][0], hole_cards[1][0], board_cards[0][0]],
                   reverse=True)
    unique_ranks = set(ranks)

    # Three of a kind
    if len(unique_ranks) == 1:
        return (4, ranks)

    # Straight: all 3 ranks consecutive
    if len(unique_ranks) == 3 and ranks[0] - ranks[2] == 2:
        return (3, ranks)

    # Pair
    if len(unique_ranks) == 2:
        from collections import Counter
        cnt = Counter(ranks)
        pair_rank = [r for r, c in cnt.items() if c == 2][0]
        kicker = [r for r, c in cnt.items() if c == 1][0]
        return (2, [pair_rank, kicker])

    # High card
    return (1, ranks)


def compare_hands(hero_hole, opp_hole, board, num_ranks):
    """Compare two hands. Returns +1 hero wins, -1 opp wins, 0 tie."""
    hero_str = hand_strength(hero_hole, board, num_ranks)
    opp_str = hand_strength(opp_hole, board, num_ranks)
    if hero_str > opp_str:
        return 1
    elif hero_str < opp_str:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Equity calculations
# ---------------------------------------------------------------------------

def compute_equity_preflop(hole_cards, num_ranks=6, num_suits=4):
    """Preflop equity via enumeration of all opponent + board combos."""
    deck = all_cards(num_ranks, num_suits)
    known = set(hole_cards)
    remaining = [c for c in deck if c not in known]

    wins = ties = total = 0
    for opp in combinations(remaining, 2):
        leftover = [c for c in remaining if c not in opp]
        for board in leftover:
            result = compare_hands(hole_cards, opp, [board], num_ranks)
            if result > 0:
                wins += 1
            elif result == 0:
                ties += 1
            total += 1

    return (wins + 0.5 * ties) / total if total > 0 else 0.5


def compute_equity_postflop(hole_cards, board_card, num_ranks=6, num_suits=4):
    """Postflop equity via enumeration of all opponent hands."""
    deck = all_cards(num_ranks, num_suits)
    known = set(hole_cards) | {board_card}
    remaining = [c for c in deck if c not in known]

    wins = ties = total = 0
    for opp in combinations(remaining, 2):
        result = compare_hands(hole_cards, opp, [board_card], num_ranks)
        if result > 0:
            wins += 1
        elif result == 0:
            ties += 1
        total += 1

    return (wins + 0.5 * ties) / total if total > 0 else 0.5


def compute_future_equity_distribution(hole_cards, num_ranks=6, num_suits=4):
    """Compute equity for each possible board card. Returns sorted list."""
    deck = all_cards(num_ranks, num_suits)
    remaining = [c for c in deck if c not in hole_cards]
    equities = []
    for board_card in remaining:
        eq = compute_equity_postflop(hole_cards, board_card, num_ranks, num_suits)
        equities.append(eq)
    return sorted(equities)


def equity_deciles(distribution, n_deciles=10):
    """Convert equity distribution into decile values."""
    if not distribution:
        return [0.5] * n_deciles
    arr = np.array(distribution)
    percentiles = np.linspace(100 / n_deciles, 100, n_deciles)
    return [float(np.percentile(arr, p)) for p in percentiles]


# ---------------------------------------------------------------------------
# Hand bucketing (postflop categories)
# ---------------------------------------------------------------------------

HAND_BUCKET_VALUES = {
    "set": 8,
    "overpair": 7,
    "top_pair_top_kicker": 6,
    "top_pair_weak_kicker": 5,
    "middle_pair": 4,
    "underpair": 3,
    "overcards": 2,
    "air": 1,
    "pocket_pair": 5,
    "connectors": 3,
    "high_card": 2,
}


def classify_hand(hole_cards, board_cards, num_ranks=6):
    """
    Classify a hand into a standard poker bucket.

    Returns:
        (bucket_name: str, bucket_value: int)
    """
    if not board_cards:
        # Preflop buckets
        r0, r1 = hole_cards[0][0], hole_cards[1][0]
        if r0 == r1:
            return ("pocket_pair", HAND_BUCKET_VALUES["pocket_pair"])
        elif abs(r0 - r1) == 1:
            return ("connectors", HAND_BUCKET_VALUES["connectors"])
        else:
            return ("high_card", HAND_BUCKET_VALUES["high_card"])

    h0_rank = hole_cards[0][0]
    h1_rank = hole_cards[1][0]
    board_rank = board_cards[0][0]
    hole_sorted = sorted([h0_rank, h1_rank], reverse=True)
    max_hole = hole_sorted[0]
    min_hole = hole_sorted[1]

    # Set: pocket pair matches board
    if h0_rank == h1_rank == board_rank:
        return ("set", HAND_BUCKET_VALUES["set"])

    # Pocket pair, no match with board
    if h0_rank == h1_rank:
        if h0_rank > board_rank:
            return ("overpair", HAND_BUCKET_VALUES["overpair"])
        else:
            return ("underpair", HAND_BUCKET_VALUES["underpair"])

    # One hole card pairs the board
    if h0_rank == board_rank or h1_rank == board_rank:
        kicker = h1_rank if h0_rank == board_rank else h0_rank
        mid_rank = (num_ranks - 1) / 2.0
        if kicker >= mid_rank:
            return ("top_pair_top_kicker", HAND_BUCKET_VALUES["top_pair_top_kicker"])
        else:
            return ("top_pair_weak_kicker", HAND_BUCKET_VALUES["top_pair_weak_kicker"])

    # No pair
    if min_hole > board_rank:
        return ("overcards", HAND_BUCKET_VALUES["overcards"])

    return ("air", HAND_BUCKET_VALUES["air"])
