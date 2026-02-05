"""
Exact equity calculator for Leduc Hold'em.

Leduc: 6 cards (J, J, Q, Q, K, K), 1 hole card each, 1 community card.
Winning: pair with board > higher card > lower card. Ties possible.
"""

import numpy as np

# Card ID -> rank: 0,1=Jack(0), 2,3=Queen(1), 4,5=King(2)
CARD_TO_RANK = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
RANK_NAMES = {0: "Jack", 1: "Queen", 2: "King"}
ALL_CARDS = [0, 1, 2, 3, 4, 5]


def card_rank(card_id):
    return CARD_TO_RANK[card_id]


def rank_name(card_id):
    return RANK_NAMES[card_rank(card_id)]


def determine_winner(my_card, opp_card, board_card):
    """
    Determine winner given all 3 cards.
    Returns: 1 if my_card wins, -1 if opp wins, 0 if tie.
    """
    my_rank = card_rank(my_card)
    opp_rank = card_rank(opp_card)
    board_rank = card_rank(board_card)

    my_pair = (my_rank == board_rank)
    opp_pair = (opp_rank == board_rank)

    if my_pair and not opp_pair:
        return 1
    if opp_pair and not my_pair:
        return -1
    # Both pair or neither pair -> higher card wins
    if my_rank > opp_rank:
        return 1
    if opp_rank > my_rank:
        return -1
    return 0  # tie


def compute_equity_preflop(my_card):
    """
    Compute equity before the community card is dealt.
    Enumerate all possible (opponent_card, board_card) combinations.

    Returns: float in [0, 1]
    """
    remaining = [c for c in ALL_CARDS if c != my_card]
    wins = 0
    ties = 0
    total = 0

    for opp_card in remaining:
        leftover = [c for c in remaining if c != opp_card]
        for board_card in leftover:
            result = determine_winner(my_card, opp_card, board_card)
            if result == 1:
                wins += 1
            elif result == 0:
                ties += 1
            total += 1

    return (wins + 0.5 * ties) / total if total > 0 else 0.5


def compute_equity_postflop(my_card, board_card):
    """
    Compute equity after the community card is dealt.
    Enumerate all possible opponent cards.

    Returns: float in [0, 1]
    """
    remaining = [c for c in ALL_CARDS if c != my_card and c != board_card]
    wins = 0
    ties = 0
    total = 0

    for opp_card in remaining:
        result = determine_winner(my_card, opp_card, board_card)
        if result == 1:
            wins += 1
        elif result == 0:
            ties += 1
        total += 1

    return (wins + 0.5 * ties) / total if total > 0 else 0.5


def compute_future_equity_distribution(my_card):
    """
    For preflop: compute equity for each possible board card.
    Returns sorted list of equity values (one per possible board card).
    """
    remaining = [c for c in ALL_CARDS if c != my_card]
    equities = []
    for board_card in remaining:
        eq = compute_equity_postflop(my_card, board_card)
        equities.append(eq)
    return sorted(equities)


def equity_quantiles(distribution, n_quantiles=5):
    """Convert equity distribution into quantile values."""
    if not distribution:
        return [0.5] * n_quantiles
    arr = np.array(distribution)
    percentiles = np.linspace(100 / n_quantiles, 100, n_quantiles)
    return [float(np.percentile(arr, p)) for p in percentiles]
