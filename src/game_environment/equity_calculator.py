"""Hand equity estimation using eval7 Monte Carlo simulation."""

import eval7
import numpy as np
from itertools import combinations


class EquityCalculator:
    """Calculates hand equity via Monte Carlo sampling (eval7)."""

    RANK_MAP = {
        0: '2', 1: '3', 2: '4', 3: '5', 4: '6', 5: '7', 6: '8',
        7: '9', 8: 'T', 9: 'J', 10: 'Q', 11: 'K', 12: 'A'
    }
    SUIT_MAP = {0: 's', 1: 'h', 2: 'd', 3: 'c'}

    @staticmethod
    def card_to_eval7(rank, suit):
        """Convert rank (0-12) and suit (0-3) to eval7 Card."""
        rank_char = EquityCalculator.RANK_MAP[rank]
        suit_char = EquityCalculator.SUIT_MAP[suit]
        return eval7.Card(rank_char + suit_char)

    @staticmethod
    def parse_openspiel_cards(info_state_string):
        """
        Parse cards from OpenSpiel universal_poker info state string.
        Returns (hole_cards, board_cards) as lists of eval7 Card objects.
        """
        # Universal poker info states look like:
        # "[Round X][Player X][Card1 Card2][Board cards][Actions]"
        # Cards are encoded as integers: card_id = rank * 4 + suit
        hole_cards = []
        board_cards = []
        # This is a simplified parser — will be refined based on actual format
        return hole_cards, board_cards

    @staticmethod
    def cards_from_ids(card_ids):
        """Convert OpenSpiel card IDs (rank*4 + suit) to eval7 Cards."""
        cards = []
        for cid in card_ids:
            rank = cid // 4
            suit = cid % 4
            cards.append(EquityCalculator.card_to_eval7(rank, suit))
        return cards

    @staticmethod
    def compute_equity(hole_cards, board_cards, num_samples=10000):
        """
        Compute hand equity via Monte Carlo simulation.

        Args:
            hole_cards: list of 2 eval7.Card objects (player's hand)
            board_cards: list of 0-5 eval7.Card objects (community cards)
            num_samples: number of Monte Carlo samples

        Returns:
            float: equity value in [0, 1]
        """
        if not hole_cards:
            return 0.5  # No cards known yet

        deck = eval7.Deck()
        known_cards = hole_cards + board_cards

        # Remove known cards from deck
        dead_cards = set()
        for card in known_cards:
            dead_cards.add(card)

        remaining = [c for c in deck.cards if c not in dead_cards]
        cards_needed_for_board = 5 - len(board_cards)

        wins = 0
        ties = 0
        total = 0

        rng = np.random.default_rng()

        for _ in range(num_samples):
            # Shuffle remaining cards
            sampled = rng.choice(len(remaining), size=cards_needed_for_board + 2, replace=False)
            sampled_cards = [remaining[i] for i in sampled]

            opp_hand = sampled_cards[:2]
            future_board = sampled_cards[2:]

            full_board = board_cards + future_board

            my_hand = eval7.evaluate(hole_cards + full_board)
            opp_hand_val = eval7.evaluate(opp_hand + full_board)

            if my_hand > opp_hand_val:
                wins += 1
            elif my_hand == opp_hand_val:
                ties += 1
            total += 1

        if total == 0:
            return 0.5

        return (wins + 0.5 * ties) / total

    @staticmethod
    def compute_future_equity_distribution(hole_cards, board_cards,
                                            future_cards_count=1,
                                            num_board_samples=200,
                                            num_equity_samples=500):
        """
        Compute distribution of future equity values for upcoming streets.

        For each possible future board card reveal, compute the equity,
        building a distribution that captures hand development potential.

        Args:
            hole_cards: list of eval7.Card (player's hand)
            board_cards: list of eval7.Card (current board)
            future_cards_count: how many new cards will be revealed (1 or 3)
            num_board_samples: how many future board completions to sample
            num_equity_samples: MC samples per equity computation

        Returns:
            list of floats: equity values for each sampled future board
        """
        deck = eval7.Deck()
        known = set(hole_cards + board_cards)
        remaining = [c for c in deck.cards if c not in known]

        rng = np.random.default_rng()
        equities = []

        for _ in range(num_board_samples):
            idxs = rng.choice(len(remaining), size=future_cards_count, replace=False)
            new_board = board_cards + [remaining[i] for i in idxs]

            eq = EquityCalculator.compute_equity(
                hole_cards, new_board, num_samples=num_equity_samples
            )
            equities.append(eq)

        return sorted(equities)

    @staticmethod
    def equity_deciles(equity_distribution):
        """
        Convert an equity distribution into 10 decile values.

        Args:
            equity_distribution: sorted list of equity values

        Returns:
            list of 10 floats: decile values (10th, 20th, ..., 100th percentile)
        """
        if not equity_distribution:
            return [0.5] * 10

        arr = np.array(equity_distribution)
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        return [float(np.percentile(arr, p)) for p in percentiles]
