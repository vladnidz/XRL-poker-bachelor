"""OpenSpiel wrapper for poker games."""

import pyspiel
import numpy as np


class PokerGame:
    """Wraps an OpenSpiel poker game for game management and state queries."""

    def __init__(self, game_name="universal_poker", game_string=None):
        if game_string:
            self.game = pyspiel.load_game(game_string)
        else:
            self.game = pyspiel.load_game(game_name)
        self.state = None

    def new_game(self):
        """Start a new hand."""
        self.state = self.game.new_initial_state()
        return self.state

    def apply_action(self, action):
        """Apply an action to the current state."""
        self.state.apply_action(action)
        return self.state

    def get_legal_actions(self, player=None):
        """Return legal actions for the current player."""
        if self.state.is_terminal():
            return []
        if player is None:
            player = self.state.current_player()
        return self.state.legal_actions(player)

    def is_terminal(self):
        """Check if the game is over."""
        return self.state.is_terminal()

    def is_chance_node(self):
        """Check if the current state is a chance (deal) node."""
        return self.state.is_chance_node()

    def current_player(self):
        """Return the current player index."""
        return self.state.current_player()

    def returns(self):
        """Return the payoffs for each player."""
        return self.state.returns()

    def get_info_state_string(self, player):
        """Return the information state string for a player."""
        return self.state.information_state_string(player)

    def get_state_string(self):
        """Return the full state string."""
        return str(self.state)

    def clone(self):
        """Clone the current game state."""
        new_game = PokerGame.__new__(PokerGame)
        new_game.game = self.game
        new_game.state = self.state.clone()
        return new_game

    @staticmethod
    def load_mini_holdem():
        """Load Mini HULH: 3 ranks x 2 suits, 2 hole cards, 1 board card."""
        from config import MINI_HULH_GAME_STRING
        return PokerGame(game_string=MINI_HULH_GAME_STRING)

    @staticmethod
    def load_leduc_poker():
        """Load Leduc Hold'em (6-card, 1 hole, 1 board)."""
        return PokerGame(game_name="leduc_poker")

    @staticmethod
    def load_kuhn_poker():
        """Load Kuhn poker (small game for fast testing)."""
        return PokerGame(game_name="kuhn_poker")
