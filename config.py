"""Global configuration for the XRL Poker Agent."""

# --- Game settings ---
# Heads-Up Limit Hold'em with reduced deck:
# 6 ranks (9,T,J,Q,K,A) x 4 suits = 24 cards
# 2 hole cards, 1 community card (flop), 2 betting rounds
NUM_RANKS = 6
NUM_SUITS = 4
GAME_STRING = (
    "universal_poker(betting=limit,numPlayers=2,numRounds=2,"
    "blind=1 2,raiseSize=2 4,maxRaises=3 3,"
    "numSuits=4,numRanks=6,numHoleCards=2,"
    "numBoardCards=0 1,firstPlayer=1 1)"
)

# --- MCCFR training ---
MCCFR_ITERATIONS = 1_000_000
MCCFR_CHECKPOINT_EVERY = 250_000

# --- Decision tree ---
TREE_MAX_DEPTHS = [5, 7, 10, 12, 15]
TREE_DEFAULT_DEPTH = 10
TREE_RANDOM_STATE = 42

# --- Action labels ---
ACTION_NAMES = {0: "fold", 1: "call/check", 2: "raise/bet"}

# --- Paths ---
MODEL_DIR = "models"
DATA_DIR = "data"
