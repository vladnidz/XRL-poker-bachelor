"""Global configuration for the XRL Poker Agent."""

# --- Game settings ---
# Mini Heads-Up Limit Hold'em: 3 ranks x 2 suits = 6 cards,
# 2 hole cards, 1 community card, 2 betting rounds.
MINI_HULH_RANKS = 3
MINI_HULH_SUITS = 2
MINI_HULH_GAME_STRING = (
    "universal_poker(betting=limit,numPlayers=2,numRounds=2,"
    "blind=1 2,raiseSize=2 4,maxRaises=3 3,"
    "numSuits=2,numRanks=3,numHoleCards=2,"
    "numBoardCards=0 1,firstPlayer=1 1)"
)

# --- CFR+ training ---
CFR_ITERATIONS = 1000
CFR_CHECKPOINT_EVERY = 200

# --- Decision tree ---
TREE_MAX_DEPTHS = [5, 7, 10, 12, 15]
TREE_DEFAULT_DEPTH = 10
TREE_RANDOM_STATE = 42

# --- Action labels ---
ACTION_NAMES = {0: "fold", 1: "call/check", 2: "raise/bet"}

# --- Paths ---
MODEL_DIR = "models"
DATA_DIR = "data"
