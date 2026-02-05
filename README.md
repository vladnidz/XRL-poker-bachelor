# XRL Poker Agent

Explainable AI Based Poker Agent for Learning Purposes.

Bachelor thesis project, Riga Technical University.

## Overview

An AI poker agent that plays **Mini Heads-Up Limit Hold'em** and explains its decisions using explainable AI (XAI) methods. The system trains a game-theoretically optimal strategy via CFR+, distills it into an interpretable CART decision tree, and generates multi-layered explanations (SHAP attribution, decision paths, counterfactuals, natural language) to help users learn correct poker strategy.

## Architecture

The system follows a four-layer architecture:

```
Layer 1: Game Environment     (OpenSpiel + exact equity computation)
Layer 2: Strategy Engine       (CFR+ solver + CART decision tree distillation)
Layer 3: Explanation Engine    (TreeSHAP + decision paths + counterfactuals + NL)
Layer 4: User Interface        (Streamlit interactive dashboard)
```

### Layer 1 -- Game Environment (`src/game_environment/`)

- **Game**: Mini HULH via OpenSpiel's `universal_poker` (3 ranks x 2 suits, 2 hole cards, 1 community card, 2 betting rounds, limit betting)
- **Equity**: Exact enumeration over all opponent hands and board cards (tractable for small decks)
- **Features** (19-dimensional vector following Bertsimas & Paskov 2022):
  - Current hand equity
  - 10 future equity deciles (preflop only)
  - Pot odds, stack-to-pot ratio
  - Position (OOP/IP)
  - Opponent aggression per round (2 features)
  - Pocket pair indicator, max hole card rank, board-pairs-hole indicator

Key files:
- `holdem_equity.py` -- exact equity calculator, hand evaluator, ACPC card parser
- `holdem_features.py` -- equity-based feature builder, OpenSpiel state parser
- `poker_game.py` -- OpenSpiel game wrapper
- `equity_calculator.py` -- Monte Carlo equity (eval7-based, for full Hold'em)
- `state_representation.py` -- generic feature vector builder (Bertsimas & Paskov design)

### Layer 2 -- Strategy Engine (`src/strategy_engine/`)

- **CFR+ Trainer** (`cfr_trainer.py`): Wraps OpenSpiel's `CFRPlusSolver`. Trains to Nash equilibrium with exploitability checkpoints. Serializes solver state with pickle.
- **Data Generator** (`data_generator.py`): Produces (features, action) pairs from the converged CFR+ policy. Supports both full game tree traversal (exact coverage) and random sampling.
- **Decision Tree Policy** (`decision_tree_policy.py`): scikit-learn `DecisionTreeClassifier` with balanced class weights. Supports depth search (cross-validation over multiple depths), save/load via joblib.

### Layer 3 -- Explanation Engine (`src/explanation_engine/`)

- **TreeSHAP** (`shap_explainer.py`): Exact Shapley values via `shap.TreeExplainer`. Returns top-k feature attributions with direction (supports/opposes).
- **Decision Path** (`decision_path.py`): Extracts root-to-leaf IF-THEN chain from the CART tree. Human-readable path string.
- **Counterfactual** (`counterfactual.py`): Walks up from leaf node, finds nearest sibling branch leading to a different action. Returns minimal "what-if" statement.
- **NL Generator** (`nl_generator.py`): Jinja2 templates combining SHAP, decision path, and counterfactual into natural language explanations (brief, path, full formats).

### Layer 4 -- User Interface (`src/ui/`)

- **Streamlit App** (`app.py`): Interactive dashboard where users set game state via sidebar widgets, get agent decisions with full explanations (SHAP table, decision path, counterfactual, NL summary).

## Pipeline

Four-step training and evaluation pipeline (`scripts/run_pipeline.py`):

1. **Train CFR+** (`scripts/train_cfr.py`) -- Solve the game to Nash equilibrium
2. **Generate Data** (`scripts/generate_data.py`) -- Extract (features, action) pairs via tree traversal
3. **Train Decision Tree** (`scripts/train_tree.py`) -- Distill CFR+ policy into interpretable CART tree
4. **Evaluate** (`scripts/evaluate.py`) -- Measure playing strength vs random + explanation quality metrics

## Quick Start

### Docker (recommended)

```bash
# Run full pipeline (train + evaluate)
docker compose --profile train run train

# Launch UI (after training)
docker compose up app
# Open http://localhost:8501
```

### Local

```bash
pip install -r requirements.txt

# Full pipeline
python scripts/run_pipeline.py --game holdem --iterations 2000 --depth 12

# Or step by step
python scripts/train_cfr.py --game holdem --iterations 2000
python scripts/generate_data.py --traverse
python scripts/train_tree.py --depth 12 --depth-search
python scripts/evaluate.py --num-games 10000

# Launch UI
streamlit run src/ui/app.py
```

## Configuration

All hyperparameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MINI_HULH_GAME_STRING` | 3r x 2s | OpenSpiel universal_poker config |
| `CFR_ITERATIONS` | 1000 | CFR+ training iterations |
| `TREE_DEFAULT_DEPTH` | 10 | CART max depth |
| `TREE_MAX_DEPTHS` | [5,7,10,12,15] | Depths for cross-validation search |

## Project Structure

```
XRL-poker-bachelor/
  config.py                        # Global configuration
  Dockerfile                       # Python 3.11 + build tools
  docker-compose.yml               # Services: app, train, evaluate
  requirements.txt                 # Dependencies
  scripts/
    run_pipeline.py                # Full pipeline orchestrator
    train_cfr.py                   # Step 1: CFR+ training
    generate_data.py               # Step 2: Data generation
    train_tree.py                  # Step 3: Decision tree training
    evaluate.py                    # Step 4: Evaluation
  src/
    game_environment/
      holdem_equity.py             # Exact equity for mini HULH
      holdem_features.py           # Equity-based feature builder
      poker_game.py                # OpenSpiel wrapper
      equity_calculator.py         # Monte Carlo equity (eval7)
      state_representation.py      # Generic feature vector design
      leduc_equity.py              # Leduc equity (legacy)
      leduc_features.py            # Leduc features (legacy)
    strategy_engine/
      cfr_trainer.py               # CFR+ solver wrapper
      data_generator.py            # Policy distillation data
      decision_tree_policy.py      # CART classifier
    explanation_engine/
      shap_explainer.py            # TreeSHAP attribution
      decision_path.py             # IF-THEN path extraction
      counterfactual.py            # Sibling-branch counterfactuals
      nl_generator.py              # Jinja2 NL templates
    ui/
      app.py                       # Streamlit dashboard
  models/                          # Trained models (gitignored)
  data/                            # Training data (gitignored)
```

## Key Dependencies

- `open_spiel` -- Game simulation, CFR+ solver
- `eval7` -- Fast poker hand evaluation (for full Hold'em equity)
- `scikit-learn` -- CART decision tree
- `shap` -- TreeSHAP feature attribution
- `jinja2` -- NL explanation templates
- `streamlit` -- Interactive UI
- `numpy`, `joblib`, `tqdm`, `matplotlib`

## Technical Notes

- **Why Mini HULH?** Full Limit HU Texas Hold'em has ~3.16 x 10^17 states and requires ~900 CPU-years to solve (Bowling et al. 2015). Mini HULH (6 cards, 2 hole, 1 board) preserves the Hold'em structure (hole cards + community card, preflop/postflop rounds) while being tractable for CFR+ (~30 min for 1000 iterations).
- **CFR+ convergence**: Exploitability drops to ~0.0002 after 1000 iterations (near-Nash equilibrium).
- **Policy distillation**: The CART tree achieves ~76% cross-validated accuracy on the CFR+ mixed strategy. Accuracy is bounded by the mixed-strategy nature of Nash equilibrium -- the tree must pick a single action where CFR+ randomizes.
- **Explanation coverage**: 100% counterfactual found rate (every leaf has a sibling with a different action).
