"""
Script 4: Evaluate the trained agent's playing strength and explanation quality.

Supports three opponent types:
  - random:    uniformly random actions
  - heuristic: rule-based equity-aware opponent
  - mccfr:     the MCCFR teacher policy (near-equilibrium)

Usage:
    python scripts/evaluate.py [--num-games N] [--opponent random|heuristic|mccfr]
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyspiel
from src.strategy_engine.decision_tree_policy import DecisionTreePolicy
from src.explanation_engine.shap_explainer import SHAPExplainer
from src.explanation_engine.decision_path import DecisionPathExtractor
from src.explanation_engine.counterfactual import CounterfactualGenerator
from src.explanation_engine.nl_generator import NLGenerator
from src.game_environment.holdem_features import build_features
from config import MODEL_DIR, GAME_STRING


def load_mccfr_policy(path):
    """Load the MCCFR average policy from a checkpoint."""
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    solver = data['solver']
    return solver.average_policy()


def heuristic_action(state, player, rng):
    """
    Rule-based opponent that uses equity-aware heuristics.
    Strategy:
      - Computes features for the current state
      - Uses equity thresholds to decide: fold/call/raise
      - Adds small randomness to avoid being fully deterministic
    """
    legal_actions = state.legal_actions(player)
    features = build_features(state, player)

    equity = features[0]  # first feature is always equity
    pot_odds = features[11]  # pot_odds feature index
    is_facing_bet = features[22]  # is_facing_bet feature index

    # If only one legal action, take it
    if len(legal_actions) == 1:
        return legal_actions[0]

    # Action mapping: 0=fold, 1=call/check, 2=raise/bet
    can_fold = 0 in legal_actions
    can_call = 1 in legal_actions
    can_raise = 2 in legal_actions

    noise = rng.uniform(-0.05, 0.05)
    adj_equity = equity + noise

    if is_facing_bet > 0.5:
        # Facing a bet: need good equity to continue
        if adj_equity >= 0.65 and can_raise:
            return 2  # raise with strong hands
        elif adj_equity >= 0.40 and can_call:
            return 1  # call with medium hands
        elif adj_equity >= 0.40 and pot_odds > 0.3 and can_call:
            return 1  # call if pot odds justify it
        elif can_fold:
            return 0  # fold weak hands
        elif can_call:
            return 1
        else:
            return legal_actions[0]
    else:
        # Not facing a bet: can check or bet
        if adj_equity >= 0.60 and can_raise:
            return 2  # bet strong hands
        elif can_call:
            return 1  # check medium/weak hands
        else:
            return legal_actions[0]


def play_game(game, tree_policy, feature_builder, opponent="random",
              mccfr_policy=None, rng=None, tree_player=0):
    """
    Play one game: tree agent vs opponent.
    tree_player: which player seat the tree agent occupies (0 or 1).
    Returns: payoff for the tree agent.
    """
    if rng is None:
        rng = np.random.default_rng()

    state = game.new_initial_state()

    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            actions, probs = zip(*outcomes)
            action = rng.choice(actions, p=probs)
            state.apply_action(action)
            continue

        current_player = state.current_player()
        legal_actions = state.legal_actions(current_player)

        if current_player == tree_player:
            # Tree agent
            features = feature_builder(state, current_player)
            action = tree_policy.predict(features)
            if action not in legal_actions:
                action = legal_actions[-1] if len(legal_actions) > 1 else legal_actions[0]
        else:
            # Opponent
            if opponent == "random":
                action = rng.choice(legal_actions)
            elif opponent == "heuristic":
                action = heuristic_action(state, current_player, rng)
                if action not in legal_actions:
                    action = rng.choice(legal_actions)
            elif opponent == "mccfr" and mccfr_policy is not None:
                action_probs = mccfr_policy.action_probabilities(state)
                probs_list = [action_probs.get(a, 0.0) for a in legal_actions]
                total = sum(probs_list)
                if total > 0:
                    probs_list = [p / total for p in probs_list]
                else:
                    probs_list = [1.0 / len(legal_actions)] * len(legal_actions)
                action = rng.choice(legal_actions, p=probs_list)
            else:
                action = rng.choice(legal_actions)

        state.apply_action(action)

    return state.returns()[tree_player]


def evaluate_playing_strength(game, tree_policy, feature_builder, num_games=10000,
                              opponent="random", mccfr_policy=None):
    """Evaluate win rate against specified opponent, alternating positions."""
    rng = np.random.default_rng(42)
    payoffs = []

    for i in range(num_games):
        # Alternate positions for fairness
        tree_player = i % 2
        payoff = play_game(game, tree_policy, feature_builder,
                           opponent=opponent, mccfr_policy=mccfr_policy,
                           rng=rng, tree_player=tree_player)
        payoffs.append(payoff)

    payoffs = np.array(payoffs)
    return {
        "opponent": opponent,
        "num_games": num_games,
        "mean_payoff": payoffs.mean(),
        "std_error": payoffs.std() / np.sqrt(len(payoffs)),
        "win_rate": (payoffs > 0).mean(),
        "loss_rate": (payoffs < 0).mean(),
        "tie_rate": (payoffs == 0).mean(),
        "median_payoff": np.median(payoffs),
    }


def evaluate_explanations(tree_policy, feature_builder, game, num_samples=200):
    """Evaluate explanation quality by running sample games."""
    explainer = SHAPExplainer(tree_policy)
    path_extractor = DecisionPathExtractor(tree_policy)
    cf_generator = CounterfactualGenerator(tree_policy)
    nl_gen = NLGenerator()

    rng = np.random.default_rng(42)
    metrics = {
        "cf_found_rate": 0,
        "avg_path_length": 0,
        "avg_top_shap_magnitude": 0,
        "explanation_lengths": [],
    }

    sample_count = 0
    for _ in range(num_samples):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions, probs = zip(*outcomes)
                state.apply_action(rng.choice(actions, p=probs))
                continue

            current_player = state.current_player()
            legal_actions = state.legal_actions(current_player)

            if current_player == 0:
                features = feature_builder(state, current_player)

                shap_result = explainer.explain(features)
                path_result = path_extractor.extract(features)
                cf_result = cf_generator.generate(features)
                explanation = nl_gen.generate(
                    shap_result, path_result, cf_result, template="full"
                )

                if cf_result["found"]:
                    metrics["cf_found_rate"] += 1
                metrics["avg_path_length"] += path_result["path_length"]
                if shap_result["top_features"]:
                    metrics["avg_top_shap_magnitude"] += abs(
                        shap_result["top_features"][0]["shap_value"]
                    )
                metrics["explanation_lengths"].append(len(explanation))
                sample_count += 1

            # Random action to continue
            state.apply_action(rng.choice(legal_actions))

    if sample_count > 0:
        metrics["cf_found_rate"] /= sample_count
        metrics["avg_path_length"] /= sample_count
        metrics["avg_top_shap_magnitude"] /= sample_count
        metrics["avg_explanation_length"] = np.mean(metrics["explanation_lengths"])
    del metrics["explanation_lengths"]
    metrics["total_samples"] = sample_count

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate poker agent")
    parser.add_argument("--num-games", type=int, default=10000)
    parser.add_argument("--model", type=str,
                        default=os.path.join(MODEL_DIR, "decision_tree.joblib"))
    parser.add_argument("--cfr-model", type=str,
                        default=os.path.join(MODEL_DIR, "cfr_final.pkl"))
    parser.add_argument("--opponent", type=str, default="all",
                        choices=["random", "heuristic", "mccfr", "all"])
    args = parser.parse_args()

    # Load tree model
    policy = DecisionTreePolicy()
    policy.load(args.model)

    game = pyspiel.load_game(GAME_STRING)
    feature_builder = build_features
    game_name = "Reduced-Deck Heads-Up Limit Hold'em"

    # Load MCCFR policy
    mccfr_policy = None
    if args.opponent in ("mccfr", "all") and os.path.exists(args.cfr_model):
        print(f"Loading MCCFR policy from {args.cfr_model}...")
        mccfr_policy = load_mccfr_policy(args.cfr_model)

    print(f"\n{'='*60}")
    print(f"  Evaluation: {game_name}")
    print(f"{'='*60}\n")

    opponents = []
    if args.opponent == "all":
        opponents = ["random", "heuristic"]
        if mccfr_policy is not None:
            opponents.append("mccfr")
    else:
        opponents = [args.opponent]

    results = []
    for opp in opponents:
        print(f"--- Playing Strength vs {opp.upper()} ({args.num_games} games) ---")
        mp = mccfr_policy if opp == "mccfr" else None
        strength = evaluate_playing_strength(
            game, policy, feature_builder, args.num_games,
            opponent=opp, mccfr_policy=mp
        )
        results.append(strength)
        print(f"  Mean payoff:   {strength['mean_payoff']:+.4f} "
              f"(+/- {strength['std_error']:.4f})")
        print(f"  Win rate:      {strength['win_rate']:.1%}")
        print(f"  Loss rate:     {strength['loss_rate']:.1%}")
        print(f"  Tie rate:      {strength['tie_rate']:.1%}")
        print()

    # Summary table
    print(f"\n{'='*60}")
    print("  Summary: Playing Strength Across Opponents")
    print(f"{'='*60}")
    print(f"{'Opponent':<12} {'Mean Payoff':>14} {'Win %':>8} {'Loss %':>8} {'Tie %':>8}")
    print("-" * 52)
    for r in results:
        print(f"{r['opponent']:<12} {r['mean_payoff']:>+10.4f}     "
              f"{r['win_rate']:>7.1%} {r['loss_rate']:>7.1%} {r['tie_rate']:>7.1%}")

    # Explanation quality
    print(f"\n--- Explanation Quality ---")
    expl = evaluate_explanations(policy, feature_builder, game)
    print(f"  Samples evaluated:          {expl['total_samples']}")
    print(f"  Counterfactual found rate:  {expl['cf_found_rate']:.1%}")
    print(f"  Avg decision path length:   {expl['avg_path_length']:.1f}")
    print(f"  Avg top SHAP magnitude:     {expl['avg_top_shap_magnitude']:.4f}")
    print(f"  Avg explanation length:     {expl['avg_explanation_length']:.0f} chars")


if __name__ == "__main__":
    main()
