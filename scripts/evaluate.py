"""
Script 4: Evaluate the trained agent's playing strength and explanation quality.

Usage:
    python scripts/evaluate.py [--num-games N]
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


def play_game(game, tree_policy, feature_builder, opponent="random", rng=None):
    """
    Play one game: tree agent (player 0) vs opponent (player 1).
    Returns: payoff for the tree agent
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

        if current_player == 0:
            # Tree agent -- use equity-based features
            features = feature_builder(state, current_player)
            action = tree_policy.predict(features)
            if action not in legal_actions:
                # Fallback: pick best legal action
                action = legal_actions[-1] if len(legal_actions) > 1 else legal_actions[0]
        else:
            # Opponent
            if opponent == "random":
                action = rng.choice(legal_actions)
            else:
                action = legal_actions[0]

        state.apply_action(action)

    return state.returns()[0]


def evaluate_playing_strength(game, tree_policy, feature_builder, num_games=10000):
    """Evaluate win rate against random opponent."""
    rng = np.random.default_rng(42)
    payoffs = []

    for _ in range(num_games):
        payoff = play_game(game, tree_policy, feature_builder,
                           opponent="random", rng=rng)
        payoffs.append(payoff)

    payoffs = np.array(payoffs)
    return {
        "num_games": num_games,
        "mean_payoff": payoffs.mean(),
        "std_error": payoffs.std() / np.sqrt(len(payoffs)),
        "win_rate": (payoffs > 0).mean(),
        "loss_rate": (payoffs < 0).mean(),
        "tie_rate": (payoffs == 0).mean(),
    }


def evaluate_explanations(tree_policy, feature_builder, game, num_samples=100):
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
    args = parser.parse_args()

    # Load model
    policy = DecisionTreePolicy()
    policy.load(args.model)

    game = pyspiel.load_game(GAME_STRING)
    feature_builder = build_features
    game_name = "Reduced-Deck Heads-Up Limit Hold'em"

    print(f"=== Evaluation: {game_name} ===\n")

    # Playing strength
    print("--- Playing Strength (vs Random) ---")
    strength = evaluate_playing_strength(game, policy, feature_builder, args.num_games)
    print(f"  Games played:  {strength['num_games']}")
    print(f"  Mean payoff:   {strength['mean_payoff']:.4f} "
          f"(+/- {strength['std_error']:.4f})")
    print(f"  Win rate:      {strength['win_rate']:.1%}")
    print(f"  Loss rate:     {strength['loss_rate']:.1%}")
    print(f"  Tie rate:      {strength['tie_rate']:.1%}")

    # Explanation quality
    print("\n--- Explanation Quality ---")
    expl = evaluate_explanations(policy, feature_builder, game)
    print(f"  Samples evaluated:         {expl['total_samples']}")
    print(f"  Counterfactual found rate:  {expl['cf_found_rate']:.1%}")
    print(f"  Avg decision path length:  {expl['avg_path_length']:.1f}")
    print(f"  Avg top SHAP magnitude:    {expl['avg_top_shap_magnitude']:.4f}")
    print(f"  Avg explanation length:    {expl['avg_explanation_length']:.0f} chars")

    # Sample explanation
    print("\n--- Sample Explanation ---")
    state = game.new_initial_state()
    rng = np.random.default_rng(123)
    while state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        state.apply_action(rng.choice(actions, p=probs))

    features = feature_builder(state, state.current_player())
    explainer = SHAPExplainer(policy)
    path_ext = DecisionPathExtractor(policy)
    cf_gen = CounterfactualGenerator(policy)
    nl_gen = NLGenerator()

    shap_r = explainer.explain(features)
    path_r = path_ext.extract(features)
    cf_r = cf_gen.generate(features)
    print(nl_gen.generate(shap_r, path_r, cf_r, template="full"))


if __name__ == "__main__":
    main()
