"""Streamlit UI for the Explainable Poker Agent."""

import sys
import os
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.strategy_engine.decision_tree_policy import DecisionTreePolicy
from src.explanation_engine.shap_explainer import SHAPExplainer
from src.explanation_engine.decision_path import DecisionPathExtractor
from src.explanation_engine.counterfactual import CounterfactualGenerator
from src.explanation_engine.nl_generator import NLGenerator
from src.game_environment.holdem_equity import (
    compute_equity_preflop, compute_equity_postflop,
    compute_future_equity_distribution, equity_deciles,
    classify_hand, DISPLAY_RANKS, DISPLAY_SUITS,
)
from config import NUM_RANKS, NUM_SUITS


# --- Page config ---
st.set_page_config(
    page_title="XRL Poker Agent",
    page_icon="\u2660",
    layout="wide"
)

# --- Card CSS ---
CARD_CSS = """
<style>
.card {
    display: inline-block;
    width: 70px;
    height: 100px;
    border: 2px solid #333;
    border-radius: 8px;
    margin: 4px;
    background: white;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    line-height: 100px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    font-family: 'Segoe UI', Arial, sans-serif;
}
.card.red { color: #d32f2f; }
.card.black { color: #222; }
.card.hidden {
    background: linear-gradient(135deg, #1a237e 25%, #283593 25%, #283593 50%, #1a237e 50%, #1a237e 75%, #283593 75%);
    background-size: 10px 10px;
    color: transparent;
}
.poker-table {
    background: linear-gradient(145deg, #1b5e20, #2e7d32, #1b5e20);
    border-radius: 120px;
    border: 8px solid #5d4037;
    padding: 30px 20px;
    margin: 10px auto;
    max-width: 700px;
    text-align: center;
    box-shadow: inset 0 0 40px rgba(0,0,0,0.3);
}
.player-area { padding: 10px; margin: 5px auto; text-align: center; }
.player-label { color: #fdd835; font-weight: bold; font-size: 14px; margin-bottom: 4px; }
.opponent-label { color: #ef9a9a; font-weight: bold; font-size: 14px; margin-bottom: 4px; }
.board-area { margin: 15px auto; padding: 10px; }
.board-label { color: #e0e0e0; font-size: 12px; margin-bottom: 4px; }
.pot-display { color: #ffd54f; font-size: 22px; font-weight: bold; margin: 8px 0; }
.action-badge {
    display: inline-block; padding: 8px 24px; border-radius: 20px;
    font-size: 20px; font-weight: bold; margin: 10px;
}
.action-fold { background: #c62828; color: white; }
.action-call { background: #1565c0; color: white; }
.action-raise { background: #2e7d32; color: white; }
.equity-bar {
    background: #424242; border-radius: 4px; height: 20px;
    margin: 4px 0; overflow: hidden;
}
.equity-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
</style>
"""

N_DECILES = 10
RANKS = list(range(NUM_RANKS))
SUITS = list(range(NUM_SUITS))

RANK_LABELS = {r: DISPLAY_RANKS.get(r, str(r)) for r in RANKS}
SUIT_SYMBOLS = {0: "\u2663", 1: "\u2666", 2: "\u2665", 3: "\u2660"}
SUIT_COLORS = {0: "black", 1: "red", 2: "red", 3: "black"}


def card_html(rank, suit):
    label = RANK_LABELS[rank] + SUIT_SYMBOLS[suit]
    color = SUIT_COLORS[suit]
    return f'<div class="card {color}">{label}</div>'


def hidden_card_html():
    return '<div class="card hidden">?</div>'


def card_label(rank, suit):
    return RANK_LABELS[rank] + SUIT_SYMBOLS[suit]


@st.cache_resource
def load_agent():
    model_path = os.path.join("models", "decision_tree.joblib")
    if not os.path.exists(model_path):
        return None
    policy = DecisionTreePolicy()
    policy.load(model_path)
    return policy


def init_explainers(policy):
    return {
        "shap": SHAPExplainer(policy),
        "path": DecisionPathExtractor(policy),
        "counterfactual": CounterfactualGenerator(policy),
        "nl": NLGenerator(),
    }


def build_features_from_ui(hole_cards, board_card, pot, position, opp_r1, opp_r2):
    """Build feature vector from UI selections (matching holdem_features.py)."""
    features = []

    if board_card is not None:
        equity = compute_equity_postflop(hole_cards, board_card, NUM_RANKS, NUM_SUITS)
    else:
        equity = compute_equity_preflop(hole_cards, NUM_RANKS, NUM_SUITS)
    features.append(equity)

    if board_card is None:
        future_dist = compute_future_equity_distribution(hole_cards, NUM_RANKS, NUM_SUITS)
        deciles = equity_deciles(future_dist, N_DECILES)
    else:
        deciles = [0.0] * N_DECILES
    features.extend(deciles)

    round_num = 0 if board_card is None else 1
    raise_size = 2 if round_num == 0 else 4
    facing_raise = (opp_r1 == 2 and round_num == 0) or (opp_r2 == 2 and round_num == 1)
    bet_to_call = raise_size if facing_raise else 0
    pot_odds = bet_to_call / (pot + bet_to_call) if (pot + bet_to_call) > 0 else 0.0
    features.append(pot_odds)

    stack_to_pot = 100.0 if pot == 0 else min(2147483645 / pot, 100.0)
    features.append(stack_to_pot)

    features.append(float(position))
    features.append(float(opp_r1))
    features.append(float(opp_r2))

    board_list = [board_card] if board_card else []
    _, bucket_value = classify_hand(hole_cards, board_list, NUM_RANKS)
    features.append(float(bucket_value))

    has_pair = 1.0 if hole_cards[0][0] == hole_cards[1][0] else 0.0
    features.append(has_pair)

    max_rank = max(hole_cards[0][0], hole_cards[1][0])
    features.append(max_rank / max(NUM_RANKS - 1, 1))

    if board_card is not None:
        board_rank = board_card[0]
        hole_ranks = [hole_cards[0][0], hole_cards[1][0]]
        features.append(1.0 if board_rank in hole_ranks else 0.0)
    else:
        features.append(0.0)

    return np.array(features, dtype=np.float32), equity


def main():
    st.markdown(CARD_CSS, unsafe_allow_html=True)
    st.title("\u2660 Explainable Poker Agent")
    st.caption("Reduced-Deck Heads-Up Limit Hold'em \u2014 AI explains every decision")

    policy = load_agent()
    if policy is None:
        st.warning(
            "No trained model found. Run the training pipeline first:\n\n"
            "```\ndocker compose --profile train run train\n```"
        )
        st.stop()

    explainers = init_explainers(policy)
    action_names = policy.action_names or {0: "fold", 1: "call/check", 2: "raise/bet"}

    # ===================== SIDEBAR =====================
    st.sidebar.header("\u2660 Deal Cards")

    card_options = [(r, s) for r in RANKS for s in SUITS]
    card_labels_list = [card_label(r, s) for r, s in card_options]

    st.sidebar.subheader("Your Hole Cards")
    c1_idx = st.sidebar.selectbox("Card 1", range(len(card_options)),
                                   format_func=lambda i: card_labels_list[i], key="c1")
    c2_idx = st.sidebar.selectbox("Card 2", range(len(card_options)),
                                   format_func=lambda i: card_labels_list[i], key="c2",
                                   index=min(1, len(card_options)-1))

    hole_cards = (card_options[c1_idx], card_options[c2_idx])

    if hole_cards[0] == hole_cards[1]:
        st.sidebar.error("Cannot use the same card twice!")
        st.stop()

    st.sidebar.subheader("Community Card")
    board_option = st.sidebar.selectbox(
        "Board", ["None (Preflop)"] + card_labels_list, key="board"
    )
    if board_option == "None (Preflop)":
        board_card = None
    else:
        board_idx = card_labels_list.index(board_option)
        board_card = card_options[board_idx]
        if board_card in hole_cards:
            st.sidebar.error("Board card cannot duplicate a hole card!")
            st.stop()

    st.sidebar.subheader("Game State")
    position = st.sidebar.radio("Position", [0, 1],
                                 format_func=lambda x: "OOP (first to act)" if x == 0 else "IP (last to act)",
                                 key="pos")
    pot = st.sidebar.slider("Pot Size", 2, 20, 4, key="pot")

    st.sidebar.subheader("Opponent Actions")
    opp_r1 = st.sidebar.selectbox("Preflop", [0, 1, 2], key="opp_r1",
                                    format_func=lambda x: ["No action", "Passive", "Aggressive"][x])
    opp_r2 = st.sidebar.selectbox("Postflop", [0, 1, 2], key="opp_r2",
                                    format_func=lambda x: ["No action", "Passive", "Aggressive"][x])

    analyze = st.sidebar.button("Analyze", type="primary", use_container_width=True)

    # ===================== POKER TABLE =====================
    features, equity = build_features_from_ui(hole_cards, board_card, pot, position, opp_r1, opp_r2)

    opponent_cards = hidden_card_html() + hidden_card_html()
    hero_cards = card_html(*hole_cards[0]) + card_html(*hole_cards[1])

    if board_card is not None:
        board_html = card_html(*board_card)
        street = "Postflop"
    else:
        board_html = '<span style="color:#999; font-size:16px;">No board yet</span>'
        street = "Preflop"

    # Hand bucket display
    board_list = [board_card] if board_card else []
    bucket_name, _ = classify_hand(hole_cards, board_list, NUM_RANKS)
    pos_label = "IP (Button)" if position == 1 else "OOP (Big Blind)"

    table_html = f"""
    <div class="poker-table">
        <div class="player-area">
            <div class="opponent-label">OPPONENT</div>
            {opponent_cards}
        </div>
        <div class="board-area">
            <div class="board-label">BOARD \u2022 {street}</div>
            {board_html}
            <div class="pot-display">Pot: {pot} chips</div>
        </div>
        <div class="player-area">
            <div class="player-label">YOU ({pos_label}) \u2022 Equity: {equity:.0%} \u2022 {bucket_name.replace('_', ' ').title()}</div>
            {hero_cards}
        </div>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # ===================== ANALYSIS =====================
    if analyze:
        action_id = policy.predict(features)
        action_name = action_names.get(action_id, str(action_id))
        proba = policy.predict_proba(features)

        shap_result = explainers["shap"].explain(features)
        path_result = explainers["path"].extract(features)
        cf_result = explainers["counterfactual"].generate(features)

        action_css = {0: "action-fold", 1: "action-call", 2: "action-raise"}
        badge_class = action_css.get(action_id, "action-call")
        st.markdown(
            f'<div style="text-align:center">'
            f'<div class="action-badge {badge_class}">{action_name.upper()}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Action Probabilities")
            for aid in sorted(proba.keys()):
                aname = action_names.get(aid, str(aid))
                prob = proba[aid]
                color = "#c62828" if aid == 0 else "#1565c0" if aid == 1 else "#2e7d32"
                st.markdown(
                    f'<div style="margin:4px 0;">'
                    f'<span style="display:inline-block;width:90px;">{aname}</span>'
                    f'<span style="font-weight:bold;">{prob:.0%}</span>'
                    f'<div class="equity-bar"><div class="equity-fill" '
                    f'style="width:{prob*100:.0f}%;background:{color};"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.subheader("Feature Attribution (SHAP)")
            for feat in shap_result["top_features"]:
                fname = feat["feature"].replace("_", " ").title()
                sv = feat["shap_value"]
                color = "#2e7d32" if sv > 0 else "#c62828"
                arrow = "\u2191" if sv > 0 else "\u2193"
                st.markdown(
                    f"**{fname}** = {feat['value']:.2f} "
                    f"<span style='color:{color};font-weight:bold'>"
                    f"{arrow} {sv:+.3f}</span>",
                    unsafe_allow_html=True
                )

        with col2:
            st.subheader("Why this action?")
            explanations = explainers["nl"].generate_all(
                shap_result, path_result, cf_result, action_names=action_names
            )
            st.markdown(explanations["full"])

        with st.expander("Full Decision Tree Path"):
            st.code(path_result["path_string"])
            st.caption(f"Path length: {path_result['path_length']} decision nodes")

        with st.expander("Raw Feature Vector"):
            fnames = policy.feature_names or [f"f{i}" for i in range(len(features))]
            for name, val in zip(fnames, features):
                st.text(f"  {name:25s} = {val:.4f}")
    else:
        st.info("Select cards and game state, then click **Analyze**.")

    st.markdown("---")
    st.caption(
        "Bachelor Thesis: Explainable AI Based Poker Agent for Learning Purposes | "
        "Riga Technical University"
    )


if __name__ == "__main__":
    main()
