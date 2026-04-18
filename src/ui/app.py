"""Streamlit UI for the Explainable Poker Agent."""

import sys
import os
import random
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
    classify_hand, compare_hands, hand_strength, DISPLAY_RANKS, DISPLAY_SUITS,
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
.game-log {
    background: #1a1a2e; color: #e0e0e0; padding: 12px; border-radius: 8px;
    font-family: 'Consolas', monospace; font-size: 13px; max-height: 300px;
    overflow-y: auto; margin: 8px 0;
}
.game-log .action-text { color: #4fc3f7; font-weight: bold; }
.game-log .result-win { color: #66bb6a; font-weight: bold; }
.game-log .result-lose { color: #ef5350; font-weight: bold; }
.score-display {
    font-size: 28px; font-weight: bold; text-align: center;
    padding: 10px; border-radius: 8px; margin: 8px 0;
}
</style>
"""

N_DECILES = 10
RANKS = list(range(NUM_RANKS))
SUITS = list(range(NUM_SUITS))

RANK_LABELS = {r: DISPLAY_RANKS.get(r, str(r)) for r in RANKS}
SUIT_SYMBOLS = {0: "\u2663", 1: "\u2666", 2: "\u2665", 3: "\u2660"}
SUIT_COLORS = {0: "black", 1: "red", 2: "red", 3: "black"}
ACTION_NAMES = {0: "fold", 1: "call/check", 2: "raise/bet"}


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


def build_features(hole_cards, board_card, pot, position, opp_r1, opp_r2):
    """Build feature vector from game state."""
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

    features.append(1.0 if hole_cards[0][1] == hole_cards[1][1] else 0.0)

    gap = abs(hole_cards[0][0] - hole_cards[1][0])
    features.append(gap / max(NUM_RANKS - 1, 1))

    features.append(1.0 if bet_to_call > 0 else 0.0)

    return np.array(features, dtype=np.float32), equity


# =====================================================================
# GAME MODE
# =====================================================================

def deal_new_game():
    """Deal a fresh hand."""
    deck = [(r, s) for r in RANKS for s in SUITS]
    random.shuffle(deck)
    hero_cards = (deck[0], deck[1])
    opp_cards = (deck[2], deck[3])
    board_card = deck[4]
    # Player 0 = SB (out of position), Player 1 = BB
    # Randomize who is dealer
    hero_position = random.choice([0, 1])
    return {
        "hero_cards": hero_cards,
        "opp_cards": opp_cards,
        "board_card": board_card,
        "hero_position": hero_position,  # 0=SB/OOP, 1=BB/IP
        "pot": 3,  # SB(1) + BB(2)
        "street": "preflop",
        "hero_invested": 1 if hero_position == 0 else 2,
        "opp_invested": 2 if hero_position == 0 else 1,
        "actions_log": [],
        "opp_aggression_r1": 0,
        "opp_aggression_r2": 0,
        "hero_aggression_r1": 0,
        "hero_aggression_r2": 0,
        "preflop_done": False,
        "postflop_done": False,
        "game_over": False,
        "result": None,
        "result_chips": 0,
        "awaiting_hero": True,
        "explanation": None,
    }


def ai_decide(policy, explainers, game):
    """AI makes a decision and returns action + explanation."""
    board = game["board_card"] if game["street"] == "postflop" else None
    opp_r1 = game["hero_aggression_r1"]  # from AI's perspective, hero is its opponent
    opp_r2 = game["hero_aggression_r2"]
    ai_position = 1 - game["hero_position"]

    features, equity = build_features(
        game["opp_cards"], board, game["pot"], ai_position, opp_r1, opp_r2
    )

    action_id = policy.predict(features)
    proba = policy.predict_proba(features)

    shap_result = explainers["shap"].explain(features)
    path_result = explainers["path"].extract(features)
    cf_result = explainers["counterfactual"].generate(features)
    nl_explanations = explainers["nl"].generate_all(
        shap_result, path_result, cf_result, action_names=ACTION_NAMES
    )

    return {
        "action_id": action_id,
        "action_name": ACTION_NAMES.get(action_id, str(action_id)),
        "proba": proba,
        "equity": equity,
        "shap_result": shap_result,
        "path_result": path_result,
        "cf_result": cf_result,
        "explanation": nl_explanations,
        "features": features,
    }


def apply_action(game, who, action_id):
    """Apply an action to the game state. who='hero' or 'ai'."""
    street = game["street"]
    raise_size = 2 if street == "preflop" else 4
    action_name = ACTION_NAMES.get(action_id, str(action_id))

    if action_id == 0:  # fold
        game["game_over"] = True
        board = [game["board_card"]] if game["street"] == "postflop" else []
        board_part = f" [Board: {card_label(*game['board_card'])}]" if board else ""
        hero_h = card_label(*game["hero_cards"][0]) + card_label(*game["hero_cards"][1])
        opp_h = card_label(*game["opp_cards"][0]) + card_label(*game["opp_cards"][1])
        if board:
            hero_combo = combo_name(game["hero_cards"], board, NUM_RANKS)
            opp_combo = combo_name(game["opp_cards"], board, NUM_RANKS)
        else:
            hero_combo = None
            opp_combo = None
        if who == "hero":
            game["result"] = "lose"
            game["result_chips"] = -game["hero_invested"]
            ai_desc = f"{opp_h}" + (f" - {opp_combo}" if opp_combo else "")
            game["actions_log"].append(
                f"You FOLD{board_part}. AI wins the pot ({game['pot']} chips). "
                f"AI had: {ai_desc}.")
        else:
            game["result"] = "win"
            game["result_chips"] = game["opp_invested"]
            hero_desc = f"{hero_h}" + (f" - {hero_combo}" if hero_combo else "")
            game["actions_log"].append(
                f"AI FOLDS{board_part}. You win the pot ({game['pot']} chips)! "
                f"Your hand: {hero_desc}.")
        return

    if action_id == 1:  # call/check
        if who == "hero":
            diff = game["opp_invested"] - game["hero_invested"]
            if diff > 0:
                game["hero_invested"] += diff
                game["pot"] += diff
                game["actions_log"].append(f"You CALL ({diff} chips).")
            else:
                game["actions_log"].append("You CHECK.")
        else:
            diff = game["hero_invested"] - game["opp_invested"]
            if diff > 0:
                game["opp_invested"] += diff
                game["pot"] += diff
                game["actions_log"].append(f"AI CALLS ({diff} chips).")
            else:
                game["actions_log"].append("AI CHECKS.")

    elif action_id == 2:  # raise/bet
        if who == "hero":
            diff = game["opp_invested"] - game["hero_invested"]
            game["hero_invested"] += diff + raise_size
            game["pot"] += diff + raise_size
            game["actions_log"].append(f"You RAISE (+{raise_size} chips).")
            if street == "preflop":
                game["hero_aggression_r1"] = 2
            else:
                game["hero_aggression_r2"] = 2
        else:
            diff = game["hero_invested"] - game["opp_invested"]
            game["opp_invested"] += diff + raise_size
            game["pot"] += diff + raise_size
            game["actions_log"].append(f"AI RAISES (+{raise_size} chips).")
            if street == "preflop":
                game["opp_aggression_r1"] = 2
            else:
                game["opp_aggression_r2"] = 2

    # Check if street is complete (both players acted, bets matched)
    if game["hero_invested"] == game["opp_invested"]:
        if street == "preflop" and not game["preflop_done"]:
            # Preflop needs at least 2 actions total to be done
            preflop_actions = sum(1 for a in game["actions_log"]
                                  if "CHECK" in a or "CALL" in a or "FOLD" in a)
            if preflop_actions >= 1 and action_id == 1:
                game["preflop_done"] = True
                if not game["game_over"]:
                    game["street"] = "postflop"
                    game["actions_log"].append("--- Community card dealt ---")
                    game["awaiting_hero"] = (game["hero_position"] == 0)
                    return
        elif street == "postflop" and not game["postflop_done"]:
            postflop_actions = sum(1 for a in game["actions_log"]
                                   if "Community card" in a
                                   for _ in [None])  # dummy
            if action_id == 1:
                game["postflop_done"] = True
                go_to_showdown(game)
                return

    # After a raise, the other player needs to respond
    if action_id == 2:
        game["awaiting_hero"] = (who == "ai")
    else:
        game["awaiting_hero"] = (who == "ai")


def combo_name(hole_cards, board_cards, num_ranks):
    """Return human-readable hand combination name."""
    cat, tiebreakers = hand_strength(hole_cards, board_cards, num_ranks)
    rank_map = RANK_LABELS
    if cat == 4:
        r = rank_map[tiebreakers[0]]
        return f"Three of a Kind ({r}s)"
    elif cat == 3:
        high = rank_map[tiebreakers[0]]
        return f"Straight ({high}-high)"
    elif cat == 2:
        pair_rank = rank_map[tiebreakers[0]]
        kicker = rank_map[tiebreakers[1]]
        return f"Pair of {pair_rank}s ({kicker} kicker)"
    else:
        high = rank_map[tiebreakers[0]]
        return f"High Card ({high})"


def go_to_showdown(game):
    """Resolve the hand at showdown."""
    game["game_over"] = True
    board = [game["board_card"]]
    result = compare_hands(
        game["hero_cards"], game["opp_cards"], board, NUM_RANKS
    )
    hero_h = card_label(*game["hero_cards"][0]) + card_label(*game["hero_cards"][1])
    opp_h = card_label(*game["opp_cards"][0]) + card_label(*game["opp_cards"][1])
    board_c = card_label(*game["board_card"])
    hero_combo = combo_name(game["hero_cards"], board, NUM_RANKS)
    opp_combo = combo_name(game["opp_cards"], board, NUM_RANKS)

    if result > 0:
        game["result"] = "win"
        game["result_chips"] = game["opp_invested"]
        game["actions_log"].append(
            f"SHOWDOWN [Board: {board_c}]: You ({hero_h} - {hero_combo}) "
            f"vs AI ({opp_h} - {opp_combo}) - YOU WIN! (+{game['opp_invested']} chips)")
    elif result < 0:
        game["result"] = "lose"
        game["result_chips"] = -game["hero_invested"]
        game["actions_log"].append(
            f"SHOWDOWN [Board: {board_c}]: You ({hero_h} - {hero_combo}) "
            f"vs AI ({opp_h} - {opp_combo}) - AI WINS. (-{game['hero_invested']} chips)")
    else:
        game["result"] = "tie"
        game["result_chips"] = 0
        game["actions_log"].append(
            f"SHOWDOWN [Board: {board_c}]: You ({hero_h} - {hero_combo}) "
            f"vs AI ({opp_h} - {opp_combo}) - TIE! (split pot)")


def render_game_tab(policy, explainers):
    """Render the interactive game tab."""
    # Initialize session state
    if "game" not in st.session_state:
        st.session_state.game = None
    if "total_chips" not in st.session_state:
        st.session_state.total_chips = 0
    if "hands_played" not in st.session_state:
        st.session_state.hands_played = 0
    if "ai_decision" not in st.session_state:
        st.session_state.ai_decision = None

    # Score display
    chip_color = "#66bb6a" if st.session_state.total_chips >= 0 else "#ef5350"
    st.markdown(
        f'<div class="score-display">'
        f'Hands: {st.session_state.hands_played} | '
        f'Chips: <span style="color:{chip_color}">{st.session_state.total_chips:+d}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    col_deal, col_reset = st.columns(2)
    with col_deal:
        if st.button("Deal New Hand", type="primary", use_container_width=True, key="deal"):
            st.session_state.game = deal_new_game()
            st.session_state.ai_decision = None
            # If AI acts first (hero is BB, AI is SB, SB acts first preflop)
            game = st.session_state.game
            if game["hero_position"] == 1:  # hero is BB, AI is SB, AI acts first
                game["awaiting_hero"] = False
            else:
                game["awaiting_hero"] = True
            st.rerun()
    with col_reset:
        if st.button("Reset Score", use_container_width=True, key="reset"):
            st.session_state.total_chips = 0
            st.session_state.hands_played = 0
            st.session_state.game = None
            st.session_state.ai_decision = None
            st.rerun()

    game = st.session_state.game
    if game is None:
        st.info("Click **Deal New Hand** to start playing!")
        return

    # If it's AI's turn, make AI act
    if not game["game_over"] and not game["awaiting_hero"]:
        decision = ai_decide(policy, explainers, game)
        st.session_state.ai_decision = decision
        apply_action(game, "ai", decision["action_id"])
        if not game["game_over"]:
            game["awaiting_hero"] = True

    # --- Render the table ---
    show_opp = game["game_over"]
    if show_opp:
        opp_cards_html = card_html(*game["opp_cards"][0]) + card_html(*game["opp_cards"][1])
    else:
        opp_cards_html = hidden_card_html() + hidden_card_html()

    hero_cards_html = card_html(*game["hero_cards"][0]) + card_html(*game["hero_cards"][1])

    if game["street"] == "postflop" or (game["game_over"] and game["preflop_done"]):
        board_html = card_html(*game["board_card"])
    else:
        board_html = '<span style="color:#999; font-size:16px;">Preflop</span>'

    pos_label = "BB (Big Blind)" if game["hero_position"] == 1 else "SB (Small Blind)"

    board_list = [game["board_card"]] if game["street"] == "postflop" else []
    bucket_name, _ = classify_hand(game["hero_cards"], board_list, NUM_RANKS)
    hero_eq_board = game["board_card"] if game["street"] == "postflop" else None
    if hero_eq_board:
        hero_equity = compute_equity_postflop(game["hero_cards"], hero_eq_board, NUM_RANKS, NUM_SUITS)
    else:
        hero_equity = compute_equity_preflop(game["hero_cards"], NUM_RANKS, NUM_SUITS)

    table_html = f"""
    <div class="poker-table">
        <div class="player-area">
            <div class="opponent-label">AI AGENT</div>
            {opp_cards_html}
        </div>
        <div class="board-area">
            <div class="board-label">BOARD</div>
            {board_html}
            <div class="pot-display">Pot: {game['pot']} chips</div>
        </div>
        <div class="player-area">
            <div class="player-label">YOU ({pos_label}) | Equity: {hero_equity:.0%} | {bucket_name.replace('_', ' ').title()}</div>
            {hero_cards_html}
        </div>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    # --- Action log ---
    if game["actions_log"]:
        log_html = '<div class="game-log">'
        for entry in game["actions_log"]:
            if "WIN" in entry or "win" in entry:
                log_html += f'<div class="result-win">{entry}</div>'
            elif "LOSE" in entry or "FOLD" in entry:
                log_html += f'<div class="result-lose">{entry}</div>'
            else:
                log_html += f'<div>{entry}</div>'
        log_html += '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

    # --- Player actions ---
    if not game["game_over"] and game["awaiting_hero"]:
        st.markdown("**Your turn:**")
        col_f, col_c, col_r = st.columns(3)
        with col_f:
            if st.button("FOLD", use_container_width=True, key="fold",
                         type="secondary"):
                apply_action(game, "hero", 0)
                st.rerun()
        with col_c:
            diff = game["opp_invested"] - game["hero_invested"]
            call_label = f"CALL ({diff})" if diff > 0 else "CHECK"
            if st.button(call_label, use_container_width=True, key="call",
                         type="primary"):
                apply_action(game, "hero", 1)
                st.rerun()
        with col_r:
            r_size = 2 if game["street"] == "preflop" else 4
            if st.button(f"RAISE (+{r_size})", use_container_width=True,
                         key="raise", type="secondary"):
                apply_action(game, "hero", 2)
                st.rerun()

    # --- Game over ---
    if game["game_over"]:
        if game["result"] == "win":
            st.success(f"You win! +{game['result_chips']} chips")
        elif game["result"] == "lose":
            st.error(f"You lose. {game['result_chips']} chips")
        else:
            st.info("Tie! Split pot.")

        # Update score (only once)
        if not game.get("scored"):
            st.session_state.total_chips += game["result_chips"]
            st.session_state.hands_played += 1
            game["scored"] = True

    # --- AI explanation (show after AI acted) ---
    ai_dec = st.session_state.ai_decision
    if ai_dec is not None:
        with st.expander("AI's Reasoning (last action)", expanded=game["game_over"]):
            action_css = {0: "action-fold", 1: "action-call", 2: "action-raise"}
            badge_class = action_css.get(ai_dec["action_id"], "action-call")
            st.markdown(
                f'<div style="text-align:center">'
                f'<div class="action-badge {badge_class}">'
                f'AI: {ai_dec["action_name"].upper()}</div></div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Action Probabilities:**")
                for aid in sorted(ai_dec["proba"].keys()):
                    aname = ACTION_NAMES.get(aid, str(aid))
                    prob = ai_dec["proba"][aid]
                    color = "#c62828" if aid == 0 else "#1565c0" if aid == 1 else "#2e7d32"
                    st.markdown(
                        f'<div style="margin:2px 0;">'
                        f'<span style="width:90px;display:inline-block;">{aname}</span>'
                        f'<b>{prob:.0%}</b>'
                        f'<div class="equity-bar"><div class="equity-fill" '
                        f'style="width:{prob*100:.0f}%;background:{color};"></div></div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                st.markdown("**Top SHAP Features:**")
                for feat in ai_dec["shap_result"]["top_features"]:
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
                st.markdown("**Full Explanation:**")
                st.markdown(ai_dec["explanation"]["full"])


# =====================================================================
# ANALYZE MODE (original)
# =====================================================================

def render_analyze_tab(policy, explainers):
    """Render the original analysis tab."""
    action_names = policy.action_names or ACTION_NAMES

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

    features, equity = build_features(hole_cards, board_card, pot, position, opp_r1, opp_r2)

    opponent_cards = hidden_card_html() + hidden_card_html()
    hero_cards_html = card_html(*hole_cards[0]) + card_html(*hole_cards[1])

    if board_card is not None:
        board_html = card_html(*board_card)
        street = "Postflop"
    else:
        board_html = '<span style="color:#999; font-size:16px;">No board yet</span>'
        street = "Preflop"

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
            {hero_cards_html}
        </div>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

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


# =====================================================================
# MAIN
# =====================================================================

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

    tab_play, tab_analyze = st.tabs(["Play vs AI", "Analyze Hand"])

    with tab_play:
        render_game_tab(policy, explainers)

    with tab_analyze:
        render_analyze_tab(policy, explainers)

    st.markdown("---")
    st.caption(
        "Bachelor Thesis: Explainable AI Based Poker Agent for Learning Purposes | "
        "Riga Technical University"
    )


if __name__ == "__main__":
    main()