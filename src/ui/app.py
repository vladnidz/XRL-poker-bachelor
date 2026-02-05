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


# --- Page config ---
st.set_page_config(
    page_title="XRL Poker Agent",
    page_icon="\u2660",
    layout="wide"
)


@st.cache_resource
def load_agent():
    """Load the trained decision tree policy and explanation components."""
    model_path = os.path.join("models", "decision_tree.joblib")
    if not os.path.exists(model_path):
        return None
    policy = DecisionTreePolicy()
    policy.load(model_path)
    return policy


def init_explainers(policy):
    """Initialize explanation engine components."""
    return {
        "shap": SHAPExplainer(policy),
        "path": DecisionPathExtractor(policy),
        "counterfactual": CounterfactualGenerator(policy),
        "nl": NLGenerator(),
    }


def build_feature_inputs(n_features, feature_names):
    """
    Build sidebar inputs dynamically based on the model's feature count.
    Returns a numpy feature vector matching the model.
    """
    st.sidebar.header("Game State")
    values = []

    for i in range(n_features):
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        label = name.replace("_", " ").title()

        # Detect feature type from name and set appropriate widget
        if "equity" in name.lower() or "pot_odds" in name.lower():
            val = st.sidebar.slider(label, 0.0, 1.0, 0.5, 0.01, key=f"feat_{i}")
        elif "position" in name.lower():
            val = st.sidebar.selectbox(
                label, [0, 1], key=f"feat_{i}",
                format_func=lambda x: "Out of Position" if x == 0 else "In Position"
            )
            val = float(val)
        elif "stack" in name.lower() or "spr" in name.lower():
            val = st.sidebar.slider(label, 0.0, 100.0, 50.0, 0.5, key=f"feat_{i}")
        elif "bet" in name.lower() or "history" in name.lower() or "aggression" in name.lower():
            val = st.sidebar.selectbox(
                label, [0, 1, 2], key=f"feat_{i}",
                format_func=lambda x: ["None", "Passive", "Aggressive"][x]
            )
            val = float(val)
        else:
            val = st.sidebar.slider(label, 0.0, 1.0, 0.5, 0.01, key=f"feat_{i}")

        values.append(val)

    return np.array(values, dtype=np.float32)


def main():
    st.title("\u2660 Explainable Poker Agent")
    st.markdown(
        "An AI agent that plays Mini Heads-Up Limit Hold'em "
        "and **explains its decisions** for learning purposes."
    )

    # Load model
    policy = load_agent()

    if policy is None:
        st.warning(
            "No trained model found. Run the training pipeline first:\n\n"
            "```\n"
            "docker compose --profile train run train\n"
            "```"
        )
        st.stop()

    explainers = init_explainers(policy)
    n_features = policy.tree.n_features_in_
    feature_names = policy.feature_names or [f"feature_{i}" for i in range(n_features)]
    action_names = policy.action_names or {0: "fold", 1: "call", 2: "raise"}

    # Build feature vector from sidebar inputs
    features = build_feature_inputs(n_features, feature_names)

    # --- Main area ---
    if st.sidebar.button("Get Agent Decision", type="primary"):
        # Predict
        action_id = policy.predict(features)
        action_name = action_names.get(action_id, str(action_id))

        # Explain
        shap_result = explainers["shap"].explain(features)
        path_result = explainers["path"].extract(features)
        cf_result = explainers["counterfactual"].generate(features)
        explanations = explainers["nl"].generate_all(
            shap_result, path_result, cf_result, action_names=action_names
        )

        # Display results
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader(f"Agent Action: {action_name.upper()}")

            # Action probabilities
            proba = policy.predict_proba(features)
            st.markdown("**Action Probabilities:**")
            for aid, prob in sorted(proba.items()):
                aname = action_names.get(aid, str(aid))
                bar = "\u2588" * int(prob * 30)
                st.text(f"  {aname:6s} {prob:.1%} {bar}")

            # SHAP attribution table
            st.markdown("**Feature Attribution (SHAP):**")
            shap_data = []
            for feat in shap_result["top_features"]:
                shap_data.append({
                    "Feature": feat["feature"].replace("_", " ").title(),
                    "Value": f"{feat['value']:.4f}",
                    "Influence": f"{feat['shap_value']:+.4f}",
                    "Direction": feat["direction"].title(),
                })
            if shap_data:
                st.table(shap_data)

        with col2:
            # Full explanation
            st.markdown(explanations["full"])

        # Decision path (collapsible)
        with st.expander("Full Decision Path"):
            st.code(path_result["path_string"])
            st.text(f"Path length: {path_result['path_length']} decisions")

    else:
        st.info("Adjust the game state in the sidebar and click **Get Agent Decision**.")

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Bachelor Thesis: Explainable AI Based Poker Agent for Learning Purposes | "
        "Riga Technical University"
    )


if __name__ == "__main__":
    main()