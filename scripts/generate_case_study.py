"""
Generate 5 representative case study states with full explanations.
Run: python scripts/generate_case_study.py
"""
import numpy as np
import joblib
import warnings
import sys, os
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'src'))

import shap
from sklearn.tree import _tree

DATA_PATH  = os.path.join(BASE, "data",   "training_data.npz")
MODEL_PATH = os.path.join(BASE, "models", "decision_tree.joblib")

FEATURE_NAMES = [
    "equity", "pot_odds", "stack_to_pot", "position",
    "is_suited", "hole_rank1", "hole_rank2", "board_rank",
    "opp_aggression_r1", "opp_aggression_r2",
    "future_eq_d1", "future_eq_d2", "future_eq_d3", "future_eq_d4",
    "future_eq_d5", "future_eq_d6", "future_eq_d7", "future_eq_d8",
    "future_eq_d9", "future_eq_d10",
    "max_hole_rank", "min_hole_rank", "hand_strength_bucket", "street",
]
ACTION_NAMES = {0: "FOLD", 1: "CALL/CHECK", 2: "RAISE/BET"}
FEATURE_DISPLAY = {
    "equity": "Equity",
    "pot_odds": "Pot Odds",
    "stack_to_pot": "Stack-to-Pot Ratio",
    "position": "Position",
    "is_suited": "Suited Hole Cards",
    "opp_aggression_r1": "Opponent Aggression (Round 1)",
    "opp_aggression_r2": "Opponent Aggression (Round 2)",
    "future_eq_d8": "Future Equity Decile 8",
    "future_eq_d5": "Future Equity Decile 5",
    "future_eq_d2": "Future Equity Decile 2",
    "max_hole_rank": "Max Hole Card Rank",
    "min_hole_rank": "Min Hole Card Rank",
    "hand_strength_bucket": "Hand Strength Bucket",
    "street": "Street",
}

data = np.load(DATA_PATH)
X, y = data['X'], data['y']
model_data = joblib.load(MODEL_PATH)
tree = model_data['tree'] if isinstance(model_data, dict) else model_data
n_features = X.shape[1]
feat_names = FEATURE_NAMES[:n_features]

explainer = shap.TreeExplainer(tree)

def get_decision_path(x):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    node = 0
    path_parts = []
    while feature[node] != _tree.TREE_UNDEFINED:
        feat_idx = feature[node]
        thr = threshold[node]
        fname = feat_names[feat_idx] if feat_idx < len(feat_names) else f"f{feat_idx}"
        fname_disp = FEATURE_DISPLAY.get(fname, fname)
        if x[feat_idx] <= thr:
            path_parts.append(f"{fname_disp} <= {thr:.4f}")
            node = tree_.children_left[node]
        else:
            path_parts.append(f"{fname_disp} > {thr:.4f}")
            node = tree_.children_right[node]
    pred = np.argmax(tree_.value[node][0])
    return path_parts, pred

def get_counterfactual(x):
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold
    node = 0
    path = []
    while feature[node] != _tree.TREE_UNDEFINED:
        feat_idx = feature[node]
        thr = threshold[node]
        path.append((node, feat_idx, thr, x[feat_idx] <= thr))
        node = tree_.children_left[node] if x[feat_idx] <= thr else tree_.children_right[node]
    orig_class = np.argmax(tree_.value[node][0])
    for node_id, feat_idx, thr, went_left in reversed(path):
        if went_left:
            cf_val = thr + 1e-4
            sibling = tree_.children_right[node_id]
        else:
            cf_val = thr - 1e-4
            sibling = tree_.children_left[node_id]
        new_class = np.argmax(tree_.value[sibling][0])
        if new_class != orig_class:
            fname = feat_names[feat_idx] if feat_idx < len(feat_names) else f"f{feat_idx}"
            fname_disp = FEATURE_DISPLAY.get(fname, fname)
            direction = "greater than" if went_left else "less than"
            return fname_disp, x[feat_idx], cf_val, ACTION_NAMES[orig_class], ACTION_NAMES[new_class], direction, thr
    return None

# Select 5 diverse, clear states: one per action class + two edge cases
# Criteria: high confidence prediction, diverse equity values
rng = np.random.RandomState(42)
cases = []

for target_class, equity_range, label in [
    (0, (0.0, 0.35), "Clear fold — low equity"),
    (1, (0.42, 0.58), "Call — medium equity, in position"),
    (1, (0.55, 0.70), "Call — decent equity, passive opponent"),
    (2, (0.70, 0.90), "Raise — high equity, favorable pot odds"),
    (2, (0.60, 0.75), "Raise — moderate equity, aggressive spot"),
]:
    mask = (
        (y == target_class) &
        (X[:, 0] >= equity_range[0]) &
        (X[:, 0] <= equity_range[1])
    )
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        continue
    probs = tree.predict_proba(X[idxs])
    confident = idxs[probs[:, target_class] > 0.55]
    if len(confident) == 0:
        confident = idxs
    chosen = rng.choice(confident)
    cases.append((chosen, label))

print(f"Selected {len(cases)} case study states\n")
print("=" * 70)

results = []
for idx, label in cases:
    x = X[idx]
    pred_class = tree.predict([x])[0]
    pred_action = ACTION_NAMES[pred_class]
    probs = tree.predict_proba([x])[0]

    sv = explainer.shap_values(x.reshape(1, -1))
    if isinstance(sv, list):
        sv_mean = np.mean([np.abs(sv[c][0]) for c in range(len(sv))], axis=0)
        sv_signed = sv[pred_class][0]
    elif sv.ndim == 3:
        sv_mean = np.mean(np.abs(sv[0]), axis=1)
        sv_signed = sv[0, :, pred_class]
    else:
        sv_mean = np.abs(sv[0])
        sv_signed = sv[0]

    top3 = np.argsort(sv_mean)[::-1][:3]
    path_parts, _ = get_decision_path(x)
    cf = get_counterfactual(x)

    equity_pct = int(x[0] * 100)
    pot_odds_val = x[1]
    position_str = "IP (last to act)" if x[3] > 0.5 else "OOP (first to act)"
    street_str = "Post-flop" if (n_features > 23 and x[23] > 0.5) else "Pre-flop"

    print(f"State {len(results)+1}: {label}")
    print(f"  Equity: {equity_pct}% | Pot Odds: {pot_odds_val:.2f} | Position: {position_str}")
    print(f"  Agent Action: {pred_action} (confidence: {probs[pred_class]*100:.0f}%)")
    print(f"  Top SHAP features:")
    shap_rows = []
    for fi in top3:
        fname = feat_names[fi] if fi < len(feat_names) else f"f{fi}"
        fname_disp = FEATURE_DISPLAY.get(fname, fname)
        influence = sv_signed[fi] if fi < len(sv_signed) else sv_mean[fi]
        direction = "supports" if influence > 0 else "opposes"
        shap_rows.append((fname_disp, x[fi], influence, direction))
        print(f"    {fname_disp} = {x[fi]:.4f} ({direction} {pred_action}, influence {influence:+.4f})")
    path_str = " => ".join(path_parts[:6]) + (" => ..." if len(path_parts) > 6 else f" => {pred_action}")
    print(f"  Decision path ({len(path_parts)} nodes): {path_str[:120]}")
    if cf:
        fname_disp, orig_val, cf_val, orig_act, new_act, direction, thr = cf
        print(f"  Counterfactual: If {fname_disp} were {direction} {thr:.4f}, agent would {new_act} instead of {orig_act}.")
    print()

    results.append({
        "label": label,
        "equity_pct": equity_pct,
        "pot_odds": pot_odds_val,
        "position": position_str,
        "action": pred_action,
        "confidence": probs[pred_class],
        "shap_rows": shap_rows,
        "path": path_parts,
        "path_str": path_str,
        "cf": cf,
        "pred_class": pred_class,
    })

# Save for docx generation
import json
out = os.path.join(BASE, "scripts", "case_study_data.json")
def to_serializable(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, tuple): return list(obj)
    return obj

import json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open(out, 'w') as f:
    json.dump(results, f, cls=NpEncoder, indent=2)
print(f"Saved case study data to {out}")
