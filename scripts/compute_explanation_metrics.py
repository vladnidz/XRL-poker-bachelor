"""
Compute quantitative explanation quality metrics for Section 4.3.
Run: docker compose run --rm app python scripts/compute_explanation_metrics.py
"""
import numpy as np
import joblib
import sys, os
sys.path.insert(0, '/app/src')

from sklearn.tree import DecisionTreeClassifier
import shap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE, "data",   "training_data.npz")
MODEL_PATH = os.path.join(BASE, "models", "decision_tree.joblib")

FEATURE_NAMES = [
    "equity", "pot_odds", "stack_to_pot", "position",
    "is_suited", "hole_rank1", "hole_rank2", "board_rank",
    "opp_aggression_r1", "opp_aggression_r2",
    "future_eq_d1", "future_eq_d2", "future_eq_d3", "future_eq_d4",
    "future_eq_d5", "future_eq_d6", "future_eq_d7", "future_eq_d8",
    "future_eq_d9", "future_eq_d10",
    "max_hole_rank", "min_hole_rank", "hand_strength_bucket",
    "street",
]

print("Loading data and model...")
data = np.load(DATA_PATH)
X = data['X']
y = data['y']
tree: DecisionTreeClassifier = joblib.load(MODEL_PATH)

n_features = X.shape[1]
feature_names = FEATURE_NAMES[:n_features]
n_samples = len(X)
if isinstance(tree, dict):
    tree = tree['tree']

print(f"Dataset: {n_samples} states, {n_features} features, {len(np.unique(y))} classes")

# -----------------------------------------------------------------------
# 1. SHAP consistency — same predicted action → similar SHAP vectors
#    Metric: mean pairwise cosine similarity within each action class
#    Sample 500 per class to keep it fast
# -----------------------------------------------------------------------
print("\n[1] SHAP consistency...")
explainer = shap.TreeExplainer(tree)

N_PER_CLASS = 500
cosine_sims = []
for cls in sorted(np.unique(y)):
    idx = np.where(y == cls)[0]
    if len(idx) > N_PER_CLASS:
        idx = np.random.RandomState(42).choice(idx, N_PER_CLASS, replace=False)
    X_cls = X[idx]
    sv = explainer.shap_values(X_cls)           # (n, n_feat, n_classes) or list
    if isinstance(sv, list):
        sv_abs = np.mean([np.abs(sv[c]) for c in range(len(sv))], axis=0)
    elif sv.ndim == 3:
        sv_abs = np.mean(np.abs(sv), axis=2)    # (n, n_feat)
    else:
        sv_abs = np.abs(sv)

    # pairwise cosine similarity on a sample of 200 pairs
    rng = np.random.RandomState(7)
    pairs = rng.choice(len(sv_abs), size=(200, 2), replace=True)
    sims = []
    for i, j in pairs:
        a, b = sv_abs[i].ravel(), sv_abs[j].ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 0:
            sims.append(np.dot(a, b) / denom)
    mean_sim = np.mean(sims)
    cosine_sims.append(mean_sim)
    print(f"  Class {cls}: mean cosine similarity = {mean_sim:.4f}  (n={len(idx)})")

overall_shap_consistency = np.mean(cosine_sims)
print(f"  Overall SHAP consistency: {overall_shap_consistency:.4f}")

# -----------------------------------------------------------------------
# 2. Counterfactual validity rate
#    For each state, generate counterfactual via tree path sibling check.
#    A counterfactual is "valid" if the feature change is domain-valid:
#      equity in [0,1], pot_odds >= 0, stack_to_pot >= 0, position in {0,1}
# -----------------------------------------------------------------------
print("\n[2] Counterfactual validity...")
from sklearn.tree import _tree

def get_counterfactual(tree_clf, x, feature_names):
    """Return (feature_name, original_val, cf_val, new_class) or None."""
    tree_ = tree_clf.tree_
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

    # Try flipping the last decision in the path
    for node_id, feat_idx, thr, went_left in reversed(path):
        if went_left:
            cf_val = thr + 1e-6  # just above threshold → go right
            sibling = tree_.children_right[node_id]
        else:
            cf_val = thr - 1e-6  # just below threshold → go left
            sibling = tree_.children_left[node_id]

        new_class = np.argmax(tree_.value[sibling][0])
        if new_class != orig_class:
            return feature_names[feat_idx], x[feat_idx], cf_val, new_class

    return None

DOMAIN_BOUNDS = {
    "equity":           (0.0, 1.0),
    "pot_odds":         (0.0, 1.0),
    "stack_to_pot":     (0.0, None),
    "position":         (0.0, 1.0),
    "is_suited":        (0.0, 1.0),
    "hole_rank1":       (0.0, 1.0),
    "hole_rank2":       (0.0, 1.0),
    "board_rank":       (0.0, 1.0),
    "opp_aggression_r1":(0.0, None),
    "opp_aggression_r2":(0.0, None),
}

N_CF = 1000
rng = np.random.RandomState(42)
sample_idx = rng.choice(n_samples, N_CF, replace=False)

valid = 0
produced = 0
for i in sample_idx:
    cf = get_counterfactual(tree, X[i], feature_names)
    if cf is None:
        continue
    produced += 1
    feat_name, orig, new_val, _ = cf
    bounds = DOMAIN_BOUNDS.get(feat_name, (None, None))
    lo, hi = bounds
    ok = True
    if lo is not None and new_val < lo:
        ok = False
    if hi is not None and new_val > hi:
        ok = False
    if ok:
        valid += 1

validity_rate = valid / produced if produced > 0 else 0
print(f"  Counterfactuals produced: {produced}/{N_CF} ({100*produced/N_CF:.1f}%)")
print(f"  Domain-valid:             {valid}/{produced} ({100*validity_rate:.1f}%)")

# -----------------------------------------------------------------------
# 3. Decision path length distribution
#    Shorter paths = simpler, more interpretable explanations
# -----------------------------------------------------------------------
print("\n[3] Decision path lengths...")
node_indicator = tree.decision_path(X)
path_lengths = np.diff(node_indicator.indptr)  # nodes visited per sample

print(f"  Mean path length:   {path_lengths.mean():.2f} nodes")
print(f"  Median path length: {np.median(path_lengths):.1f} nodes")
print(f"  Min / Max:          {path_lengths.min()} / {path_lengths.max()} nodes")
print(f"  Paths <= 7 nodes:    {100*(path_lengths <= 7).mean():.1f}%")
print(f"  Paths <= 10 nodes:   {100*(path_lengths <= 10).mean():.1f}%")

# -----------------------------------------------------------------------
# 4. Feature grounding — confirm top-5 SHAP features are named concepts
# -----------------------------------------------------------------------
print("\n[4] Feature grounding (top SHAP features)...")
sv_all = explainer.shap_values(X[:2000])
if isinstance(sv_all, list):
    mean_abs = np.mean([np.abs(sv_all[c]) for c in range(len(sv_all))], axis=0).mean(axis=0)
elif sv_all.ndim == 3:
    mean_abs = np.mean(np.abs(sv_all), axis=(0, 2))  # mean over samples and classes
else:
    mean_abs = np.abs(sv_all).mean(axis=0)
mean_abs = mean_abs.ravel()[:n_features]

top5_idx = np.argsort(mean_abs)[::-1][:5]
print("  Top 5 features by mean |SHAP|:")
for rank, idx in enumerate(top5_idx, 1):
    name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
    print(f"    {rank}. {name:30s}  mean |SHAP| = {mean_abs[idx]:.5f}")

# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY — Explanation Quality Metrics")
print("="*60)
print(f"  SHAP consistency (mean cosine sim):    {overall_shap_consistency:.4f}")
print(f"  Counterfactual coverage:               {100*produced/N_CF:.1f}%")
print(f"  Counterfactual domain validity:        {100*validity_rate:.1f}%")
print(f"  Mean decision path length:             {path_lengths.mean():.2f} nodes")
print(f"  Paths <= 8 nodes (interpretable):       {100*(path_lengths <= 8).mean():.1f}%")
print("="*60)
