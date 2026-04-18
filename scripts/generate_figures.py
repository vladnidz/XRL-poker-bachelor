"""
Generate publication-quality matplotlib figures for the thesis.

Replaces terminal screenshots with proper scientific charts.

Usage:
    python scripts/generate_figures.py
"""

import sys
import os
import numpy as np
import pickle
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import plot_tree

from config import MODEL_DIR, DATA_DIR, ACTION_NAMES, TREE_MAX_DEPTHS, TREE_RANDOM_STATE

FIGURE_DIR = os.path.join("thesis", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def fig_mccfr_convergence():
    """Fig 4.1: MCCFR training convergence — speed over iterations."""
    print("Generating Fig 4.1: MCCFR convergence...")

    checkpoints = [250_000, 500_000, 750_000, 1_000_000]
    times_sec = [195, 390, 589, 786]
    speeds = [c / t for c, t in zip(checkpoints, times_sec)]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = '#1565c0'
    color2 = '#c62828'

    ax1.plot([c / 1000 for c in checkpoints], times_sec,
             'o-', color=color1, linewidth=2, markersize=8, label='Elapsed time')
    ax1.set_xlabel('Iterations (thousands)')
    ax1.set_ylabel('Elapsed Time (seconds)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot([c / 1000 for c in checkpoints], speeds,
             's--', color=color2, linewidth=2, markersize=8, label='Throughput')
    ax2.set_ylabel('Throughput (iterations/sec)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.set_title('MCCFR Training Convergence (1M Iterations, 24-Card Deck)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig4_1_mccfr_convergence.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig_depth_search():
    """Fig 4.2: Decision tree depth vs accuracy."""
    print("Generating Fig 4.2: Depth search...")

    depths = [5, 7, 10, 12, 15]
    train_acc = [0.767, 0.789, 0.797, 0.800, 0.802]
    cv_acc = [0.766, 0.788, 0.796, 0.798, 0.798]
    leaves = [32, 110, 520, 1113, 2045]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color1 = '#1565c0'
    color2 = '#2e7d32'
    color3 = '#ff8f00'

    ax1.plot(depths, [a * 100 for a in train_acc], 'o-', color=color1,
             linewidth=2, markersize=8, label='Train accuracy')
    ax1.plot(depths, [a * 100 for a in cv_acc], 's-', color=color2,
             linewidth=2, markersize=8, label='CV accuracy (5-fold)')
    ax1.set_xlabel('Maximum Tree Depth')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(74, 82)
    ax1.axhline(y=33.3, color='gray', linestyle=':', alpha=0.5, label='Random baseline (33.3%)')

    # Highlight selected depth
    ax1.axvline(x=12, color=color3, linestyle='--', alpha=0.6, label='Selected depth (d=12)')

    ax2 = ax1.twinx()
    ax2.bar(depths, leaves, alpha=0.15, color='gray', width=0.8, label='Leaf count')
    ax2.set_ylabel('Number of Leaves', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    ax1.set_title('CART Decision Tree: Accuracy vs. Depth')
    ax1.set_xticks(depths)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig4_2_depth_search.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig_action_distribution():
    """Fig 4.3: Training data action distribution."""
    print("Generating Fig 4.3: Action distribution...")

    data_path = os.path.join(DATA_DIR, "training_data.npz")
    if not os.path.exists(data_path):
        print("  SKIP: training_data.npz not found")
        return

    data = np.load(data_path)
    y = data["y"]

    actions, counts = np.unique(y, return_counts=True)
    action_labels = [ACTION_NAMES.get(int(a), str(a)) for a in actions]
    colors = ['#c62828', '#1565c0', '#2e7d32']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(action_labels, counts, color=colors[:len(actions)], edgecolor='white',
                  linewidth=1.5)

    for bar, count in zip(bars, counts):
        pct = count / counts.sum() * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 0.5,
                f'{count:,}\n({pct:.1f}%)', ha='center', va='center', fontsize=11,
                color='white', fontweight='bold')

    ax.set_xlabel('Action')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Training Data Action Distribution (n = {counts.sum():,})')

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig4_3_action_distribution.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig_confusion_matrix():
    """Fig 4.4: Confusion matrix of CART predictions on training data."""
    print("Generating Fig 4.4: Confusion matrix...")

    data_path = os.path.join(DATA_DIR, "training_data.npz")
    model_path = os.path.join(MODEL_DIR, "decision_tree.joblib")
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("  SKIP: data or model not found")
        return

    data = np.load(data_path)
    X, y = data["X"], data["y"]

    model_data = joblib.load(model_path)
    tree = model_data["tree"]
    y_pred = tree.predict(X)

    labels = sorted(ACTION_NAMES.keys())
    label_names = [ACTION_NAMES[l] for l in labels]

    cm = confusion_matrix(y, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, cmap='Blues', values_format=',')
    ax.set_title('CART Decision Tree Confusion Matrix (Depth 12)')
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('True Action (MCCFR)')

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig4_4_confusion_matrix.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig_shap_importance():
    """Fig 4.5: SHAP feature importance (mean absolute SHAP values)."""
    print("Generating Fig 4.5: SHAP feature importance...")

    data_path = os.path.join(DATA_DIR, "training_data.npz")
    model_path = os.path.join(MODEL_DIR, "decision_tree.joblib")
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("  SKIP: data or model not found")
        return

    import shap

    data = np.load(data_path, allow_pickle=True)
    X = data["X"]
    feature_names = data["feature_names"].tolist() if "feature_names" in data else None

    model_data = joblib.load(model_path)
    tree = model_data["tree"]

    # Use a sample for SHAP (full dataset may be slow)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(5000, len(X)), replace=False)
    X_sample = X[idx]

    explainer = shap.TreeExplainer(tree)
    shap_values = explainer.shap_values(X_sample)

    # For multi-class, shap_values is a list of arrays (one per class)
    # Each array has shape (n_samples, n_features)
    n_features = X_sample.shape[1]
    if isinstance(shap_values, list):
        # Average absolute SHAP across all classes
        mean_abs = np.zeros(n_features)
        for sv in shap_values:
            mean_abs += np.abs(sv).mean(axis=0)[:n_features]
        mean_abs /= len(shap_values)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)[:n_features]

    mean_abs = np.array(mean_abs).flatten()[:n_features]

    if feature_names and len(feature_names) == n_features:
        names = [n.replace('_', ' ').title() for n in feature_names]
    else:
        names = [f"Feature {i}" for i in range(n_features)]

    # Sort by importance
    order = list(np.argsort(mean_abs))
    sorted_names = [names[i] for i in order]
    sorted_vals = [float(mean_abs[i]) for i in order]

    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(range(len(sorted_vals)), sorted_vals, color='#1565c0',
                   edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Feature Importance (TreeSHAP, 5000-sample average)')

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig4_5_shap_importance.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig_opponent_comparison():
    """Fig 4.6: Bar chart comparing agent performance against different opponents."""
    print("Generating Fig 4.6: Opponent comparison...")

    opponents = ['Random', 'Heuristic', 'MCCFR\n(teacher)']
    mean_payoffs = [0.9371, 0.3080, 0.1642]
    win_rates   = [0.518,  0.424,  0.462]
    colors = ['#2e7d32', '#1565c0', '#c62828']

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(opponents, mean_payoffs, color=colors[:len(opponents)],
                  edgecolor='white', linewidth=1.5, width=0.5)

    for bar, val, wr in zip(bars, mean_payoffs, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f'{val:+.3f}\n({wr:.1%} win)', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Mean Payoff Per Hand (chips)')
    ax.set_title('CART Agent Performance Against Different Opponents (n=10,000 games)')

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig4_6_opponent_comparison.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def fig_tree_top_levels():
    """Fig 3.2: Top 4 levels of the decision tree (partial visualization)."""
    print("Generating Fig 3.2: Decision tree (top levels)...")

    model_path = os.path.join(MODEL_DIR, "decision_tree.joblib")
    if not os.path.exists(model_path):
        print("  SKIP: model not found")
        return

    model_data = joblib.load(model_path)
    tree = model_data["tree"]
    feature_names = model_data.get("feature_names")
    if feature_names:
        feature_names = [n.replace('_', ' ') for n in feature_names]

    class_names = ['fold', 'call/check', 'raise/bet']

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(tree, max_depth=3, feature_names=feature_names,
              class_names=class_names, filled=True, rounded=True,
              ax=ax, fontsize=8, impurity=False, proportion=True)
    ax.set_title('CART Decision Tree (Top 3 Levels of Depth-12 Tree)', fontsize=14)

    path = os.path.join(FIGURE_DIR, "fig3_2_tree_visualization.png")
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("Generating thesis figures...\n")

    fig_mccfr_convergence()
    fig_depth_search()
    fig_action_distribution()
    fig_confusion_matrix()
    fig_shap_importance()
    fig_opponent_comparison()
    fig_tree_top_levels()

    print(f"\nAll figures saved to {FIGURE_DIR}/")


if __name__ == "__main__":
    main()
