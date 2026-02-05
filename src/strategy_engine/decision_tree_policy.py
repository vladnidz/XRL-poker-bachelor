"""CART decision tree policy distillation from CFR+ strategy."""

import os
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


class DecisionTreePolicy:
    """
    Trains a CART decision tree on CFR+ generated state-action pairs.
    The tree serves as the interpretable policy for the poker agent.
    """

    def __init__(self, max_depth=10, random_state=42):
        """
        Args:
            max_depth: maximum tree depth
            random_state: random seed for reproducibility
        """
        self.max_depth = max_depth
        self.random_state = random_state
        self.tree = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced"
        )
        self.feature_names = None
        self.action_names = None
        self.is_trained = False

    def train(self, X, y, feature_names=None, action_names=None):
        """
        Train the decision tree on state-action data.

        Args:
            X: feature matrix (n_samples, n_features)
            y: action labels (n_samples,)
            feature_names: list of feature name strings
            action_names: dict {action_id: name}

        Returns:
            dict: training metrics
        """
        self.feature_names = feature_names
        self.action_names = action_names or {0: "fold", 1: "call", 2: "raise"}

        self.tree.fit(X, y)
        self.is_trained = True

        # Compute metrics
        train_accuracy = self.tree.score(X, y)
        cv_scores = cross_val_score(
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced"
            ),
            X, y, cv=5, scoring="accuracy"
        )

        metrics = {
            "train_accuracy": train_accuracy,
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "n_leaves": self.tree.get_n_leaves(),
            "max_depth_actual": self.tree.get_depth(),
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_classes": len(np.unique(y)),
        }

        print(f"Decision tree trained:")
        print(f"  Train accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  CV accuracy:    {metrics['cv_accuracy_mean']:.4f} "
              f"(+/- {metrics['cv_accuracy_std']:.4f})")
        print(f"  Leaves: {metrics['n_leaves']}, "
              f"Depth: {metrics['max_depth_actual']}")

        return metrics

    def predict(self, features):
        """
        Predict action for a single state.

        Args:
            features: 1D numpy array or 2D (1, n_features)

        Returns:
            int: predicted action
        """
        if not self.is_trained:
            raise RuntimeError("Tree not trained yet.")

        if features.ndim == 1:
            features = features.reshape(1, -1)
        return int(self.tree.predict(features)[0])

    def predict_proba(self, features):
        """
        Predict action probabilities for a single state.

        Args:
            features: 1D numpy array or 2D (1, n_features)

        Returns:
            dict: {action: probability}
        """
        if not self.is_trained:
            raise RuntimeError("Tree not trained yet.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        proba = self.tree.predict_proba(features)[0]
        classes = self.tree.classes_

        return {int(c): float(p) for c, p in zip(classes, proba)}

    def get_decision_path(self, features):
        """
        Get the decision path (list of node indices) for a given input.

        Args:
            features: 1D numpy array

        Returns:
            list of int: node indices from root to leaf
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        path = self.tree.decision_path(features)
        node_indices = path.indices.tolist()
        return node_indices

    def depth_search(self, X, y, depths=None, feature_names=None):
        """
        Train trees at multiple depths and compare performance.

        Args:
            X: feature matrix
            y: action labels
            depths: list of max_depth values to try
            feature_names: feature names

        Returns:
            list of dicts: metrics for each depth
        """
        if depths is None:
            depths = [5, 7, 10, 12, 15]

        results = []
        for d in depths:
            print(f"\n--- Depth {d} ---")
            tree = DecisionTreePolicy(max_depth=d, random_state=self.random_state)
            metrics = tree.train(X, y, feature_names=feature_names)
            metrics["max_depth_param"] = d
            results.append(metrics)

        return results

    def save(self, path):
        """Save trained tree to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "tree": self.tree,
            "feature_names": self.feature_names,
            "action_names": self.action_names,
            "max_depth": self.max_depth,
            "is_trained": self.is_trained,
        }, path)
        print(f"Decision tree saved to {path}")

    def load(self, path):
        """Load trained tree from disk."""
        data = joblib.load(path)
        self.tree = data["tree"]
        self.feature_names = data["feature_names"]
        self.action_names = data["action_names"]
        self.max_depth = data["max_depth"]
        self.is_trained = data["is_trained"]
        print(f"Decision tree loaded from {path}")

    @property
    def sklearn_tree(self):
        """Access the underlying sklearn tree for SHAP, path extraction, etc."""
        return self.tree
