"""Decision path extraction from the trained decision tree."""

import numpy as np


class DecisionPathExtractor:
    """
    Extracts the exact root-to-leaf path from the decision tree
    for a given input, producing a human-readable IF-THEN chain.
    """

    def __init__(self, decision_tree_policy):
        """
        Args:
            decision_tree_policy: DecisionTreePolicy instance (must be trained)
        """
        self.policy = decision_tree_policy
        self.tree = decision_tree_policy.sklearn_tree.tree_
        self.feature_names = decision_tree_policy.feature_names
        self.action_names = decision_tree_policy.action_names

    def extract(self, features):
        """
        Extract the decision path for a single input.

        Args:
            features: 1D numpy array

        Returns:
            dict with:
                - path_nodes: list of node dicts with feature, threshold, direction
                - leaf_action: predicted action name
                - path_string: human-readable string
                - path_length: number of decisions
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get node indices on the path
        node_indicator = self.policy.sklearn_tree.decision_path(features)
        node_indices = node_indicator.indices

        tree = self.tree
        path_nodes = []

        for i, node_id in enumerate(node_indices):
            # Skip the leaf node (no split)
            if tree.children_left[node_id] == tree.children_right[node_id]:
                # Leaf node
                continue

            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_val = features[0, feature_idx]

            if self.feature_names and feature_idx < len(self.feature_names):
                feature_name = self.feature_names[feature_idx]
            else:
                feature_name = f"feature_{feature_idx}"

            went_left = feature_val <= threshold
            direction = "<=" if went_left else ">"

            path_nodes.append({
                "node_id": int(node_id),
                "feature": feature_name,
                "feature_index": int(feature_idx),
                "threshold": float(threshold),
                "feature_value": float(feature_val),
                "direction": direction,
                "went_left": went_left,
            })

        # Get the leaf prediction
        predicted_action = self.policy.predict(features)
        action_name = self.action_names.get(predicted_action, str(predicted_action))

        # Build readable string
        conditions = []
        for node in path_nodes:
            conditions.append(
                f"{node['feature']} {node['direction']} {node['threshold']:.4f}"
            )
        path_string = " => ".join(conditions) + f" => {action_name.upper()}"

        return {
            "path_nodes": path_nodes,
            "leaf_action": action_name,
            "leaf_action_id": predicted_action,
            "path_string": path_string,
            "path_length": len(path_nodes),
        }
