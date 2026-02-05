"""Counterfactual generation via sibling branch traversal."""

import numpy as np


class CounterfactualGenerator:
    """
    Generates minimal counterfactual explanations by finding the nearest
    branch point in the decision tree where a different action would result.

    Method: walk up from the current leaf, find the first internal node
    whose sibling subtree leads to a different action.
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

    def generate(self, features):
        """
        Find the minimal counterfactual for a given state.

        Args:
            features: 1D numpy array

        Returns:
            dict with:
                - found: bool (whether a counterfactual was found)
                - feature: name of the feature that would need to change
                - current_value: current value of that feature
                - threshold: the threshold that defines the boundary
                - direction: which way the feature needs to change
                - alternative_action: what action the agent would take
                - statement: natural language counterfactual statement
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get path nodes
        node_indicator = self.policy.sklearn_tree.decision_path(features)
        path_nodes = node_indicator.indices.tolist()

        tree = self.tree
        current_action = self.policy.predict(features)
        current_action_name = self.action_names.get(current_action, str(current_action))

        # Walk up from leaf, find the first node whose sibling
        # leads to a different majority action
        for i in range(len(path_nodes) - 2, -1, -1):
            node_id = path_nodes[i]

            # Skip leaf nodes
            if tree.children_left[node_id] == tree.children_right[node_id]:
                continue

            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_val = features[0, feature_idx]

            # Determine which child we went to and which is the sibling
            went_left = feature_val <= threshold
            if went_left:
                sibling = tree.children_right[node_id]
            else:
                sibling = tree.children_left[node_id]

            # Get the majority action of the sibling subtree
            sibling_action = self._get_subtree_majority_action(sibling)

            if sibling_action != current_action:
                # Found a counterfactual
                if self.feature_names and feature_idx < len(self.feature_names):
                    feature_name = self.feature_names[feature_idx]
                else:
                    feature_name = f"feature_{feature_idx}"

                alt_action_name = self.action_names.get(
                    sibling_action, str(sibling_action)
                )

                if went_left:
                    direction = "greater than"
                    condition = f"{feature_name} were greater than {threshold:.4f}"
                else:
                    direction = "less than or equal to"
                    condition = f"{feature_name} were less than or equal to {threshold:.4f}"

                statement = (
                    f"If {condition}, "
                    f"the agent would {alt_action_name} instead of {current_action_name}."
                )

                return {
                    "found": True,
                    "feature": feature_name,
                    "feature_index": int(feature_idx),
                    "current_value": float(feature_val),
                    "threshold": float(threshold),
                    "direction": direction,
                    "current_action": current_action_name,
                    "alternative_action": alt_action_name,
                    "alternative_action_id": int(sibling_action),
                    "statement": statement,
                    "node_depth": i,
                }

        return {
            "found": False,
            "statement": "No counterfactual found — all branches lead to the same action.",
            "current_action": current_action_name,
        }

    def _get_subtree_majority_action(self, node_id):
        """
        Get the majority action class of a subtree rooted at node_id.
        Uses weighted sample count at the node.
        """
        tree = self.tree
        class_counts = tree.value[node_id][0]
        return int(np.argmax(class_counts))
