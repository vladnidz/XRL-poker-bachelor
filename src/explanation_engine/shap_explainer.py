"""TreeSHAP feature attribution for the decision tree policy."""

import numpy as np
import shap


class SHAPExplainer:
    """
    Computes exact Shapley values for decision tree predictions
    using TreeSHAP (Lundberg et al., 2020).

    Complexity: O(TLD^2) per prediction — polynomial, real-time feasible.
    """

    def __init__(self, decision_tree_policy):
        """
        Args:
            decision_tree_policy: DecisionTreePolicy instance (must be trained)
        """
        self.policy = decision_tree_policy
        self.explainer = shap.TreeExplainer(
            decision_tree_policy.sklearn_tree
        )
        self.feature_names = decision_tree_policy.feature_names

    def explain(self, features, top_k=3):
        """
        Compute SHAP values for a single prediction.

        Args:
            features: 1D numpy array (feature vector for one state)
            top_k: number of top contributing features to highlight

        Returns:
            dict with:
                - shap_values: dict {feature_name: shap_value} per class
                - predicted_action: int
                - top_features: list of (feature_name, value, shap_value)
                  for the predicted class, sorted by |shap|
                - base_value: expected value (base rate)
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        shap_values = self.explainer.shap_values(features)
        predicted_action = self.policy.predict(features)

        # Normalize shap_values shape to per-class arrays
        # Newer SHAP: (n_samples, n_features, n_classes) single ndarray
        # Older SHAP: list of (n_samples, n_features), one per class
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Shape: (1, n_features, n_classes)
            n_classes = shap_values.shape[2]
            class_idx = min(predicted_action, n_classes - 1)
            sv_for_predicted = shap_values[0, :, class_idx]
            all_sv = {
                cls: shap_values[0, :, cls].tolist()
                for cls in range(n_classes)
            }
        elif isinstance(shap_values, list):
            sv_for_predicted = shap_values[predicted_action][0]
            all_sv = {
                cls: shap_values[cls][0].tolist()
                for cls in range(len(shap_values))
            }
        else:
            # Binary or single-class: (1, n_features)
            sv_for_predicted = shap_values[0]
            all_sv = {0: shap_values[0].tolist()}

        # Build feature attribution dict
        feature_attributions = {}
        if self.feature_names:
            for i, name in enumerate(self.feature_names):
                feature_attributions[name] = float(sv_for_predicted[i])
        else:
            for i in range(len(sv_for_predicted)):
                feature_attributions[f"feature_{i}"] = float(sv_for_predicted[i])

        # Top-k features by absolute SHAP value
        sorted_features = sorted(
            feature_attributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_features = []
        for name, sv in sorted_features[:top_k]:
            if self.feature_names:
                idx = self.feature_names.index(name)
            else:
                idx = int(name.split("_")[-1])
            top_features.append({
                "feature": name,
                "value": float(features[0, idx]),
                "shap_value": sv,
                "direction": "supports" if sv > 0 else "opposes"
            })

        # Base value
        base_value = self.explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            idx = min(predicted_action, len(base_value) - 1)
            base_value = base_value[idx]

        return {
            "shap_values": feature_attributions,
            "all_class_shap": all_sv,
            "predicted_action": predicted_action,
            "top_features": top_features,
            "base_value": float(base_value),
        }
