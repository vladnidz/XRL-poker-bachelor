"""Natural language explanation generation using Jinja2 templates."""

from jinja2 import Environment, FileSystemLoader, BaseLoader
import os


# Inline templates (no external file dependency)
EXPLANATION_TEMPLATE = """The agent {{ action }}s.
{%- if top_features|length > 0 %}
 {{ top_features[0].feature | replace('_', ' ') | title }} is \
{% if top_features[0].direction == 'supports' %}favorable{% else %}unfavorable{% endif %} \
at {{ "%.2f"|format(top_features[0].value) }} \
(influence: {{ "%+.2f"|format(top_features[0].shap_value) }}), \
which is the primary factor in this decision.
{%- endif %}
{%- if top_features|length > 1 %}
 {{ top_features[1].feature | replace('_', ' ') | title }} \
at {{ "%.2f"|format(top_features[1].value) }} \
also {{ top_features[1].direction }} this action \
(influence: {{ "%+.2f"|format(top_features[1].shap_value) }}).
{%- endif %}
{%- if counterfactual and counterfactual.found %}
 {{ counterfactual.statement }}
{%- endif %}"""

DECISION_PATH_TEMPLATE = """Decision reasoning: {% for node in path_nodes %}\
{{ node.feature | replace('_', ' ') }} {{ node.direction }} {{ "%.4f"|format(node.threshold) }}\
{% if not loop.last %} => {% endif %}\
{% endfor %} => {{ leaf_action | upper }}"""

FULL_EXPLANATION_TEMPLATE = """## Agent Decision: {{ action | upper }}

**Why this action?**
{%- if top_features|length > 0 %}
- **{{ top_features[0].feature | replace('_', ' ') | title }}** = {{ "%.2f"|format(top_features[0].value) }} \
(influence: {{ "%+.2f"|format(top_features[0].shap_value) }}) — \
{% if top_features[0].direction == 'supports' %}supports{% else %}opposes{% endif %} {{ action }}ing
{%- endif %}
{%- if top_features|length > 1 %}
- **{{ top_features[1].feature | replace('_', ' ') | title }}** = {{ "%.2f"|format(top_features[1].value) }} \
(influence: {{ "%+.2f"|format(top_features[1].shap_value) }}) — \
{% if top_features[1].direction == 'supports' %}supports{% else %}opposes{% endif %} {{ action }}ing
{%- endif %}
{%- if top_features|length > 2 %}
- **{{ top_features[2].feature | replace('_', ' ') | title }}** = {{ "%.2f"|format(top_features[2].value) }} \
(influence: {{ "%+.2f"|format(top_features[2].shap_value) }}) — \
{% if top_features[2].direction == 'supports' %}supports{% else %}opposes{% endif %} {{ action }}ing
{%- endif %}

**Decision path:**
{% for node in path_nodes %}\
{{ node.feature | replace('_', ' ') }} {{ node.direction }} {{ "%.4f"|format(node.threshold) }}\
{% if not loop.last %} => {% endif %}\
{% endfor %} => {{ action | upper }}

{%- if counterfactual and counterfactual.found %}

**What would change the decision?**
{{ counterfactual.statement }}
{%- endif %}"""


class NLGenerator:
    """
    Generates natural language explanations from structured explanation data
    using Jinja2 templates.
    """

    def __init__(self, template_dir=None):
        """
        Args:
            template_dir: directory containing .j2 template files.
                          If None, uses built-in inline templates.
        """
        if template_dir and os.path.isdir(template_dir):
            self.env = Environment(loader=FileSystemLoader(template_dir))
            self.use_files = True
        else:
            self.env = Environment(loader=BaseLoader())
            self.use_files = False

        # Register inline templates
        self._templates = {
            "brief": self.env.from_string(EXPLANATION_TEMPLATE),
            "path": self.env.from_string(DECISION_PATH_TEMPLATE),
            "full": self.env.from_string(FULL_EXPLANATION_TEMPLATE),
        }

    def generate(self, shap_result, path_result, counterfactual_result,
                 action_names=None, template="full"):
        """
        Generate a natural language explanation.

        Args:
            shap_result: output from SHAPExplainer.explain()
            path_result: output from DecisionPathExtractor.extract()
            counterfactual_result: output from CounterfactualGenerator.generate()
            action_names: dict {action_id: name}
            template: "brief", "path", or "full"

        Returns:
            str: natural language explanation
        """
        if action_names is None:
            action_names = {0: "fold", 1: "call", 2: "raise"}

        action_id = shap_result["predicted_action"]
        action = action_names.get(action_id, str(action_id))

        context = {
            "action": action,
            "action_id": action_id,
            "top_features": shap_result["top_features"],
            "shap_values": shap_result["shap_values"],
            "path_nodes": path_result["path_nodes"],
            "leaf_action": path_result["leaf_action"],
            "path_length": path_result["path_length"],
            "counterfactual": counterfactual_result,
        }

        tmpl = self._templates.get(template, self._templates["full"])
        return tmpl.render(**context).strip()

    def generate_all(self, shap_result, path_result, counterfactual_result,
                     action_names=None):
        """
        Generate all explanation formats.

        Returns:
            dict: {"brief": str, "path": str, "full": str}
        """
        return {
            name: self.generate(
                shap_result, path_result, counterfactual_result,
                action_names=action_names, template=name
            )
            for name in self._templates
        }
