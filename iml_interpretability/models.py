"""Inherently interpretable model candidates.

Three tree-based families are compared, each over a range of complexities so that the threshold
at which a model stops being interpretable becomes visible:

* ``decision_tree_depth_<d>``: one decision tree of maximum depth ``d`` (``full`` = unrestricted).
* ``random_forest_trees_<n>``: random forest of ``n`` trees with depth 3.
* ``gradient_boosting_trees_<n>``: gradient boosting with ``n`` boosting rounds and depth 2.
"""

from __future__ import annotations

from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

TREE_DEPTHS = (1, 2, 3, 4, 5, 6, 8, 10, 15, None)
FOREST_SIZES = (1, 2, 3, 5, 10, 25, 50)
BOOSTING_ROUNDS = (1, 2, 3, 5, 10, 25, 50)


def get_models(seed: int = 0) -> dict[str, ClassifierMixin]:
    """Return unfitted models, keyed by a unique name used in result tables.

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn classifier.
    :rtype: dict[str, ClassifierMixin]
    """
    models: dict[str, ClassifierMixin] = {}
    for depth in TREE_DEPTHS:
        label = "full" if depth is None else depth
        models[f"decision_tree_depth_{label}"] = DecisionTreeClassifier(max_depth=depth, random_state=seed)
    for n_trees in FOREST_SIZES:
        models[f"random_forest_trees_{n_trees}"] = RandomForestClassifier(
            n_estimators=n_trees, max_depth=3, random_state=seed
        )
    for n_rounds in BOOSTING_ROUNDS:
        models[f"gradient_boosting_trees_{n_rounds}"] = GradientBoostingClassifier(
            n_estimators=n_rounds, max_depth=2, random_state=seed
        )
    return models
