"""Interpretability metrics for tree models (decision trees, random forests, gradient boosting).

The metric implemented here is the *chunk load*. It measures how much a person has to hold in mind
to use a tree model, in units of "chunks" (Miller's 7 +/- 2 limit of working memory):

* **Local load**: to reproduce one prediction, a person walks the decision path of every tree. All
  split conditions on the same feature along one path collapse into one interval, so a path costs
  one chunk per *distinct feature* it tests, plus one chunk for the leaf output it ends in. A model
  with several trees adds one chunk for the aggregation step (vote or sum). The local load is the
  mean over the samples of the dataset.
* **Global load**: to audit the whole model, a person has to find the rule that applies among R
  rules (the leaves of all trees). Singling out one of R rules takes log2(R) bits, which is added
  as chunk-equivalents.

``chunk_load = local_load + global_load``. The score maps it into ``(0, 1]``::

    score = capacity / (capacity + chunk_load)      with capacity = 7

A model counts as interpretable when ``chunk_load <= capacity``, i.e. ``score >= 0.5``. Higher
scores mean more interpretable models.

The metric goes beyond the number of features: it depends on the paths that samples actually
take, on repeated use of one feature along a path, on the number of trees and on the number of
rules. It applies to every model whose predictions are made by scikit-learn decision trees.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.base import ClassifierMixin
from sklearn.tree import BaseDecisionTree

from iml_interpretability.data import Dataset

#: Working-memory capacity in chunks (Miller, 1956: 7 +/- 2).
CAPACITY = 7.0


class InterpretabilityMetric(ABC):
    """Base class for a metric that quantifies how interpretable a fitted model is."""

    #: Column name under which the score is reported.
    name: str

    @abstractmethod
    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        """Compute the interpretability score of a fitted model.

        :param model: A model fitted on ``dataset``.
        :param dataset: The dataset the model was fitted on.
        :return: The interpretability score. Document whether higher means more or less interpretable.
        :rtype: float
        """


def get_trees(model: ClassifierMixin) -> list[BaseDecisionTree]:
    """Return all decision trees that contribute to the predictions of a fitted model.

    Supports a single decision tree and tree ensembles such as random forests, extra trees and
    gradient boosting (for multiclass gradient boosting, one tree per class and boosting round).

    :param model: A fitted scikit-learn tree model or tree ensemble.
    :return: The individual fitted trees.
    :rtype: list[BaseDecisionTree]
    :raises TypeError: If the model is not a tree or an ensemble of trees.
    """
    if isinstance(model, BaseDecisionTree):
        return [model]
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        trees = list(np.asarray(estimators, dtype=object).ravel())
        if trees and all(isinstance(tree, BaseDecisionTree) for tree in trees):
            return trees
    raise TypeError(f"Chunk load only supports trees and tree ensembles, got {type(model).__name__}")


def _distinct_features_on_path(tree: BaseDecisionTree, X: pd.DataFrame) -> np.ndarray:
    """Number of distinct features tested on the decision path of each sample."""
    if not hasattr(tree, "feature_names_in_"):
        X = X.to_numpy(dtype=np.float32)
    structure = tree.tree_
    internal = np.flatnonzero(structure.feature >= 0)
    node_to_feature = csr_matrix(
        (np.ones(len(internal)), (internal, structure.feature[internal])),
        shape=(structure.node_count, structure.n_features),
    )
    return (tree.decision_path(X) @ node_to_feature).getnnz(axis=1)


class LocalChunkLoad(InterpretabilityMetric):
    """Mean number of chunks needed to reproduce one prediction (lower is more interpretable)."""

    name = "local_load"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        trees = get_trees(model)
        per_sample = np.zeros(len(dataset.X))
        for tree in trees:
            per_sample += _distinct_features_on_path(tree, dataset.X) + 1
        if len(trees) > 1:
            per_sample += 1
        return float(per_sample.mean())


class GlobalChunkLoad(InterpretabilityMetric):
    """Bits needed to single out the applicable rule among all leaves (lower is more interpretable)."""

    name = "global_load"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        n_rules = sum(tree.get_n_leaves() for tree in get_trees(model))
        return float(np.log2(n_rules))


class ChunkLoad(InterpretabilityMetric):
    """Local plus global chunk load (lower is more interpretable)."""

    name = "chunk_load"

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        return LocalChunkLoad().score(model, dataset) + GlobalChunkLoad().score(model, dataset)


class InterpretabilityScore(InterpretabilityMetric):
    """Chunk load mapped to ``(0, 1]``; higher is more interpretable, 0.5 is the threshold.

    :param capacity: Working-memory capacity in chunks. The model counts as interpretable if its
        chunk load does not exceed this value.
    """

    name = "interpretability"

    def __init__(self, capacity: float = CAPACITY) -> None:
        self.capacity = capacity

    def score(self, model: ClassifierMixin, dataset: Dataset) -> float:
        return self.capacity / (self.capacity + ChunkLoad().score(model, dataset))
