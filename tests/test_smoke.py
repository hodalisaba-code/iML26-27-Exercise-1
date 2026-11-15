"""Tests that run offline on synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from iml_interpretability import (
    ChunkLoad,
    Dataset,
    GlobalChunkLoad,
    InterpretabilityMetric,
    InterpretabilityScore,
    LocalChunkLoad,
    evaluate,
    get_models,
)


class _ConstantMetric(InterpretabilityMetric):
    name = "constant"

    def score(self, model, dataset):
        return 1.0


@pytest.fixture
def dataset() -> Dataset:
    X, y = make_classification(n_samples=300, n_features=8, n_informative=5, random_state=0)
    return Dataset(name="synthetic", X=pd.DataFrame(X).add_prefix("x"), y=pd.Series(y))


def test_evaluate_returns_one_row_per_dataset_and_model(dataset):
    models = get_models()
    results = evaluate([dataset], models, [_ConstantMetric()])
    assert len(results) == len(models)
    assert set(results.columns) == {"dataset", "model", "accuracy", "constant"}


def test_stump_has_known_load(dataset):
    stump = DecisionTreeClassifier(max_depth=1, random_state=0).fit(dataset.X, dataset.y)
    assert LocalChunkLoad().score(stump, dataset) == 2  # one feature plus the leaf output
    assert GlobalChunkLoad().score(stump, dataset) == 1  # two rules, one bit
    assert ChunkLoad().score(stump, dataset) == 3
    assert InterpretabilityScore().score(stump, dataset) == pytest.approx(7 / 10)


def test_repeated_feature_on_a_path_counts_once():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    data = Dataset(name="one_feature", X=X, y=y)
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)
    assert tree.get_depth() > 1
    assert LocalChunkLoad().score(tree, data) == 2  # the single feature "a" plus the leaf output


def test_load_increases_with_tree_depth(dataset):
    loads = [
        ChunkLoad().score(DecisionTreeClassifier(max_depth=depth, random_state=0).fit(dataset.X, dataset.y), dataset)
        for depth in (1, 2, 4, 8)
    ]
    assert loads == sorted(loads)
    assert loads[0] < loads[-1]


@pytest.mark.parametrize(
    "ensemble",
    [
        RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0),
        GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=0),
    ],
)
def test_ensemble_is_less_interpretable_than_a_single_tree(dataset, ensemble):
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(dataset.X, dataset.y)
    ensemble = ensemble.fit(dataset.X, dataset.y)
    assert InterpretabilityScore().score(ensemble, dataset) < InterpretabilityScore().score(tree, dataset)


def test_more_trees_means_lower_interpretability(dataset):
    scores = [
        InterpretabilityScore().score(
            RandomForestClassifier(n_estimators=n, max_depth=3, random_state=0).fit(dataset.X, dataset.y), dataset
        )
        for n in (1, 5, 25)
    ]
    assert scores == sorted(scores, reverse=True)


def test_score_is_in_unit_interval(dataset):
    for model in get_models().values():
        score = InterpretabilityScore().score(model.fit(dataset.X, dataset.y), dataset)
        assert 0 < score <= 1


def test_non_tree_models_are_rejected(dataset):
    model = LogisticRegression().fit(dataset.X, dataset.y)
    with pytest.raises(TypeError):
        ChunkLoad().score(model, dataset)
