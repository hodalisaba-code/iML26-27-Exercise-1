"""Run the interpretability evaluation, store the results as CSV files and plot them.

Usage::

    uv run python scripts/run_experiment.py

Outputs in ``results/``: ``scores_per_seed.csv``, ``scores.csv`` (mean over seeds),
``thresholds.csv`` and the plots ``interpretability_by_complexity.png`` and
``interpretability_vs_accuracy.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from iml_interpretability import (  # noqa: E402
    CAPACITY,
    ChunkLoad,
    GlobalChunkLoad,
    InterpretabilityScore,
    LocalChunkLoad,
    evaluate,
    get_models,
    load_default_datasets,
)

OUTPUT_DIR = Path("results")
SEEDS = (0, 1, 2, 3, 4)
THRESHOLD_SCORE = 0.5  # chunk load equal to the capacity of 7 chunks

FAMILY_LABELS = {
    "decision_tree_depth": "Decision tree (max depth)",
    "random_forest_trees": "Random forest, depth 3 (trees)",
    "gradient_boosting_trees": "Gradient boosting, depth 2 (rounds)",
}
FULL_DEPTH_X = 20  # position of the unrestricted tree on the complexity axis


def run() -> pd.DataFrame:
    """Evaluate all models on all datasets for every seed and return the per-seed results."""
    datasets = load_default_datasets()
    metrics = [LocalChunkLoad(), GlobalChunkLoad(), ChunkLoad(), InterpretabilityScore()]
    frames = []
    for seed in SEEDS:
        frame = evaluate(datasets=datasets, models=get_models(seed=seed), metrics=metrics, seed=seed)
        frames.append(frame.assign(seed=seed))
        print(f"seed {seed} done")
    return pd.concat(frames, ignore_index=True)


def add_family_and_complexity(results: pd.DataFrame) -> pd.DataFrame:
    """Split the model name into its family and a numeric complexity value."""
    parts = results["model"].str.rsplit("_", n=1, expand=True)
    results = results.copy()
    results["family"] = parts[0]
    results["complexity"] = parts[1].replace({"full": str(FULL_DEPTH_X)}).astype(int)
    return results


def threshold_table(mean_scores: pd.DataFrame) -> pd.DataFrame:
    """For each dataset and family: the most complex model that is still interpretable."""
    rows = []
    for (dataset, family), group in mean_scores.groupby(["dataset", "family"]):
        group = group.sort_values("complexity")
        interpretable = group[group["interpretability"] >= THRESHOLD_SCORE]
        beyond = group[group["interpretability"] < THRESHOLD_SCORE]
        last = interpretable.iloc[-1] if len(interpretable) else None
        first_not = beyond.iloc[0] if len(beyond) else None
        rows.append(
            {
                "dataset": dataset,
                "family": family,
                "last_interpretable_complexity": None if last is None else int(last["complexity"]),
                "accuracy_at_last_interpretable": None if last is None else round(last["accuracy"], 3),
                "first_not_interpretable_complexity": None if first_not is None else int(first_not["complexity"]),
                "accuracy_at_first_not_interpretable": None if first_not is None else round(first_not["accuracy"], 3),
            }
        )
    columns = [
        "last_interpretable_complexity",
        "first_not_interpretable_complexity",
    ]
    table = pd.DataFrame(rows)
    table[columns] = table[columns].astype("Int64")
    return table


def plot_by_complexity(mean_scores: pd.DataFrame, path: Path) -> None:
    datasets = list(mean_scores["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), sharey=True)
    for ax, dataset in zip(axes, datasets):
        for family, label in FAMILY_LABELS.items():
            group = mean_scores[(mean_scores["dataset"] == dataset) & (mean_scores["family"] == family)]
            group = group.sort_values("complexity")
            ax.plot(group["complexity"], group["interpretability"], marker="o", label=label)
        ax.axhline(THRESHOLD_SCORE, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(dataset)
        ax.set_xlabel("complexity (depth / trees / rounds)")
    axes[0].set_ylabel("interpretability score")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_vs_accuracy(mean_scores: pd.DataFrame, path: Path) -> None:
    datasets = list(mean_scores["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6), sharey=False)
    for ax, dataset in zip(axes, datasets):
        for family, label in FAMILY_LABELS.items():
            group = mean_scores[(mean_scores["dataset"] == dataset) & (mean_scores["family"] == family)]
            ax.scatter(group["interpretability"], group["accuracy"], label=label, s=25)
        ax.axvline(THRESHOLD_SCORE, color="black", linestyle="--", linewidth=1)
        ax.set_title(dataset)
        ax.set_xlabel("interpretability score")
    axes[0].set_ylabel("test accuracy")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed = add_family_and_complexity(run())
    per_seed.to_csv(OUTPUT_DIR / "scores_per_seed.csv", index=False)

    value_columns = ["accuracy", "local_load", "global_load", "chunk_load", "interpretability"]
    mean_scores = (
        per_seed.groupby(["dataset", "family", "model", "complexity"], as_index=False)[value_columns]
        .mean()
        .sort_values(["dataset", "family", "complexity"])
    )
    mean_scores.to_csv(OUTPUT_DIR / "scores.csv", index=False)

    thresholds = threshold_table(mean_scores)
    thresholds.to_csv(OUTPUT_DIR / "thresholds.csv", index=False)

    plot_by_complexity(mean_scores, OUTPUT_DIR / "interpretability_by_complexity.png")
    plot_vs_accuracy(mean_scores, OUTPUT_DIR / "interpretability_vs_accuracy.png")

    print(f"Capacity: {CAPACITY} chunks; threshold: interpretability >= {THRESHOLD_SCORE}")
    print(mean_scores.round(3).to_string(index=False))
    print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
