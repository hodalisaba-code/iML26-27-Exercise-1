# iML 26/27 – Exercise 1: Quantifying Interpretability

Task 1 (1 point): a metric that quantifies the interpretability of tree models, evaluated on decision
trees, random forests and gradient-boosted trees across four datasets, and the resulting threshold of
interpretability. Task 2 is not part of this submission.

## The metric: chunk load

The metric measures how much a person has to hold in mind to use a tree model, in units of
"chunks". Working memory holds roughly 7 +/- 2 chunks (Miller, 1956), which gives a natural
capacity of 7 chunks.

- **Local load.** To reproduce one prediction, a person walks the decision path of every tree.
  All conditions on the same feature along one path merge into one interval, so a path costs one
  chunk per *distinct feature* on it, plus one chunk for the leaf output. A model with several
  trees adds one chunk for the aggregation step (vote or sum). The local load is the mean over the
  samples of the dataset.
- **Global load.** To audit the whole model, a person must find the applicable rule among R rules
  (the leaves of all trees). Singling out one of R rules takes log2(R) bits, added as
  chunk-equivalents.

```
chunk_load = local_load + global_load
score      = 7 / (7 + chunk_load)        in (0, 1], higher = more interpretable
```

A model counts as **interpretable if its chunk load is at most 7, i.e. score >= 0.5**. This is the
threshold of interpretability used below.

The metric goes beyond the number of features: it depends on the paths samples actually take,
on repeated use of a feature along a path, on the number of trees, and on the number of rules. It
applies to every model whose predictions come from scikit-learn decision trees (single tree,
random forest, extra trees, gradient boosting). Implementation: `iml_interpretability/metric.py`.

I did not copy a published interpretability metric. It builds on common ideas (decision path
length, number of rules, Miller's working-memory limit); the way they are combined, and the
capacity of 7 chunks, are my own modelling choices.

## Experiment

- **Datasets** (scikit-learn, offline): iris (4 features, 3 classes), wine (13, 3), breast cancer
  (30, 2), digits (64, 10).
- **Models:** decision trees with maximum depth 1 to 15 and unrestricted; random forests with
  1 to 50 trees of depth 3; gradient boosting with 1 to 50 rounds of depth 2.
- **Protocol:** stratified 75/25 train/test split, metrics computed on the training data, results
  averaged over 5 seeds (0 to 4).

Outputs are in `results/`: `scores.csv` (mean over seeds), `scores_per_seed.csv`, `thresholds.csv`,
`interpretability_by_complexity.png`, `interpretability_vs_accuracy.png`.

## Results

- **Single decision trees:** on breast cancer and digits the threshold is at depth 3 (depth 4
  has chunk load 8.1 and 9.0). On iris and wine, trees stay interpretable at any depth (score
  0.55 and 0.51), because there are few features and the trees stop growing early.
- **Random forests:** a single depth-3 tree is borderline (score 0.50 to 0.59). Two or more trees
  are never interpretable on any dataset (score <= 0.43), since the local load adds up across trees.
- **Gradient boosting:** a single round is interpretable only on breast cancer (score 0.58). On
  multiclass data one round already means one tree per class, so it falls below the threshold
  (digits: 0.17). Everything with more rounds is below the threshold.
- **Accuracy trade-off:** the cost of staying under the threshold differs by dataset. On digits
  the deepest interpretable tree reaches 0.47 test accuracy versus 0.95 for 50 boosting rounds;
  on breast cancer it is 0.93 versus 0.95.

## Limitations

- The capacity of 7 chunks and the global term log2(R) are modelling assumptions, not validated
  against human judgments.
- The metric only supports tree models. Metrics are computed on the training data.
- On low-dimensional data an unrestricted tree is rated interpretable although it has many leaves,
  because the local load is bounded by the number of features.

## Reproduce

```bash
uv sync --extra dev
uv run pytest
uv run python scripts/run_experiment.py
```

With plain pip: `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`,
then run the same pytest and script commands. The experiment needs no download and takes about
30 seconds. `load_titanic` from the template is still in `iml_interpretability/data.py` but is
not used.

```
iml_interpretability/
├── data.py        # Dataset container, load_default_datasets(), load_titanic()
├── models.py      # get_models(): decision trees, random forests, gradient boosting
├── metric.py      # LocalChunkLoad, GlobalChunkLoad, ChunkLoad, InterpretabilityScore
└── evaluate.py    # evaluate(): fits all models on all datasets and computes all metrics
scripts/run_experiment.py   # runs everything, writes results/
tests/                      # pytest tests (offline)
presentation/               # presentation.pdf
```

## AI statement

I used Claude (Claude Sonnet 5.5, through Claude Code). After discussing the task with me, it
wrote the metric, models, dataset loaders, experiment script, tests and this README, starting
from the course template. It ran the tests and the experiment. I reviewed the code and results
and take full responsibility for the content of this repository.
