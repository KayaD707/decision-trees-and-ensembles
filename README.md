# Decision Trees From Scratch, and Ensembles

## Why this exists

Fourth entry in a data science/AI portfolio, alongside the
[Cookie Cats A/B test project](https://github.com/KayaD707/cookiecats-ab-test-analysis)
(statistics), the
[JEPA self-supervised learning portfolio](https://github.com/KayaD707/jepa-representation-learning)
(deep learning, representation learning), and the
[model-free vs. model-based RL project](https://github.com/KayaD707/rl-model-free-vs-model-based).
Those are all either statistics or deep learning — this project deliberately covers
classical, interpretable tabular ML, the kind of modeling that comes up constantly in
real Data Scientist work and interviews, and is a different skill from anything else in
this portfolio so far.

## The approach

1. **Implement a decision tree classifier from scratch** (Gini-impurity-based greedy
   splitting, recursive partitioning) — not to replace scikit-learn in practice, but so
   the mechanics of *how* a tree actually decides where to split are fully understood,
   not just used as a black box.
2. **Validate it directly against scikit-learn's own `DecisionTreeClassifier`** with
   matching hyperparameters, on the same data — if the from-scratch implementation is
   correct, its accuracy (and ideally its actual splits) should closely match
   scikit-learn's reference implementation. This is the correctness check, not just "it
   ran without an error."
3. **Compare a single tree against Random Forest and Gradient Boosting** (scikit-learn)
   on a real tabular dataset, to see how much ensembling actually buys over one tree.
4. **Interpretability:** compare feature importances computed from the from-scratch
   tree against scikit-learn's built-in feature importances, and use them to explain
   *why* the models decide what they decide — not just report an accuracy number.

See [`PLAN.md`](PLAN.md) for the full pre-registration: dataset choice, evaluation
protocol, and success criteria, written before any code.

## Shared practices (same as the other portfolio repos)

- A `PLAN.md` written before serious code: objective, baseline, evaluation metric, and
  success criteria stated upfront.
- Heavily commented code, written to be read and understood, not just to work.
- A `docs/` folder holding the reasoning log.
- Gradual, session-by-session pace, pushed to GitHub as it goes.

## Status

Complete. The from-scratch tree's feature importances matched scikit-learn's reference
implementation exactly (to four decimal places), and accuracy was within one test
sample (94.74% vs. 93.86%) — strong validation of the splitting logic. Random Forest
and Gradient Boosting both modestly beat the single tree (95.61% vs. ~94%), and Random
Forest's feature importance turned out visibly more spread across features than the
single tree's or Gradient Boosting's — a concrete illustration of what per-split
feature subsampling does. See [`PLAN.md`](PLAN.md) and
[`docs/PROGRESS_LOG.md`](docs/PROGRESS_LOG.md) for full results.
