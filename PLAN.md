# Plan: Decision Trees From Scratch, Validated, Then Compared Against Ensembles

Written before any code, same discipline as the rest of this portfolio: commit to the
dataset, the baseline, and the evaluation metric now.

## The question

Two separate questions, answered in order:

1. Is a from-scratch decision tree implementation *correct* — does it match a trusted
   reference implementation (scikit-learn's) closely enough to trust its results?
2. Given a correct single tree, how much does ensembling (Random Forest, Gradient
   Boosting) actually improve on it, and which features drive the predictions either
   way?

## Dataset

**Breast Cancer Wisconsin (Diagnostic)** — built into scikit-learn
(`sklearn.datasets.load_breast_cancer`), 569 samples, 30 real-valued features computed
from digitized images of breast mass cell nuclei, binary target (malignant/benign).
Chosen deliberately for this project's first phase because it's clean (no missing
values, no categorical-encoding decisions to make), well-known, and lets the "from
scratch" implementation be validated without data-wrangling noise getting in the way of
judging whether the TREE LOGIC itself is correct. The features are also individually
meaningful (e.g. "mean radius," "worst texture"), which matters for the
interpretability phase — a feature-importance result can be sanity-checked against
actual domain intuition, not just read as an opaque ranking.

## The method, in plain terms

1. **Build `DecisionTree` from scratch:** recursive, greedy, Gini-impurity-based
   splitting. At each node, try every feature and (for continuous features) a set of
   candidate thresholds, pick the split that most reduces weighted Gini impurity across
   the two children, and recurse. Stopping criteria: `max_depth`, `min_samples_split`,
   or a pure (single-class) node.
2. **Validate against `sklearn.tree.DecisionTreeClassifier`:** train both on an
   identical train/test split with matching `max_depth` and `min_samples_split`, and
   compare test accuracy. A close match (not necessarily bit-for-bit identical, since
   tie-breaking between equally-good splits can differ) is the correctness signal this
   project treats as "the implementation works."
3. **Compare three models** on the same split: the (now-validated) single tree, a
   Random Forest, and Gradient Boosting (scikit-learn implementations for both
   ensembles — reimplementing bagging/boosting from scratch is explicitly out of scope
   for this first pass, see below).
4. **Interpretability:** compute feature importance two ways — (a) from the
   from-scratch tree, by summing each feature's total impurity reduction across every
   split that uses it (the standard "Gini importance" definition, implemented directly
   rather than imported), and (b) scikit-learn's built-in
   `feature_importances_` for all three models. Compare rankings, not just raw values.

## Evaluation protocol

- Single held-out test split (stratified, to keep the malignant/benign ratio consistent
  between train and test — the classes are imbalanced, roughly 63/37).
- Accuracy and F1-score reported for all three models (accuracy alone can be misleading
  on an imbalanced binary target).
- Feature importance rankings compared qualitatively (top-5 features, do they roughly
  agree across models and against the from-scratch tree's own computation).

## What counts as success — stated before training

- **Minimum bar (correctness):** the from-scratch tree's test accuracy is within a few
  percentage points of scikit-learn's reference tree at matching hyperparameters. If
  it's noticeably worse, that's a bug in the splitting logic to find and fix before
  trusting anything built on top of it.
- **The actual claim being tested:** Random Forest and/or Gradient Boosting clearly
  outperform the single tree — a well-established result in the literature for most
  tabular datasets, so (similar to the RL project's framing) a null result here would
  be more likely to indicate a setup issue (too-small ensemble, bad hyperparameters)
  than a genuine finding, and should be investigated rather than reported as
  interesting on its own.
- **Being honest in advance:** this is a small, relatively "easy" dataset by modern
  standards (high class separability is well documented for this benchmark) — a strong
  result here says less about these methods' general power than a messier, real-world
  dataset would. Stated as a real limitation of this project's scope, not hidden.

## Explicitly out of scope for this first pass

- Implementing Random Forest or Gradient Boosting from scratch — the from-scratch
  effort here is specifically about understanding a single tree's splitting logic;
  ensembles are used via scikit-learn so the comparison phase can focus on the
  *results*, not on re-deriving well-known ensembling algorithms.
- Hyperparameter tuning beyond a small, reasonable search — this project is about
  understanding and comparing methods, not squeezing out the last percentage point of
  accuracy.
- A messier, real-world dataset with missing data / categorical features — a natural
  follow-up (see Status), not the starting point.

## Tooling

- **NumPy** for the from-scratch tree.
- **scikit-learn** for the reference tree, Random Forest, Gradient Boosting, and the
  dataset itself. No conflict with the JEPA portfolio's `torch==2.2.2` / `numpy<2`
  pinning (documented in that portfolio's `SETUP_NOTES.md`) — this project needs no
  PyTorch at all, so it gets its own clean venv with current scikit-learn/numpy, no
  version-juggling required.
- **Matplotlib** for the feature-importance comparison plots.

## Known risks, named upfront

- **Split-selection ties:** when two candidate splits produce identical impurity
  reduction, the from-scratch tree and scikit-learn's tree may break the tie
  differently, producing a structurally different (but similarly accurate) tree — a
  reasonable, expected source of small divergence, not necessarily a bug.
- **Imbalanced classes:** ~63/37 split means accuracy alone could look good from a
  model that's biased toward the majority class — F1-score and a stratified split are
  included specifically to catch this.

## Status

Complete. **Correctness validated more strongly than the minimum bar required:** the
from-scratch tree's accuracy (94.74%) is within 0.88pp of sklearn's reference tree
(93.86%, a one-sample difference), and its top-5 feature importances match sklearn's
*exactly* to four decimal places — strong evidence the splitting logic itself is
correct. Ensembles modestly beat the single tree as expected (Random Forest and
Gradient Boosting both 95.61% vs. ~94%), consistent with this being a small, highly
separable dataset as anticipated above. One genuine, explainable finding beyond the
plan: Random Forest's feature importance is visibly more spread across features than
either a single tree's or Gradient Boosting's, which concentrate heavily on one or two
features — a concrete illustration of what per-split feature subsampling does to
decorrelate trees. Full results and the feature-importance plot in
`docs/PROGRESS_LOG.md`.
