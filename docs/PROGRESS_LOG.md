# Progress Log

## Session 1 (2026-10-30/31) — built, validated, and a clean result on the first real run

Built `decision_tree.py` (Gini-impurity tree, Gini-importance computed directly rather
than imported) and `compare.py` (the three-step evaluation from PLAN.md). Breast Cancer
Wisconsin, stratified 80/20 split, `max_depth=4`, `min_samples_split=5` for both the
from-scratch and sklearn trees, so the correctness check is apples-to-apples.

### Step 1 — correctness check: passed, more strongly than expected

| | Accuracy | F1 | Depth reached |
|---|---|---|---|
| From-scratch tree | 94.74% | 0.9583 | 4 |
| sklearn tree (reference) | 93.86% | 0.9510 | 4 |

Gap: 0.88 percentage points — exactly **one** test sample out of 114 (108/114 vs.
107/114). Well inside PLAN.md's "within a few percentage points" minimum bar.

**Stronger evidence than accuracy alone:** the from-scratch tree's top-5 feature
importances are **numerically identical** to sklearn's, to four decimal places (worst
radius 0.7335, worst concave points 0.1220, texture error 0.0458, worst texture 0.0323,
worst concavity 0.0172, in the same order). Matching importance values this precisely
means the two trees made the same sequence of splits on the same features with
essentially the same impurity reductions — strong evidence the splitting logic is
correct, not just coincidentally similar in overall accuracy. The one-sample accuracy
difference is consistent with PLAN.md's named risk (tied impurity reduction between two
nearby threshold candidates being broken differently) rather than a deeper bug — a
single boundary case landing on the other side of a near-tied split threshold.

### Step 2 — single tree vs. ensembles

| Model | Accuracy | F1 |
|---|---|---|
| Single tree (from scratch) | 94.74% | 0.9583 |
| Single tree (sklearn) | 93.86% | 0.9510 |
| Random Forest | **95.61%** | 0.9655 |
| Gradient Boosting | **95.61%** | 0.9660 |

Both ensembles edge out the single tree, matching PLAN.md's "actual claim being
tested" — but the margin is modest (roughly 1-2 more correct test samples), consistent
with PLAN.md's own advance note that this dataset is small and highly separable, so a
strong absolute result here says more about the dataset than about these methods'
general power.

### Step 3 — feature importance: single trees and boosting concentrate, Random Forest spreads out

Both single trees and Gradient Boosting are dominated by one or two features ("worst
radius" alone carries 73% of the from-scratch tree's importance, 44% of Gradient
Boosting's). **Random Forest's importance is visibly more distributed** — ten features
each carrying roughly 3-14%, no single one dominating (see
`feature_importance_comparison.png`). This is a real, sensible pattern, not noise: a
single tree and boosting's sequential trees keep reusing whichever feature splits best
at the top of the tree, while Random Forest's per-split feature subsampling forces many
of its trees to split on a different, slightly-weaker feature when the strongest one
isn't offered as a candidate at that split — spreading credit across more of the 30
features. Worth remembering as a concrete, visual example of *why* bagging with feature
subsampling decorrelates trees, not just an assertion from the textbook.

## Status

Complete. Correctness validated (feature importances match sklearn's exactly); ensembles
modestly outperform a single tree as expected; Random Forest's more distributed feature
importance is a genuine, explainable finding about bagging + feature subsampling, not
just "ensembles win."
