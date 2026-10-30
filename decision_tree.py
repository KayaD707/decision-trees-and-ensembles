"""
decision_tree.py -- a Gini-impurity decision tree classifier, built from scratch.

THE CORE IDEA: at every node, try every feature and a set of candidate thresholds, and
pick whichever (feature, threshold) split reduces the weighted average Gini impurity of
the two resulting children the most. Recurse into each child. Stop when a node is pure
(impurity 0), too small to split further, or a maximum depth is reached.

This mirrors exactly what `sklearn.tree.DecisionTreeClassifier(criterion="gini")` does
internally -- the point of writing it from scratch is to see that mechanism directly,
then validate it against sklearn's own implementation (see compare.py) rather than take
"it works" on faith.
"""

import numpy as np


def gini_impurity(y: np.ndarray) -> float:
    """Gini impurity: 1 - sum(p_c^2) over classes c, where p_c is the fraction of
    samples in class c. 0 means perfectly pure (one class only); the maximum possible
    value approaches 1 as classes become more evenly mixed. This is the quantity every
    candidate split is judged by: a split is good if it makes the two children, on
    average, purer than the parent."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1.0 - np.sum(probabilities ** 2)


class Node:
    """A single tree node. Every node stores its own Gini impurity, sample count, and
    class distribution, whether or not it ends up being a leaf -- useful for inspecting
    the tree afterward, not just for making predictions."""

    def __init__(self, gini: float, n_samples: int, class_counts: np.ndarray, predicted_class: int):
        self.gini = gini
        self.n_samples = n_samples
        self.class_counts = class_counts
        self.predicted_class = predicted_class
        # Set only if this node ends up being split (i.e., is NOT a leaf):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

    def is_leaf(self) -> bool:
        return self.left is None


class DecisionTreeFromScratch:
    def __init__(self, max_depth: int = None, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_features_ = None
        self.n_classes_ = None
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes_ = int(y.max()) + 1
        self.n_features_ = X.shape[1]
        # Accumulates, per feature, the total (sample-weighted) impurity decrease from
        # every split that uses it -- this IS the standard "Gini importance" definition,
        # computed directly here rather than imported (see PLAN.md).
        self._importance_accumulator = np.zeros(self.n_features_)

        self.root = self._grow_tree(X, y, depth=0)

        total_importance = self._importance_accumulator.sum()
        self.feature_importances_ = (
            self._importance_accumulator / total_importance
            if total_importance > 0
            else self._importance_accumulator
        )
        return self

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples = X.shape[0]
        class_counts = np.bincount(y, minlength=self.n_classes_)
        predicted_class = int(np.argmax(class_counts))
        node_gini = gini_impurity(y)

        node = Node(gini=node_gini, n_samples=n_samples, class_counts=class_counts, predicted_class=predicted_class)

        # Stopping criteria: a maximum depth reached, too few samples to split
        # meaningfully, or the node is already pure (impurity 0 -- nothing to gain).
        reached_max_depth = self.max_depth is not None and depth >= self.max_depth
        too_small_to_split = n_samples < self.min_samples_split
        if reached_max_depth or too_small_to_split or node_gini == 0.0:
            return node

        feature_index, threshold, weighted_impurity_decrease = self._best_split(X, y, node_gini, n_samples)
        if feature_index is None:
            # No candidate split improved on the parent's impurity at all -- leave as a leaf
            # rather than force a split that wouldn't help.
            return node

        self._importance_accumulator[feature_index] += weighted_impurity_decrease

        left_mask = X[:, feature_index] <= threshold
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X: np.ndarray, y: np.ndarray, parent_gini: float, n_samples: int):
        """Tries every feature and every midpoint between consecutive sorted unique
        values as a candidate threshold, and returns whichever single (feature,
        threshold) pair reduces weighted Gini impurity the most.

        Returns (feature_index, threshold, weighted_impurity_decrease), or
        (None, None, 0.0) if no split improves on the parent at all.
        """
        best_gain = 0.0
        best_feature, best_threshold = None, None

        for feature_index in range(self.n_features_):
            values = X[:, feature_index]
            unique_sorted = np.unique(values)
            if len(unique_sorted) < 2:
                continue  # a constant feature (within this node) can't split anything

            candidate_thresholds = (unique_sorted[:-1] + unique_sorted[1:]) / 2.0

            for threshold in candidate_thresholds:
                left_mask = values <= threshold
                n_left = int(left_mask.sum())
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue

                weighted_child_gini = (
                    (n_left / n_samples) * gini_impurity(y[left_mask])
                    + (n_right / n_samples) * gini_impurity(y[~left_mask])
                )
                gain = parent_gini - weighted_child_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        if best_feature is None:
            return None, None, 0.0

        # Weighted by this NODE's sample count (not the whole dataset's) -- dividing by
        # the dataset total happens once, at the end, in fit()'s normalization step.
        # This is the standard Gini-importance formula, applied incrementally per split.
        weighted_impurity_decrease = best_gain * n_samples
        return best_feature, best_threshold, weighted_impurity_decrease

    def _predict_one(self, x: np.ndarray) -> int:
        node = self.root
        while not node.is_leaf():
            node = node.left if x[node.feature_index] <= node.threshold else node.right
        return node.predicted_class

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])

    def depth(self) -> int:
        """Actual depth reached (may be less than max_depth if the tree ran out of
        useful splits or pure nodes first) -- useful for sanity-checking against
        sklearn's own `tree_.max_depth` when validating."""
        def _depth(node):
            if node.is_leaf():
                return 0
            return 1 + max(_depth(node.left), _depth(node.right))
        return _depth(self.root)
