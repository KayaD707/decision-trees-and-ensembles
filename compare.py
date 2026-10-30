"""
compare.py -- the two questions this project asks, in order (see PLAN.md):

1. Is the from-scratch DecisionTreeFromScratch actually correct? Validated by training
   it and scikit-learn's own DecisionTreeClassifier on the identical split with matching
   hyperparameters, and comparing test accuracy/F1.
2. Given a correct single tree, how much do Random Forest and Gradient Boosting improve
   on it, and which features drive the predictions?

Run with:
    .venv\\Scripts\\python.exe compare.py
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from decision_tree import DecisionTreeFromScratch

MAX_DEPTH = 4
MIN_SAMPLES_SPLIT = 5
RANDOM_STATE = 42
TOP_K_FEATURES = 5


def main():
    data = load_breast_cancer()
    X, y, feature_names = data.data, data.target, data.feature_names

    # Stratified split: the classes are imbalanced (~63/37 malignant/benign), so keeping
    # that ratio consistent between train and test matters -- see PLAN.md's named risk.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{np.bincount(y)[0]}/{np.bincount(y)[1]} class split (malignant/benign)")
    print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples\n")

    # -------------------------------------------------------------------------------
    # Step 1: correctness check -- from-scratch tree vs. sklearn's reference tree,
    # identical hyperparameters, identical split.
    # -------------------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: VALIDATING THE FROM-SCRATCH TREE AGAINST SKLEARN")
    print("=" * 70)

    scratch_tree = DecisionTreeFromScratch(max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT)
    scratch_tree.fit(X_train, y_train)
    scratch_predictions = scratch_tree.predict(X_test)
    scratch_accuracy = accuracy_score(y_test, scratch_predictions)
    scratch_f1 = f1_score(y_test, scratch_predictions)

    sklearn_tree = DecisionTreeClassifier(
        criterion="gini", max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT, random_state=RANDOM_STATE
    )
    sklearn_tree.fit(X_train, y_train)
    sklearn_predictions = sklearn_tree.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    sklearn_f1 = f1_score(y_test, sklearn_predictions)

    print(f"From-scratch tree -- accuracy: {scratch_accuracy*100:.2f}%  F1: {scratch_f1:.4f}  "
          f"depth reached: {scratch_tree.depth()}")
    print(f"sklearn tree      -- accuracy: {sklearn_accuracy*100:.2f}%  F1: {sklearn_f1:.4f}  "
          f"depth reached: {sklearn_tree.get_depth()}")
    accuracy_gap = abs(scratch_accuracy - sklearn_accuracy)
    print(f"\nAccuracy gap: {accuracy_gap*100:.2f} percentage points "
          f"({'PASS' if accuracy_gap <= 0.05 else 'INVESTIGATE'} -- PLAN.md's minimum bar is "
          f"'within a few percentage points')")

    # -------------------------------------------------------------------------------
    # Step 2: single tree vs. ensembles
    # -------------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: SINGLE TREE VS. RANDOM FOREST VS. GRADIENT BOOSTING")
    print("=" * 70)

    random_forest = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    random_forest.fit(X_train, y_train)
    rf_predictions = random_forest.predict(X_test)

    gradient_boosting = GradientBoostingClassifier(random_state=RANDOM_STATE)
    gradient_boosting.fit(X_train, y_train)
    gb_predictions = gradient_boosting.predict(X_test)

    results = {
        "Single tree (from scratch)": (scratch_accuracy, scratch_f1),
        "Single tree (sklearn, reference)": (sklearn_accuracy, sklearn_f1),
        "Random Forest": (accuracy_score(y_test, rf_predictions), f1_score(y_test, rf_predictions)),
        "Gradient Boosting": (accuracy_score(y_test, gb_predictions), f1_score(y_test, gb_predictions)),
    }

    print(f"\n{'Model':35s} {'Accuracy':>10s} {'F1':>8s}")
    for name, (acc, f1) in results.items():
        print(f"{name:35s} {acc*100:9.2f}% {f1:8.4f}")

    # -------------------------------------------------------------------------------
    # Step 3: feature importance -- from-scratch vs. sklearn's own, across all models
    # -------------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 3: TOP {TOP_K_FEATURES} FEATURES BY IMPORTANCE, PER MODEL")
    print("=" * 70)

    importance_sources = {
        "Single tree (from scratch)": scratch_tree.feature_importances_,
        "Single tree (sklearn)": sklearn_tree.feature_importances_,
        "Random Forest": random_forest.feature_importances_,
        "Gradient Boosting": gradient_boosting.feature_importances_,
    }

    top_features_per_model = {}
    for name, importances in importance_sources.items():
        top_indices = np.argsort(importances)[::-1][:TOP_K_FEATURES]
        top_features_per_model[name] = [(feature_names[i], importances[i]) for i in top_indices]
        print(f"\n{name}:")
        for feature_name, importance in top_features_per_model[name]:
            print(f"  {feature_name:30s} {importance:.4f}")

    # How much do the top-5 feature SETS agree across models, regardless of exact ranking?
    top_sets = {name: {f for f, _ in feats} for name, feats in top_features_per_model.items()}
    scratch_vs_sklearn_overlap = len(top_sets["Single tree (from scratch)"] & top_sets["Single tree (sklearn)"])
    print(f"\nTop-{TOP_K_FEATURES} feature SET overlap, from-scratch tree vs. sklearn tree: "
          f"{scratch_vs_sklearn_overlap}/{TOP_K_FEATURES}")

    # -------------------------------------------------------------------------------
    # Save a results file and a comparison plot
    # -------------------------------------------------------------------------------
    import os
    os.makedirs("docs", exist_ok=True)
    with open("docs/results.txt", "w") as f:
        f.write("Decision Trees & Ensembles -- Breast Cancer Wisconsin results\n\n")
        f.write(f"Correctness check: from-scratch {scratch_accuracy*100:.2f}% vs. "
                f"sklearn {sklearn_accuracy*100:.2f}% (gap: {accuracy_gap*100:.2f}pp)\n\n")
        f.write("Model comparison:\n")
        for name, (acc, f1) in results.items():
            f.write(f"  {name}: accuracy={acc*100:.2f}%, F1={f1:.4f}\n")
        f.write(f"\nTop-{TOP_K_FEATURES} feature SET overlap (from-scratch vs. sklearn tree): "
                f"{scratch_vs_sklearn_overlap}/{TOP_K_FEATURES}\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, importances) in zip(axes.flat, importance_sources.items()):
        top_indices = np.argsort(importances)[::-1][:10]
        ax.barh([feature_names[i] for i in top_indices][::-1], importances[top_indices][::-1])
        ax.set_title(name)
        ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance_comparison.png", dpi=120)
    print("\nSaved docs/results.txt and feature_importance_comparison.png")


if __name__ == "__main__":
    main()
