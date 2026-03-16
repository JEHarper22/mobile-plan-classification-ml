import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def train_random_forest(features_train, target_train, features_valid, target_valid, random_state):
    best_forest_model = None
    best_forest_accuracy = 0
    best_forest_estimators = 0
    best_forest_depth = 0

    forest_results = []

    for est in range(10, 51, 10):
        for depth in range(1, 11):
            model = RandomForestClassifier(
                random_state=random_state,
                n_estimators=est,
                max_depth=depth
            )
            model.fit(features_train, target_train)

            predictions_valid = model.predict(features_valid)
            accuracy = accuracy_score(target_valid, predictions_valid)

            forest_results.append((est, depth, accuracy))

            print(
                f"Random Forest | n_estimators = {est}, "
                f"max_depth = {depth}: validation accuracy = {accuracy:.4f}"
            )

            if accuracy > best_forest_accuracy:
                best_forest_accuracy = accuracy
                best_forest_estimators = est
                best_forest_depth = depth
                best_forest_model = model

    print()
    print(
        f"Best Random Forest: n_estimators = {best_forest_estimators}, "
        f"max_depth = {best_forest_depth}, "
        f"validation accuracy = {best_forest_accuracy:.4f}"
    )
    print()

    return (
        best_forest_model,
        best_forest_accuracy,
        best_forest_estimators,
        best_forest_depth,
        forest_results
    )


def plot_random_forest_heatmap(forest_results):
    df_results = pd.DataFrame(
        forest_results,
        columns=["n_estimators", "max_depth", "accuracy"]
    )

    pivot_table = df_results.pivot(
        index="max_depth",
        columns="n_estimators",
        values="accuracy"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f")
    plt.title("Random Forest Validation Accuracy")
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")
    plt.show()
