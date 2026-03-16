import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def compare_models(best_tree_accuracy, best_forest_accuracy, logistic_accuracy):
    print("Model comparison on validation set:")
    print(f"Decision Tree Accuracy:        {best_tree_accuracy:.4f}")
    print(f"Random Forest Accuracy:       {best_forest_accuracy:.4f}")
    print(f"Logistic Regression Accuracy: {logistic_accuracy:.4f}")
    print()


def select_best_model(
    best_tree_model,
    best_tree_accuracy,
    best_tree_depth,
    best_forest_model,
    best_forest_accuracy,
    best_forest_estimators,
    best_forest_depth,
    logistic_model,
    logistic_accuracy
):
    best_model_name = ""
    best_model = None
    best_validation_accuracy = 0

    if best_tree_accuracy > best_validation_accuracy:
        best_validation_accuracy = best_tree_accuracy
        best_model = best_tree_model
        best_model_name = f"Decision Tree (max_depth={best_tree_depth})"

    if best_forest_accuracy > best_validation_accuracy:
        best_validation_accuracy = best_forest_accuracy
        best_model = best_forest_model
        best_model_name = (
            f"Random Forest (n_estimators={best_forest_estimators}, "
            f"max_depth={best_forest_depth})"
        )

    if logistic_accuracy > best_validation_accuracy:
        best_validation_accuracy = logistic_accuracy
        best_model = logistic_model
        best_model_name = "Logistic Regression"

    print("Best model based on validation set:")
    print(best_model_name)
    print(f"Validation accuracy: {best_validation_accuracy:.4f}")
    print()

    return best_model, best_model_name, best_validation_accuracy


def plot_model_comparison(best_tree_accuracy, best_forest_accuracy, logistic_accuracy):
    models = ["Decision Tree", "Random Forest", "Logistic Regression"]
    accuracies = [best_tree_accuracy, best_forest_accuracy, logistic_accuracy]

    plt.bar(models, accuracies)
    plt.title("Validation Accuracy by Model")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.002, f"{v:.4f}", ha="center")

    plt.show()


def evaluate_test_model(best_model, features_test, target_test):
    test_predictions = best_model.predict(features_test)
    test_accuracy = accuracy_score(target_test, test_predictions)
    return test_predictions, test_accuracy
