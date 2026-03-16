import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_decision_tree(features_train, target_train, features_valid, target_valid, random_state):
    best_tree_model = None
    best_tree_accuracy = 0
    best_tree_depth = 0

    tree_depths = []
    tree_accuracies = []

    for depth in range(1, 11):
        model = DecisionTreeClassifier(
            random_state=random_state,
            max_depth=depth
        )
        model.fit(features_train, target_train)

        predictions_valid = model.predict(features_valid)
        accuracy = accuracy_score(target_valid, predictions_valid)

        tree_depths.append(depth)
        tree_accuracies.append(accuracy)

        print(
            f"Decision Tree | max_depth = {depth}: "
            f"validation accuracy = {accuracy:.4f}"
        )

        if accuracy > best_tree_accuracy:
            best_tree_accuracy = accuracy
            best_tree_depth = depth
            best_tree_model = model

    print()
    print(
        f"Best Decision Tree: max_depth = {best_tree_depth}, "
        f"validation accuracy = {best_tree_accuracy:.4f}"
    )
    print()

    return best_tree_model, best_tree_accuracy, best_tree_depth, tree_depths, tree_accuracies


def plot_decision_tree_results(tree_depths, tree_accuracies, best_tree_depth, best_tree_accuracy):
    plt.figure(figsize=(8, 5))
    plt.plot(tree_depths, tree_accuracies, marker="o")
    plt.title("Decision Tree Depth vs Validation Accuracy")
    plt.xlabel("Max Depth")
    plt.ylabel("Validation Accuracy")
    plt.xticks(tree_depths)
    plt.scatter(best_tree_depth, best_tree_accuracy)

    plt.annotate(
        f"Best depth = {best_tree_depth}\nAccuracy = {best_tree_accuracy:.4f}",
        xy=(best_tree_depth, best_tree_accuracy),
        xytext=(best_tree_depth, best_tree_accuracy + 0.003),
        arrowprops=dict(arrowstyle="->")
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()
