import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def run_sanity_check(target_train, target_test):
    majority_class = target_train.mode()[0]
    sanity_predictions = [majority_class] * len(target_test)
    sanity_accuracy = accuracy_score(target_test, sanity_predictions)

    print("Sanity check:")
    print("Most frequent class in training set:", majority_class)
    print(f"Baseline accuracy (constant prediction): {sanity_accuracy:.4f}")
    print()

    return majority_class, sanity_predictions, sanity_accuracy


def print_final_conclusion(best_model_name, best_validation_accuracy, test_accuracy, sanity_accuracy):
    print("Final conclusion:")
    print(f"Selected model: {best_model_name}")
    print(f"Validation accuracy: {best_validation_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Sanity check accuracy: {sanity_accuracy:.4f}")

    if test_accuracy >= 0.75:
        print("Project requirement was met: accuracy is at least 0.75.")
    else:
        print("Project requirement was NOT met: accuracy is below 0.75.")


def plot_final_results(sanity_accuracy, test_accuracy):
    labels = ["Sanity Check (Baseline)", "Final Model"]
    values = [sanity_accuracy, test_accuracy]

    plt.bar(labels, values)
    plt.title("Final Model vs Baseline Accuracy")
    plt.ylabel("Accuracy")

    for i, v in enumerate(values):
        plt.text(i, v + 0.002, f"{v:.4f}", ha="center")

    plt.show()
