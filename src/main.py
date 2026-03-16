from src.data_loader import load_data, print_basic_info, plot_target_distribution
from src.split_data import split_data, print_split_sizes, plot_split_sizes
from src.decision_tree_model import train_decision_tree, plot_decision_tree_results
from src.random_forest_model import train_random_forest, plot_random_forest_heatmap
from src.logistic_model import train_logistic_regression, plot_logistic_confusion_matrix
from src.model_selection import (
    compare_models,
    select_best_model,
    plot_model_comparison,
    evaluate_test_model
)
from src.sanity_check import (
    run_sanity_check,
    print_final_conclusion,
    plot_final_results
)


RANDOM_STATE = 12345
DATA_PATH = "PYTHON_SCRIPT_2026/users_behavior.csv"


def main():
    df = load_data(DATA_PATH)

    print_basic_info(df)
    plot_target_distribution(df)

    (
        features,
        target,
        features_train,
        features_valid,
        features_test,
        target_train,
        target_valid,
        target_test
    ) = split_data(df, RANDOM_STATE)

    print_split_sizes(
        features_train,
        target_train,
        features_valid,
        target_valid,
        features_test,
        target_test
    )

    plot_split_sizes(target_train, target_valid, target_test)

    (
        best_tree_model,
        best_tree_accuracy,
        best_tree_depth,
        tree_depths,
        tree_accuracies
    ) = train_decision_tree(
        features_train,
        target_train,
        features_valid,
        target_valid,
        RANDOM_STATE
    )

    plot_decision_tree_results(
        tree_depths,
        tree_accuracies,
        best_tree_depth,
        best_tree_accuracy
    )

    (
        best_forest_model,
        best_forest_accuracy,
        best_forest_estimators,
        best_forest_depth,
        forest_results
    ) = train_random_forest(
        features_train,
        target_train,
        features_valid,
        target_valid,
        RANDOM_STATE
    )

    plot_random_forest_heatmap(forest_results)

    (
        logistic_model,
        logistic_predictions_valid,
        logistic_accuracy
    ) = train_logistic_regression(
        features_train,
        target_train,
        features_valid,
        target_valid,
        RANDOM_STATE
    )

    plot_logistic_confusion_matrix(target_valid, logistic_predictions_valid)

    compare_models(
        best_tree_accuracy,
        best_forest_accuracy,
        logistic_accuracy
    )

    best_model, best_model_name, best_validation_accuracy = select_best_model(
        best_tree_model,
        best_tree_accuracy,
        best_tree_depth,
        best_forest_model,
        best_forest_accuracy,
        best_forest_estimators,
        best_forest_depth,
        logistic_model,
        logistic_accuracy
    )

    plot_model_comparison(
        best_tree_accuracy,
        best_forest_accuracy,
        logistic_accuracy
    )

    test_predictions, test_accuracy = evaluate_test_model(
        best_model,
        features_test,
        target_test
    )

    majority_class, sanity_predictions, sanity_accuracy = run_sanity_check(
        target_train,
        target_test
    )

    print_final_conclusion(
        best_model_name,
        best_validation_accuracy,
        test_accuracy,
        sanity_accuracy
    )

    plot_final_results(sanity_accuracy, test_accuracy)


if __name__ == "__main__":
    main()
