import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def split_data(df, random_state):
    features = df.drop(columns=["is_ultra"])
    target = df["is_ultra"]

    features_train, features_temp, target_train, target_temp = train_test_split(
        features,
        target,
        test_size=0.4,
        random_state=random_state
    )

    features_valid, features_test, target_valid, target_test = train_test_split(
        features_temp,
        target_temp,
        test_size=0.5,
        random_state=random_state
    )

    return (
        features,
        target,
        features_train,
        features_valid,
        features_test,
        target_train,
        target_valid,
        target_test
    )


def print_split_sizes(features_train, target_train, features_valid, target_valid, features_test, target_test):
    print("Training set size:", features_train.shape, target_train.shape)
    print("Validation set size:", features_valid.shape, target_valid.shape)
    print("Test set size:", features_test.shape, target_test.shape)
    print()


def plot_split_sizes(target_train, target_valid, target_test):
    sets = ["Training", "Validation", "Test"]
    sizes = [len(target_train), len(target_valid), len(target_test)]

    plt.bar(sets, sizes)
    plt.title("Dataset Split for Model Development")
    plt.xlabel("Dataset")
    plt.ylabel("Number of Observations")

    for i, v in enumerate(sizes):
        plt.text(i, v + 20, str(v), ha="center")

    plt.show()
