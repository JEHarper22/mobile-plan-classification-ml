import pandas as pd
import matplotlib.pyplot as plt


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df


def print_basic_info(df):
    print("First 5 rows:")
    print(df.head())
    print()

    print("Dataset info:")
    print(df.info())
    print()

    print("Descriptive statistics:")
    print(df.describe())
    print()

    print("Missing values:")
    print(df.isna().sum())
    print()

    print("Target distribution:")
    print(df["is_ultra"].value_counts())
    print()

    print("Target distribution (normalized):")
    print(df["is_ultra"].value_counts(normalize=True))
    print()


def plot_target_distribution(df):
    df["is_ultra"].value_counts().plot(
        kind="bar",
        title="Distribution of Mobile Plans"
    )
    plt.xlabel("Plan Type (0 = Smart, 1 = Ultra)")
    plt.ylabel("Number of Users")
    plt.xticks(rotation=0)
    plt.show()
