
import pandas as pd
import numpy as np

from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def create_time_features(df):

    """
    Creates cyclical time-based features.

    Parameters:
        df: Preprocessed compressor dataset.

    Returns:
        DataFrame with additional time features.
    """

    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Extract decimal hour
    hour = (
        df["timestamp"].dt.hour
        + df["timestamp"].dt.minute / 60
    )

    # Create cyclical time features
    df["hour_sin"] = np.sin(
        2 * np.pi * hour / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * hour / 24
    )

    return df


def engineer_features(train_df, test_df):

    """
    Applies feature engineering to training
    and testing datasets.
    """

    print("\n========== FEATURE ENGINEERING ==========")

    # Create time features for training data
    train_df = create_time_features(train_df)

    # Create time features for testing data
    test_df = create_time_features(test_df)

    print("\nTime-based features created!")

    # Save engineered datasets
    train_path = PROCESSED_DIR / "train_data.csv"

    test_path = PROCESSED_DIR / "test_data.csv"

    train_df.to_csv(
        train_path,
        index=False
    )

    test_df.to_csv(
        test_path,
        index=False
    )

    # Display dataset information
    print("\nEngineered training shape:", train_df.shape)

    print("Engineered testing shape:", test_df.shape)

    print("\nNew features:")

    print(["hour_sin", "hour_cos"])

    print("\nTraining columns:")

    print(train_df.columns.tolist())

    print("\nTesting columns:")

    print(test_df.columns.tolist())

    print("\nFeature engineering completed successfully!")

    return train_df, test_df


if __name__ == "__main__":

    # Load preprocessed training data
    train_df = pd.read_csv(
        PROCESSED_DIR / "train_data.csv"
    )

    # Load preprocessed testing data
    test_df = pd.read_csv(
        PROCESSED_DIR / "test_data.csv"
    )

    # Apply feature engineering
    train_df, test_df = engineer_features(
        train_df,
        test_df
    )
