
import pandas as pd
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_compressor_data


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Create directories if they do not exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Sensor columns used for forecasting
SENSOR_FEATURES = [
    "pressure",
    "temperature",
    "motor_current",
    "vibration",
    "airflow",
    "rpm"
]


def split_data_by_time(df, train_ratio=0.80):

    """
    Splits each compressor's records chronologically.

    First 80%  -> Training data
    Last 20%   -> Testing data
    """

    train_parts = []
    test_parts = []

    for compressor_id, group in df.groupby("compressor_id"):

        # Sort records chronologically
        group = group.sort_values("timestamp").reset_index(drop=True)

        # Calculate split position
        split_index = int(len(group) * train_ratio)

        # Separate training and testing records
        train_group = group.iloc[:split_index].copy()

        test_group = group.iloc[split_index:].copy()

        train_parts.append(train_group)

        test_parts.append(test_group)

    # Combine all compressors
    train_df = pd.concat(train_parts, ignore_index=True)

    test_df = pd.concat(test_parts, ignore_index=True)

    return train_df, test_df


def scale_features(train_df, test_df):

    """
    Fits StandardScaler on training data only
    and transforms both training and testing data.
    """

    scaler = StandardScaler()

    # Learn scaling parameters from training data
    train_df[SENSOR_FEATURES] = scaler.fit_transform(
        train_df[SENSOR_FEATURES]
    )

    # Apply the same scaling to testing data
    test_df[SENSOR_FEATURES] = scaler.transform(
        test_df[SENSOR_FEATURES]
    )

    return train_df, test_df, scaler


def preprocess_data(df):

    """
    Complete preprocessing pipeline.
    """

    print("\n========== DATA PREPROCESSING ==========")

    # Check for missing values
    if df[SENSOR_FEATURES].isnull().sum().sum() > 0:

        raise ValueError("Missing values found in sensor features.")

    # Check for duplicate records
    if df.duplicated().sum() > 0:

        raise ValueError("Duplicate records found in dataset.")

    print("\nData quality checks passed!")

    # Step 1: Chronological train-test split
    train_df, test_df = split_data_by_time(df)

    print("\nTraining data shape:", train_df.shape)

    print("Testing data shape:", test_df.shape)

    # Step 2: Scale sensor features
    train_df, test_df, scaler = scale_features(
        train_df,
        test_df
    )

    print("\nFeature scaling completed!")

    # Step 3: Save processed datasets
    train_path = PROCESSED_DIR / "train_data.csv"

    test_path = PROCESSED_DIR / "test_data.csv"

    train_df.to_csv(train_path, index=False)

    test_df.to_csv(test_path, index=False)

    # Step 4: Save fitted scaler
    scaler_path = OUTPUT_DIR / "scaler.joblib"

    joblib.dump(scaler, scaler_path)

    print("\nProcessed datasets saved successfully!")

    print("Training data:", train_path)

    print("Testing data:", test_path)

    print("Scaler saved:", scaler_path)

    # Step 5: Display anomaly distribution
    print("\nTraining anomaly distribution:")

    print(train_df["anomaly"].value_counts())

    print("\nTesting anomaly distribution:")

    print(test_df["anomaly"].value_counts())

    return train_df, test_df, scaler


if __name__ == "__main__":

    # Load original dataset
    df = load_compressor_data()

    # Run preprocessing
    train_df, test_df, scaler = preprocess_data(df)

    print("\nPreprocessing completed successfully!")
