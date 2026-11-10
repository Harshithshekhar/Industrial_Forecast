
import pandas as pd
from pathlib import Path


# Define project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Define dataset path
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "compressor_data.csv"


def load_compressor_data(file_path=DATA_PATH):

    """
    Loads industrial compressor sensor data.

    Parameters:
        file_path: Location of the CSV dataset.

    Returns:
        pandas DataFrame
    """

    # Check whether the file exists
    if not Path(file_path).exists():

        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Load CSV
    df = pd.read_csv(file_path)

    # Required columns
    required_columns = [
        "timestamp",
        "compressor_id",
        "pressure",
        "temperature",
        "motor_current",
        "vibration",
        "airflow",
        "rpm",
        "anomaly",
        "anomaly_type"
    ]

    # Validate columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise"
    )

    # Sort records
    df = df.sort_values(
        by=["compressor_id", "timestamp"]
    ).reset_index(drop=True)

    print("Dataset loaded successfully!")

    return df


def inspect_data(df):

    """
    Displays basic information about the dataset.
    """

    print("\n========== DATASET OVERVIEW ==========")

    print("\nShape:")
    print(df.shape)

    print("\nFirst Five Records:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nStatistical Summary:")
    print(df.describe())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    print("\nCompressor Counts:")
    print(df["compressor_id"].value_counts())

    print("\nAnomaly Distribution:")
    print(df["anomaly"].value_counts())

    print("\nAnomaly Types:")
    print(df["anomaly_type"].value_counts())


if __name__ == "__main__":

    df = load_compressor_data()

    inspect_data(df)
