
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from src.dataset import CompressorDataset
from src.models import CompressorLSTM


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = OUTPUT_DIR / "compressor_lstm.pth"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Configuration
BATCH_SIZE = 64

SEQUENCE_LENGTH = 30

THRESHOLD_PERCENTILE = 99


# Device configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# Calculate prediction errors
def calculate_prediction_scores(model, dataset):

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_scores = []

    model.eval()

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            y_batch = y_batch.to(DEVICE)

            predictions = model(X_batch)

            errors = torch.abs(
                predictions - y_batch
            )

            scores = torch.mean(
                errors,
                dim=1
            )

            all_scores.extend(
                scores.cpu().numpy()
            )

    return np.array(all_scores)


# Create metadata for matching prediction scores
def create_metadata(
    df,
    normal_only=False,
    target_start=None,
    target_end=None
):

    metadata = []

    for compressor_id, group in df.groupby("compressor_id"):

        group = group.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        anomaly_labels = group["anomaly"].to_numpy()

        start = SEQUENCE_LENGTH if target_start is None else max(
            SEQUENCE_LENGTH,
            target_start
        )

        end = len(group) if target_end is None else min(
            len(group),
            target_end
        )

        for i in range(start, end):

            if normal_only:

                window_labels = anomaly_labels[
                    i - SEQUENCE_LENGTH : i + 1
                ]

                if np.any(window_labels == 1):
                    continue

            metadata.append(
                group.iloc[i][
                    [
                        "timestamp",
                        "compressor_id",
                        "anomaly",
                        "anomaly_type"
                    ]
                ].to_dict()
            )

    return pd.DataFrame(metadata)


# Main anomaly detection function
def run_anomaly_analysis():

    print("\n========== ANOMALY DETECTION ==========")

    # Load training and testing data
    train_df = pd.read_csv(
        PROCESSED_DIR / "train_data.csv"
    )

    test_df = pd.read_csv(
        PROCESSED_DIR / "test_data.csv"
    )

    train_df["timestamp"] = pd.to_datetime(
        train_df["timestamp"]
    )

    test_df["timestamp"] = pd.to_datetime(
        test_df["timestamp"]
    )

    # Validation dataset: last 20% of training period
    validation_dataset = CompressorDataset(
        train_df,
        normal_only=True,
        target_start=2560,
        target_end=3200
    )

    # Test dataset: separate testing period
    test_dataset = CompressorDataset(
        test_df,
        normal_only=False
    )

    print("\nValidation samples:", len(validation_dataset))

    print("Testing samples:", len(test_dataset))

    # Load trained model
    model = CompressorLSTM().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    print("\nModel loaded successfully!")

    # Calculate validation anomaly scores
    validation_scores = calculate_prediction_scores(
        model,
        validation_dataset
    )

    # Calculate test anomaly scores
    test_scores = calculate_prediction_scores(
        model,
        test_dataset
    )

    # Create validation metadata
    validation_metadata = create_metadata(
        train_df,
        normal_only=True,
        target_start=2560,
        target_end=3200
    )

    # Create test metadata
    test_metadata = create_metadata(
        test_df,
        normal_only=False
    )

    # Verify score and metadata alignment
    assert len(validation_scores) == len(validation_metadata), (
        "Validation score and metadata lengths do not match!"
    )

    assert len(test_scores) == len(test_metadata), (
        "Test score and metadata lengths do not match!"
    )

    # Calculate threshold using validation data only
    threshold = np.percentile(
        validation_scores,
        THRESHOLD_PERCENTILE
    )

    print("\n========== ANOMALY THRESHOLD ==========")

    print(
        "Threshold percentile:",
        THRESHOLD_PERCENTILE
    )

    print(
        "Validation samples:",
        len(validation_scores)
    )

    print(
        f"Anomaly threshold: {threshold:.6f}"
    )

    # Save threshold for future use
    threshold_path = OUTPUT_DIR / "anomaly_threshold.txt"

    with open(threshold_path, "w") as file:
        file.write(str(threshold))

    # Prepare test results
    results = test_metadata.copy()

    results["anomaly_score"] = test_scores

    results["predicted_anomaly"] = (
        results["anomaly_score"] > threshold
    ).astype(int)

    # Actual labels
    actual = results["anomaly"].to_numpy()

    # Predicted labels
    predicted = results["predicted_anomaly"].to_numpy()

    # Confusion matrix
    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    print("\n========== CONFUSION MATRIX ==========")

    print(cm)

    print("\nTrue Negatives:", tn)

    print("False Positives:", fp)

    print("False Negatives:", fn)

    print("True Positives:", tp)

    # Classification metrics
    print("\n========== CLASSIFICATION METRICS ==========")

    print(
        f"Precision: {precision_score(actual, predicted, zero_division=0):.4f}"
    )

    print(
        f"Recall: {recall_score(actual, predicted, zero_division=0):.4f}"
    )

    print(
        f"F1 Score: {f1_score(actual, predicted, zero_division=0):.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0
        )
    )

    # Compressor-wise anomaly analysis
    print("\n========== COMPRESSOR-WISE ANOMALIES ==========")

    print(
        results.groupby("compressor_id")[
            ["anomaly", "predicted_anomaly"]
        ].sum()
    )

    # Actual anomaly types
    print("\n========== ACTUAL ANOMALY TYPES ==========")

    print(
        results[results["anomaly"] == 1][
            "anomaly_type"
        ].value_counts()
    )

    # Save anomaly results
    results_path = OUTPUT_DIR / "anomaly_results.csv"

    results.to_csv(
        results_path,
        index=False
    )

    print("\nAnomaly results saved at:")

    print(results_path)

    print("\nThreshold saved at:")

    print(threshold_path)

    print("\nAnomaly detection completed successfully!")


if __name__ == "__main__":

    run_anomaly_analysis()
