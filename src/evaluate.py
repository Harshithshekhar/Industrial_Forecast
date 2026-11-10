
import torch
import pandas as pd
import numpy as np
import joblib

from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dataset import CompressorDataset
from src.models import CompressorLSTM


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_PATH = OUTPUT_DIR / "compressor_lstm.pth"
SCALER_PATH = OUTPUT_DIR / "scaler.joblib"

TARGET_FEATURES = [
    "pressure",
    "temperature",
    "motor_current",
    "vibration",
    "airflow",
    "rpm"
]

SEQUENCE_LENGTH = 30
BATCH_SIZE = 64

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def evaluate_model():

    print("\n========== MODEL EVALUATION ==========")

    # Load test data
    test_df = pd.read_csv(PROCESSED_DIR / "test_data.csv")
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    test_dataset = CompressorDataset(test_df)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("\nTesting samples:", len(test_dataset))

    # Load scaler
    scaler = joblib.load(SCALER_PATH)

    # Load trained model
    model = CompressorLSTM().to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    all_predictions = []
    all_actuals = []

    # Generate predictions
    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(DEVICE)

            predictions = model(X_batch)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_actuals.extend(
                y_batch.numpy()
            )

    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)

    # Convert standardized values back to original units
    predictions_original = scaler.inverse_transform(predictions)
    actuals_original = scaler.inverse_transform(actuals)

    # Calculate evaluation metrics
    print("\n========== FORECASTING METRICS ==========")

    for i, feature in enumerate(TARGET_FEATURES):

        mae = mean_absolute_error(
            actuals_original[:, i],
            predictions_original[:, i]
        )

        rmse = np.sqrt(
            mean_squared_error(
                actuals_original[:, i],
                predictions_original[:, i]
            )
        )

        r2 = r2_score(
            actuals_original[:, i],
            predictions_original[:, i]
        )

        print(f"\n{feature.upper()}")

        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}")

    # Prepare metadata in the same order as the dataset
    metadata = []

    for compressor_id, group in test_df.groupby("compressor_id"):

        group = group.sort_values("timestamp").reset_index(drop=True)

        metadata.extend(
            group.iloc[SEQUENCE_LENGTH:][
                ["timestamp", "compressor_id", "anomaly", "anomaly_type"]
            ].to_dict("records")
        )

    results = pd.DataFrame(metadata)

    # Store actual and predicted values
    for i, feature in enumerate(TARGET_FEATURES):

        results[f"actual_{feature}"] = actuals_original[:, i]

        results[f"predicted_{feature}"] = predictions_original[:, i]

        results[f"error_{feature}"] = np.abs(
            actuals_original[:, i] - predictions_original[:, i]
        )

    # Mean absolute error across standardized sensor values
    results["anomaly_score"] = np.mean(
        np.abs(actuals - predictions),
        axis=1
    )

    # Save predictions
    prediction_path = OUTPUT_DIR / "forecast_predictions.csv"

    results.to_csv(prediction_path, index=False)

    print("\n========== EVALUATION SUMMARY ==========")

    print("Total predictions:", len(results))

    print("Prediction file saved at:")
    print(prediction_path)

    print("\nModel evaluation completed successfully!")


if __name__ == "__main__":
    evaluate_model()
