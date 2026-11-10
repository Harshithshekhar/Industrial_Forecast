
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FEATURES = [
    "pressure",
    "temperature",
    "motor_current",
    "vibration",
    "airflow",
    "rpm",
    "hour_sin",
    "hour_cos"
]

TARGET_FEATURES = [
    "pressure",
    "temperature",
    "motor_current",
    "vibration",
    "airflow",
    "rpm"
]

SEQUENCE_LENGTH = 30


class CompressorDataset(Dataset):

    def __init__(
        self,
        df,
        sequence_length=SEQUENCE_LENGTH,
        normal_only=False,
        target_start=None,
        target_end=None
    ):

        self.sequence_length = sequence_length
        self.X = []
        self.y = []

        for compressor_id, group in df.groupby("compressor_id"):

            group = group.sort_values("timestamp").reset_index(drop=True)

            input_data = group[INPUT_FEATURES].to_numpy(dtype=np.float32)
            target_data = group[TARGET_FEATURES].to_numpy(dtype=np.float32)
            anomaly_labels = group["anomaly"].to_numpy()

            start = sequence_length if target_start is None else max(
                sequence_length, target_start
            )

            end = len(group) if target_end is None else min(
                len(group), target_end
            )

            for i in range(start, end):

                if normal_only:

                    window_labels = anomaly_labels[
                        i - sequence_length:i + 1
                    ]

                    if np.any(window_labels == 1):
                        continue

                x_sequence = input_data[i - sequence_length:i]
                y_target = target_data[i]

                self.X.append(x_sequence)
                self.y.append(y_target)

        self.X = torch.tensor(
            np.array(self.X),
            dtype=torch.float32
        )

        self.y = torch.tensor(
            np.array(self.y),
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


if __name__ == "__main__":

    print("\n========== DATASET SPLIT VERIFICATION ==========")

    train_df = pd.read_csv(PROCESSED_DIR / "train_data.csv")

    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])

    # First 80% of each compressor's training period
    fit_dataset = CompressorDataset(
        train_df,
        normal_only=True,
        target_start=0,
        target_end=2560
    )

    # Last 20% of each compressor's training period
    validation_dataset = CompressorDataset(
        train_df,
        normal_only=True,
        target_start=2560,
        target_end=3200
    )

    print("\nModel training samples:", len(fit_dataset))
    print("Validation samples:", len(validation_dataset))

    X_sample, y_sample = fit_dataset[0]

    print("\nInput sample shape:", X_sample.shape)
    print("Target sample shape:", y_sample.shape)

    print("\nDataset split completed successfully!")
