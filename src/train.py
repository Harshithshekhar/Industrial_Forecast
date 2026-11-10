
import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader
from pathlib import Path

from src.dataset import CompressorDataset
from src.models import CompressorLSTM


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Training configuration
EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001


# Device configuration
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def train_model():

    print("\n========== LSTM MODEL TRAINING ==========")

    print("\nUsing device:", DEVICE)

    # Load training data
    train_df = pd.read_csv(
        PROCESSED_DIR / "train_data.csv"
    )

    train_df["timestamp"] = pd.to_datetime(
        train_df["timestamp"]
    )

    # Use only the first 80% of training data
    # Exclude anomaly-containing sequences
    train_dataset = CompressorDataset(
        train_df,
        normal_only=True,
        target_start=0,
        target_end=2560
    )

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print("\nTraining samples:", len(train_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)

    # Initialize model
    model = CompressorLSTM().to(DEVICE)

    # Loss function
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nStarting model training...\n")

    # Training loop
    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(X_batch)

            # Calculate loss
            loss = criterion(
                predictions,
                y_batch
            )

            # Backpropagation
            loss.backward()

            # Update model weights
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        # Average epoch loss
        epoch_loss = total_loss / len(train_dataset)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"- Training Loss: {epoch_loss:.6f}"
        )

    # Save trained model
    model_path = OUTPUT_DIR / "compressor_lstm.pth"

    torch.save(
        model.state_dict(),
        model_path
    )

    print("\nModel training completed successfully!")
    print("\nTrained model saved at:")
    print(model_path)

    return model


if __name__ == "__main__":

    train_model()
