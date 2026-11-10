
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"

RESULTS_PATH = OUTPUT_DIR / "anomaly_results.csv"

THRESHOLD_PATH = OUTPUT_DIR / "anomaly_threshold.txt"


def visualize_anomalies():

    print("\n========== ANOMALY VISUALIZATION ==========")

    # Load anomaly results
    df = pd.read_csv(RESULTS_PATH)

    # Load calibrated threshold
    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read())

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("\nCalibrated anomaly threshold:", round(threshold, 6))

    # 1. Anomaly score distribution
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=df,
        x="anomaly_score",
        hue="anomaly",
        bins=50,
        kde=True,
        element="step"
    )

    plt.axvline(
        x=threshold,
        linestyle="--",
        label="Calibrated Threshold"
    )

    plt.title("Anomaly Score Distribution")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Number of Observations")
    plt.legend(
        title="Actual Class",
        labels=["Anomaly", "Normal"]
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "anomaly_score_distribution.png"
    )

    plt.show()

    # 2. Actual vs predicted anomaly counts
    comparison = pd.DataFrame({
        "Actual": df["anomaly"].value_counts(),
        "Predicted": df["predicted_anomaly"].value_counts()
    }).reindex([0, 1]).fillna(0)

    comparison.index = ["Normal", "Anomaly"]

    comparison.plot(
        kind="bar",
        figsize=(9, 5)
    )

    plt.title("Actual vs Predicted Anomaly Counts")
    plt.xlabel("Class")
    plt.ylabel("Number of Observations")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "anomaly_count_comparison.png"
    )

    plt.show()

    # 3. Anomaly score over time
    compressor_id = "COMP_001"

    compressor_data = df[
        df["compressor_id"] == compressor_id
    ].sort_values("timestamp")

    plt.figure(figsize=(14, 6))

    plt.plot(
        compressor_data["timestamp"],
        compressor_data["anomaly_score"],
        label="Anomaly Score"
    )

    plt.axhline(
        y=threshold,
        linestyle="--",
        label=f"Calibrated Threshold: {threshold:.4f}"
    )

    actual_anomalies = compressor_data[
        compressor_data["anomaly"] == 1
    ]

    plt.scatter(
        actual_anomalies["timestamp"],
        actual_anomalies["anomaly_score"],
        label="Actual Anomalies",
        marker="x"
    )

    plt.title("Compressor 001 - Anomaly Score Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Anomaly Score")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "compressor_anomaly_timeline.png"
    )

    plt.show()

    print("\nVisualizations saved successfully!")

    print("\nGenerated files:")
    print("1. anomaly_score_distribution.png")
    print("2. anomaly_count_comparison.png")
    print("3. compressor_anomaly_timeline.png")


if __name__ == "__main__":

    visualize_anomalies()
