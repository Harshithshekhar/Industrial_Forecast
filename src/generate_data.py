
import numpy as np
import pandas as pd
from pathlib import Path


def generate_compressor_data():

    # Reproducibility
    np.random.seed(42)

    # Dataset configuration
    num_compressors = 5
    records_per_compressor = 4000

    all_data = []

    # Generate data for each compressor
    for compressor_num in range(1, num_compressors + 1):

        compressor_id = f"COMP_{compressor_num:03d}"

        # Generate timestamps
        timestamps = pd.date_range(
            start="2026-01-01",
            periods=records_per_compressor,
            freq="min"
        )

        # Time-dependent operating load
        time = np.linspace(0, 12 * np.pi, records_per_compressor)

        load = (
            0.5
            + 0.2 * np.sin(time)
            + 0.1 * np.sin(time / 3)
        )

        # Normal sensor readings
        pressure = (
            100
            + 12 * load
            + 3 * np.sin(time / 2)
            + np.random.normal(0, 1.5, records_per_compressor)
        )

        temperature = (
            55
            + 18 * load
            + 2 * np.sin(time / 4)
            + np.random.normal(0, 1.2, records_per_compressor)
        )

        motor_current = (
            25
            + 20 * load
            + np.random.normal(0, 1.5, records_per_compressor)
        )

        vibration = (
            1.5
            + 1.2 * load
            + np.random.normal(0, 0.15, records_per_compressor)
        )

        airflow = (
            6
            + 4 * load
            + np.random.normal(0, 0.3, records_per_compressor)
        )

        rpm = (
            1500
            + 400 * load
            + np.random.normal(0, 15, records_per_compressor)
        )

        # Ground-truth anomaly labels
        anomaly = np.zeros(records_per_compressor, dtype=int)

        anomaly_type = np.full(
            records_per_compressor,
            "Normal",
            dtype=object
        )

        # Four types of abnormal events
        event_types = [
            "Pressure_Fluctuation",
            "Temperature_Increase",
            "High_Motor_Current",
            "Excessive_Vibration"
        ]

        # Two events in training period
        # Two events in testing period
        event_start_positions = [350, 1200, 3300, 3650]

        for event_index, event in enumerate(event_types):

            start = (
                event_start_positions[event_index]
                + (compressor_num * 17)
            )

            end = start + 45

            # Mark abnormal records
            anomaly[start:end] = 1

            anomaly_type[start:end] = event

            # Simulate abnormal sensor behaviour
            if event == "Pressure_Fluctuation":

                pressure[start:end] -= 22

                airflow[start:end] -= 2

            elif event == "Temperature_Increase":

                temperature[start:end] += 20

            elif event == "High_Motor_Current":

                motor_current[start:end] += 18

            elif event == "Excessive_Vibration":

                vibration[start:end] += 5

        # Create compressor dataframe
        compressor_data = pd.DataFrame({

            "timestamp": timestamps,

            "compressor_id": compressor_id,

            "pressure": pressure,

            "temperature": temperature,

            "motor_current": motor_current,

            "vibration": vibration,

            "airflow": airflow,

            "rpm": rpm,

            "anomaly": anomaly,

            "anomaly_type": anomaly_type

        })

        all_data.append(compressor_data)

    # Combine all compressor records
    final_data = pd.concat(
        all_data,
        ignore_index=True
    )

    # Sort records
    final_data = final_data.sort_values(
        by=["compressor_id", "timestamp"]
    ).reset_index(drop=True)

    # Save dataset
    project_root = Path(__file__).resolve().parents[1]

    output_path = (
        project_root
        / "data"
        / "raw"
        / "compressor_data.csv"
    )

    final_data.to_csv(output_path, index=False)

    # Display dataset information
    print("\nDataset generated successfully!")

    print("\nDataset Shape:", final_data.shape)

    print("\nFirst Five Records:")

    print(final_data.head())

    print("\nAnomaly Distribution:")

    print(final_data["anomaly"].value_counts())

    print("\nAnomaly Types:")

    print(final_data["anomaly_type"].value_counts())

    print("\nCompressor Distribution:")

    print(final_data["compressor_id"].value_counts())

    print("\nMissing Values:")

    print(final_data.isnull().sum())

    print("\nDataset saved at:")

    print(output_path)


if __name__ == "__main__":

    generate_compressor_data()
