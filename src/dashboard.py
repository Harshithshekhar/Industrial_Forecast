
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


# ==========================================
# PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

ANOMALY_PATH = OUTPUT_DIR / "anomaly_results.csv"
RAW_DATA_PATH = DATA_DIR / "compressor_data.csv"
THRESHOLD_PATH = OUTPUT_DIR / "anomaly_threshold.txt"
FORECAST_PATH = OUTPUT_DIR / "forecast_predictions.csv"


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Industrial Compressor Monitoring",
    page_icon="⚙️",
    layout="wide"
)


# ==========================================
# LOAD DATA
# ==========================================

@st.cache_data
def load_data():

    anomaly_df = pd.read_csv(ANOMALY_PATH)

    raw_df = pd.read_csv(RAW_DATA_PATH)

    forecast_df = pd.read_csv(FORECAST_PATH)

    anomaly_df["timestamp"] = pd.to_datetime(
        anomaly_df["timestamp"]
    )

    raw_df["timestamp"] = pd.to_datetime(
        raw_df["timestamp"]
    )

    forecast_df["timestamp"] = pd.to_datetime(
        forecast_df["timestamp"]
    )

    with open(THRESHOLD_PATH, "r") as file:
        threshold = float(file.read())

    return anomaly_df, raw_df, forecast_df, threshold


# ==========================================
# CHECK REQUIRED FILES
# ==========================================

st.title("⚙️ Industrial Compressor Monitoring System")

st.caption(
    "LSTM-Based Sensor Forecasting and Anomaly Detection"
)

required_files = [
    ANOMALY_PATH,
    RAW_DATA_PATH,
    THRESHOLD_PATH,
    FORECAST_PATH
]

missing_files = [
    file for file in required_files
    if not file.exists()
]

if missing_files:

    st.error("Required project files are missing:")

    for file in missing_files:
        st.write(str(file))

    st.stop()


# ==========================================
# LOAD PROJECT DATA
# ==========================================

try:

    anomaly_df, raw_df, forecast_df, threshold = load_data()

except Exception as error:

    st.error(f"Unable to load project data: {error}")

    st.stop()


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Select Dashboard Page",
    [
        "Plant Overview",
        "Sensor Monitoring",
        "Anomaly Alerts",
        "Model Performance",
        "Forecasting"
    ]
)


# ==========================================
# COMPRESSOR SELECTION
# ==========================================

compressors = sorted(
    anomaly_df["compressor_id"].unique()
)

selected_compressor = st.sidebar.selectbox(
    "Select Compressor",
    compressors
)


# Filter selected compressor

compressor_data = anomaly_df[
    anomaly_df["compressor_id"] == selected_compressor
].sort_values("timestamp")


# ==========================================
# COMMON METRICS
# ==========================================

total_compressors = anomaly_df["compressor_id"].nunique()

total_observations = len(anomaly_df)

actual_anomalies = int(
    anomaly_df["anomaly"].sum()
)

predicted_anomalies = int(
    anomaly_df["predicted_anomaly"].sum()
)

normal_observations = (
    total_observations - actual_anomalies
)

actual = anomaly_df["anomaly"]

predicted = anomaly_df["predicted_anomaly"]


precision = precision_score(
    actual,
    predicted,
    zero_division=0
)

recall = recall_score(
    actual,
    predicted,
    zero_division=0
)

f1 = f1_score(
    actual,
    predicted,
    zero_division=0
)

accuracy = accuracy_score(
    actual,
    predicted
)


# ==========================================
# PAGE 1: PLANT OVERVIEW
# ==========================================

if page == "Plant Overview":

    st.header("Plant Overview")

    st.write(
        "Monitor the overall performance of your industrial compressors."
    )

    # KPI metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Compressors",
        total_compressors
    )

    col2.metric(
        "Total Observations",
        f"{total_observations:,}"
    )

    col3.metric(
        "Actual Anomalies",
        actual_anomalies
    )

    col4.metric(
        "Predicted Anomalies",
        predicted_anomalies
    )

    st.divider()

    # Anomaly distribution

    st.subheader("Anomaly Distribution")

    col1, col2 = st.columns(2)

    with col1:

        fig, ax = plt.subplots(figsize=(7, 5))

        sns.countplot(
            data=anomaly_df,
            x="anomaly",
            ax=ax
        )

        ax.set_xticks([0, 1])

        ax.set_xticklabels(
            ["Normal", "Anomaly"]
        )

        ax.set_title("Actual Anomaly Distribution")

        st.pyplot(fig)

        plt.close(fig)

    with col2:

        comparison = pd.DataFrame({

            "Actual": anomaly_df["anomaly"].value_counts(),

            "Predicted": anomaly_df[
                "predicted_anomaly"
            ].value_counts()

        }).reindex([0, 1]).fillna(0)

        comparison.index = [
            "Normal",
            "Anomaly"
        ]

        fig, ax = plt.subplots(figsize=(7, 5))

        comparison.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Actual vs Predicted Anomalies")

        ax.set_xlabel("Category")

        ax.set_ylabel("Observations")

        ax.tick_params(
            axis="x",
            rotation=0
        )

        st.pyplot(fig)

        plt.close(fig)

    st.divider()

    # Compressor-wise summary

    st.subheader("Compressor-wise Anomaly Summary")

    summary = anomaly_df.groupby(
        "compressor_id"
    ).agg(
        Actual_Anomalies=("anomaly", "sum"),
        Predicted_Anomalies=("predicted_anomaly", "sum")
    ).reset_index()

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


# ==========================================
# PAGE 2: SENSOR MONITORING
# ==========================================

elif page == "Sensor Monitoring":

    st.header("Sensor Monitoring")

    st.write(
        f"Monitoring sensor readings for {selected_compressor}."
    )

    sensor_columns = [
        "pressure",
        "temperature",
        "motor_current",
        "vibration",
        "airflow",
        "rpm"
    ]

    selected_sensor = st.selectbox(
        "Select Sensor",
        sensor_columns
    )

    # Filter test results

    selected_results = anomaly_df[
        anomaly_df["compressor_id"] == selected_compressor
    ].copy()

    # Select raw sensor columns

    selected_raw = raw_df[
        raw_df["compressor_id"] == selected_compressor
    ][
        [
            "timestamp",
            "compressor_id",
            *sensor_columns
        ]
    ].copy()

    # Merge sensor readings with predictions

    sensor_data = selected_raw.merge(
        selected_results[
            [
                "timestamp",
                "compressor_id",
                "anomaly",
                "predicted_anomaly",
                "anomaly_score"
            ]
        ],
        on=["timestamp", "compressor_id"],
        how="inner"
    )

    sensor_data = sensor_data.sort_values(
        "timestamp"
    )

    # Sensor trend

    st.subheader(
        f"{selected_sensor.replace('_', ' ').title()} Trend"
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        sensor_data["timestamp"],
        sensor_data[selected_sensor],
        label=selected_sensor,
        linewidth=1
    )

    # Actual anomalies

    actual_points = sensor_data[
        sensor_data["anomaly"] == 1
    ]

    ax.scatter(
        actual_points["timestamp"],
        actual_points[selected_sensor],
        marker="x",
        label="Actual Anomaly"
    )

    # Predicted anomalies

    predicted_points = sensor_data[
        sensor_data["predicted_anomaly"] == 1
    ]

    ax.scatter(
        predicted_points["timestamp"],
        predicted_points[selected_sensor],
        marker="o",
        facecolors="none",
        label="Predicted Anomaly"
    )

    ax.set_xlabel("Timestamp")

    ax.set_ylabel(selected_sensor)

    ax.legend()

    ax.grid(alpha=0.3)

    plt.xticks(rotation=30)

    plt.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # Sensor statistics

    st.subheader("Sensor Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average",
        f"{sensor_data[selected_sensor].mean():.2f}"
    )

    col2.metric(
        "Minimum",
        f"{sensor_data[selected_sensor].min():.2f}"
    )

    col3.metric(
        "Maximum",
        f"{sensor_data[selected_sensor].max():.2f}"
    )


# ==========================================
# PAGE 3: ANOMALY ALERTS
# ==========================================

elif page == "Anomaly Alerts":

    st.header("Anomaly Alerts")

    st.write(
        "Review observations flagged by the LSTM anomaly detection model."
    )

    # Calibrated threshold

    st.metric(
        "Calibrated Anomaly Threshold",
        f"{threshold:.6f}"
    )

    # Filter predicted anomalies

    alerts = anomaly_df[
        anomaly_df["predicted_anomaly"] == 1
    ].copy()

    alerts = alerts.sort_values(
        "anomaly_score",
        ascending=False
    )

    # Alert metrics

    col1, col2 = st.columns(2)

    col1.metric(
        "Total Alerts",
        len(alerts)
    )

    col2.metric(
        "Actual Anomalies Detected",
        int(
            (alerts["anomaly"] == 1).sum()
        )
    )

    # Alert history

    st.subheader("Alert History")

    st.dataframe(
        alerts[
            [
                "timestamp",
                "compressor_id",
                "anomaly_type",
                "anomaly_score",
                "predicted_anomaly"
            ]
        ],
        use_container_width=True,
        hide_index=True
    )

    # Download alerts

    st.download_button(
        label="Download Alert History",
        data=alerts.to_csv(index=False),
        file_name="compressor_alert_history.csv",
        mime="text/csv"
    )


# ==========================================
# PAGE 4: MODEL PERFORMANCE
# ==========================================

elif page == "Model Performance":

    st.header("Model Performance")

    st.write(
        "Evaluation metrics calculated from the test dataset."
    )

    # Performance metrics

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    col2.metric(
        "Precision",
        f"{precision:.2%}"
    )

    col3.metric(
        "Recall",
        f"{recall:.2%}"
    )

    col4.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    st.divider()

    # Confusion matrix

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")

    ax.set_ylabel("Actual Label")

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # Anomaly score distribution

    st.subheader("Anomaly Score Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(
        data=anomaly_df,
        x="anomaly_score",
        hue="anomaly",
        bins=50,
        kde=True,
        element="step",
        ax=ax
    )

    ax.axvline(
        threshold,
        linestyle="--",
        label="Calibrated Threshold"
    )

    ax.set_xlabel("Anomaly Score")

    ax.set_ylabel("Observations")

    st.pyplot(fig)

    plt.close(fig)


# ==========================================
# PAGE 5: FORECASTING
# ==========================================

elif page == "Forecasting":

    st.header("Compressor Sensor Forecasting")

    st.write(
        "Compare actual sensor readings with predictions "
        "generated by the LSTM forecasting model."
    )

    # Sensor selection

    sensor_mapping = {

        "Pressure": (
            "actual_pressure",
            "predicted_pressure"
        ),

        "Temperature": (
            "actual_temperature",
            "predicted_temperature"
        ),

        "Motor Current": (
            "actual_motor_current",
            "predicted_motor_current"
        ),

        "Vibration": (
            "actual_vibration",
            "predicted_vibration"
        ),

        "Airflow": (
            "actual_airflow",
            "predicted_airflow"
        ),

        "RPM": (
            "actual_rpm",
            "predicted_rpm"
        )

    }

    selected_forecast_sensor = st.selectbox(
        "Select Sensor for Forecasting",
        list(sensor_mapping.keys())
    )

    actual_column, predicted_column = sensor_mapping[
        selected_forecast_sensor
    ]

    # Filter forecasting results

    forecast_data = forecast_df[
        forecast_df["compressor_id"] == selected_compressor
    ].copy()

    forecast_data = forecast_data.sort_values(
        "timestamp"
    )

    if forecast_data.empty:

        st.warning(
            "No forecasting results available for this compressor."
        )

    else:

        # Calculate forecasting errors

        forecast_data["absolute_error"] = (
            forecast_data[actual_column] -
            forecast_data[predicted_column]
        ).abs()

        mae = forecast_data["absolute_error"].mean()

        rmse = (
            forecast_data["absolute_error"] ** 2
        ).mean() ** 0.5

        # Forecasting KPIs

        st.subheader("Forecasting Performance")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Mean Absolute Error (MAE)",
            f"{mae:.4f}"
        )

        col2.metric(
            "Root Mean Squared Error (RMSE)",
            f"{rmse:.4f}"
        )

        col3.metric(
            "Forecast Observations",
            f"{len(forecast_data):,}"
        )

        st.divider()

        # Actual vs predicted chart

        st.subheader(
            f"Actual vs Predicted {selected_forecast_sensor}"
        )

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(
            forecast_data["timestamp"],
            forecast_data[actual_column],
            label="Actual",
            linewidth=1.2
        )

        ax.plot(
            forecast_data["timestamp"],
            forecast_data[predicted_column],
            label="Predicted",
            linewidth=1.2
        )

        ax.set_xlabel("Timestamp")

        ax.set_ylabel(selected_forecast_sensor)

        ax.set_title(
            f"{selected_forecast_sensor} Forecast Comparison - "
            f"{selected_compressor}"
        )

        ax.legend()

        ax.grid(alpha=0.3)

        plt.xticks(rotation=30)

        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.divider()

        # Prediction error distribution

        st.subheader("Prediction Error Distribution")

        fig, ax = plt.subplots(figsize=(10, 4))

        sns.histplot(
            forecast_data["absolute_error"],
            bins=40,
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Absolute Prediction Error")

        ax.set_ylabel("Frequency")

        st.pyplot(fig)

        plt.close(fig)

        st.divider()

        # Forecast data table

        st.subheader("Forecast Data")

        display_data = forecast_data[
            [
                "timestamp",
                "compressor_id",
                actual_column,
                predicted_column,
                "absolute_error"
            ]
        ].copy()

        display_data.columns = [
            "Timestamp",
            "Compressor",
            "Actual",
            "Predicted",
            "Absolute Error"
        ]

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        # Download forecasting results

        st.download_button(
            label="Download Forecast Results",
            data=display_data.to_csv(index=False),
            file_name="compressor_forecast_results.csv",
            mime="text/csv"
        )


# ==========================================
# FOOTER
# ==========================================

st.sidebar.divider()

st.sidebar.caption(
    "Industrial Compressor Monitoring System"
)

st.sidebar.caption(
    "Powered by Python, PyTorch and Streamlit"
)
