# Industrial Compressor Forecasting and Anomaly Detection

An end-to-end machine learning project that uses Long Short-Term Memory (LSTM) neural networks to forecast industrial compressor sensor readings and identify abnormal operating conditions.

The project combines time-series forecasting, reconstruction-error-based anomaly detection, model evaluation, data visualization, and an interactive Streamlit dashboard.

## Project Overview

Industrial compressors operate under changing conditions. Monitoring sensor readings and identifying unusual behavior can help maintenance teams investigate potential equipment problems before they become serious.

This project demonstrates a data-driven approach to:

- Generate synthetic industrial compressor sensor data.
- Preprocess and scale time-series sensor readings.
- Engineer time-based features.
- Train an LSTM model to predict sensor values.
- Identify anomalies using prediction errors.
- Calibrate an anomaly threshold using validation data.
- Evaluate anomaly detection performance.
- Visualize compressor behavior through an interactive dashboard.

**Note:** This project uses synthetic data generated for experimentation and educational purposes. Its results should not be interpreted as validated performance on real industrial equipment.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| NumPy | Numerical computations |
| Pandas | Data manipulation |
| PyTorch | LSTM neural network |
| Scikit-learn | Preprocessing and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive dashboard |
| PyYAML | YAML configuration support |
| Joblib | Saving preprocessing objects |
| Pytest | Testing framework |

## Dataset Description

The synthetic dataset represents five industrial compressors:

- COMP_001
- COMP_002
- COMP_003
- COMP_004
- COMP_005

Each compressor contains 4,000 time-series records, giving a total of 20,000 records.

### Sensor Features

| Feature | Description |
|---|---|
| Pressure | Compressor operating pressure |
| Temperature | Operating temperature |
| Motor Current | Electrical current consumed by the motor |
| Vibration | Compressor vibration measurement |
| Airflow | Airflow measurement |
| RPM | Rotational speed |
| Hour Sin | Cyclical representation of hour |
| Hour Cos | Cyclical representation of hour |

The model uses eight input features to predict six sensor measurements.

### Synthetic Anomaly Types

The dataset includes four types of abnormal operating conditions:

1. Pressure Fluctuation
2. Temperature Increase
3. High Motor Current
4. Excessive Vibration

The training data includes the first two anomaly types, while the testing data includes the latter two. This provides a test of detection on anomaly patterns not included in training.

## Project Architecture

```text
Industrial_Forecast/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── compressor_data.csv
│   │
│   └── processed/
│       ├── train_data.csv
│       └── test_data.csv
│
├── src/
│   ├── generate_data.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── anomaly_analysis.py
│   ├── visualize_anomalies.py
│   └── dashboard.py
│
├── outputs/
│   ├── compressor_lstm.pth
│   ├── compressor_lstm_baseline.pth
│   ├── scaler.joblib
│   ├── forecast_predictions.csv
│   ├── anomaly_results.csv
│   ├── anomaly_threshold.txt
│   └── visualization files
│
├── tests/
├── requirements.txt
├── check_environment.py
├── README.md
└── PROJECT_DOCUMENTATION.md
```

## Machine Learning Workflow

### 1. Data Generation

Synthetic sensor data is generated for five compressors, with normal operating patterns and injected abnormal conditions.

### 2. Data Preprocessing

- Load and validate the dataset.
- Sort records chronologically.
- Split the data into training and testing periods.
- Scale numerical sensor features using StandardScaler.
- Save the fitted scaler for reuse.

The training dataset contains 16,000 records, and the testing dataset contains 4,000 records.

### 3. Feature Engineering

Cyclical time features are created using sine and cosine transformations:

- `hour_sin`
- `hour_cos`

These features represent the cyclical nature of time.

### 4. LSTM Model Training

A PyTorch LSTM model learns temporal relationships between historical sensor readings.

Model configuration:

| Parameter | Value |
|---|---:|
| Input features | 8 |
| Hidden size | 64 |
| LSTM layers | 2 |
| Output features | 6 |
| Sequence length | 30 |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 30 |
| Batch size | 64 |
| Loss function | Mean Squared Error |

The model uses the previous 30 time steps to predict the next sensor measurements.

### 5. Forecast Evaluation

Predicted sensor readings are compared with actual readings.

Evaluation includes forecasting error analysis and comparisons between the baseline and normal-only training approaches.

### 6. Anomaly Detection

Prediction errors are used as anomaly scores.

A threshold is calculated using the 99th percentile of prediction errors from normal validation samples.

If a test sample's anomaly score exceeds the threshold, it is classified as anomalous.

This approach uses validation data for threshold calibration rather than selecting the threshold using the final test labels.

## Anomaly Detection Results

The final test evaluation produced the following results:

| Metric | Result |
|---|---:|
| Accuracy | 98.55% |
| Precision | 88.93% |
| Recall | 100.00% |
| F1-score | 94.14% |
| True Positives | 450 |
| True Negatives | 3,344 |
| False Positives | 56 |
| False Negatives | 0 |

### Confusion Matrix

| | Predicted Normal | Predicted Anomaly |
|---|---:|---:|
| Actual Normal | 3,344 | 56 |
| Actual Anomaly | 0 | 450 |

The model detected all 450 injected test anomalies, with 56 false-positive alerts.

These metrics describe performance on the generated synthetic test dataset only.

## Streamlit Dashboard

The project includes an interactive dashboard with five sections:

1. **Plant Overview:** Overall compressor monitoring and anomaly summaries.
2. **Sensor Monitoring:** Sensor trends and anomaly markers.
3. **Anomaly Alerts:** Predicted anomaly alerts and downloadable results.
4. **Model Performance:** Classification metrics, confusion matrix, and anomaly score distribution.
5. **Forecasting:** Actual versus predicted sensor readings, MAE, RMSE, error distributions, and downloadable forecast results.

The forecasting page displays historical one-step-ahead predictions on the test dataset. It is not a live forecasting system.

## Installation and Execution

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Industrial_Forecast
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Check the environment

```bash
python check_environment.py
```

### 6. Run the project pipeline

Run the following commands from the project root:

```bash
python -m src.generate_data
python -m src.data_loader
python -m src.preprocessing
python -m src.feature_engineering
python -m src.dataset
python -m src.train
python -m src.evaluate
python -m src.anomaly_analysis
python -m src.visualize_anomalies
```

The pipeline should be executed in this order because later stages depend on the outputs of earlier stages.

### 7. Launch the Streamlit dashboard

```bash
streamlit run src/dashboard.py
```

The dashboard will open in your browser.

## Generated Outputs

Important project outputs include:

- Trained LSTM model.
- Baseline LSTM model.
- Saved feature scaler.
- Forecast predictions.
- Anomaly detection results.
- Calibrated anomaly threshold.
- Anomaly score distribution.
- Actual versus predicted anomaly comparison.
- Compressor anomaly timeline.

## Limitations

- The dataset is synthetic rather than collected from real industrial equipment.
- The anomaly patterns are predefined.
- The detection threshold is calibrated using normal validation samples.
- Forecasting is evaluated on historical test records and is not live future forecasting.
- Real-world deployment would require validation against actual compressor data, operational conditions, and maintenance outcomes.

## Future Improvements

- Integrate real industrial sensor data.
- Implement live sensor monitoring.
- Develop multi-step forecasting.
- Add automated alert notifications.
- Experiment with Transformer-based time-series models.
- Implement model drift monitoring.
- Add explainable anomaly detection.
- Deploy the dashboard to a cloud platform.

## Author

Developed as an end-to-end industrial time-series forecasting and anomaly detection project using Python, PyTorch, and Streamlit.
