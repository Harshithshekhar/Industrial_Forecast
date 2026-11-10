# Industrial Compressor Forecasting and Anomaly Detection
## Technical Project Documentation

## 1. Introduction

This project demonstrates an end-to-end time-series machine learning workflow for monitoring industrial compressor sensor measurements.

The primary objective is to use historical sensor readings to predict future measurements and identify unusual operating behavior through prediction errors.

The project uses a Long Short-Term Memory (LSTM) neural network implemented using PyTorch.

It also includes a Streamlit dashboard for monitoring sensor trends, forecasting results, and anomaly detection performance.

The dataset is synthetically generated for experimentation and does not represent measurements collected from an actual industrial plant.

## 2. Business Problem

Industrial compressors are important components in manufacturing and industrial operations.

Unexpected operating conditions can lead to equipment damage, maintenance costs, and operational interruptions.

Traditional monitoring systems often rely on fixed thresholds for individual sensor measurements.

However, abnormal behavior may involve relationships between multiple sensor variables and their changes over time.

This project explores how time-series forecasting errors can help identify unusual compressor behavior.

### Business Objectives

- Monitor multiple compressor units.
- Understand normal sensor behavior.
- Forecast upcoming sensor measurements.
- Identify deviations from expected operating patterns.
- Reduce missed anomaly detections.
- Present monitoring information through an interactive dashboard.

## 3. Dataset Description

The project generates a synthetic dataset containing:

- Five compressor units.
- 4,000 records per compressor.
- 20,000 total records.
- One-minute sampling intervals.
- Six sensor measurements.
- Two cyclical time features.

### Input Features

| Feature | Description |
|---|---|
| Pressure | Operating pressure |
| Temperature | Operating temperature |
| Motor Current | Electrical current |
| Vibration | Mechanical vibration |
| Airflow | Airflow measurement |
| RPM | Rotational speed |
| Hour Sin | Sine transformation of hour |
| Hour Cos | Cosine transformation of hour |

### Target Features

The model predicts six sensor variables:

1. Pressure
2. Temperature
3. Motor Current
4. Vibration
5. Airflow
6. RPM

### Anomaly Types

| Anomaly | Description |
|---|---|
| Pressure Fluctuation | Abnormal pressure behavior |
| Temperature Increase | Unusual temperature increase |
| High Motor Current | Excessive motor current |
| Excessive Vibration | Unusual vibration behavior |

The training and testing periods contain different injected anomaly types. This is an experimental design choice intended to test detection of patterns not included in training.

## 4. Data Preprocessing

The preprocessing workflow includes:

1. Loading the raw CSV dataset.
2. Checking data quality.
3. Sorting records chronologically.
4. Splitting each compressor's records into training and testing periods.
5. Scaling numerical sensor measurements.
6. Saving the processed datasets.

### Train-Test Split

| Dataset | Records |
|---|---:|
| Training | 16,000 |
| Testing | 4,000 |
| Total | 20,000 |

The scaler is fitted using training data only to avoid information leakage from the test dataset.

The scaler is saved as:

`outputs/scaler.joblib`

## 5. Feature Engineering

Time is cyclical. For example, 23:00 and 00:00 are close in time, even though their numerical hour values differ considerably.

To represent this relationship, sine and cosine transformations are applied.

\[
hour\_sin = \sin\left(\frac{2\pi \times hour}{24}\right)
\]

\[
hour\_cos = \cos\left(\frac{2\pi \times hour}{24}\right)
\]

These features help the model learn time-dependent operating patterns.

## 6. LSTM Model Architecture

LSTM stands for Long Short-Term Memory.

It is a type of recurrent neural network designed to learn patterns from sequential data.

Unlike a standard feedforward neural network, an LSTM maintains information from previous time steps.

This makes it suitable for time-series forecasting.

### Model Configuration

| Parameter | Value |
|---|---:|
| Input size | 8 |
| Hidden size | 64 |
| Number of LSTM layers | 2 |
| Output size | 6 |
| Sequence length | 30 |
| Dropout | 0.2 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 30 |
| Batch size | 64 |
| Loss function | MSELoss |

### Model Workflow

The model receives a sequence of 30 historical time steps.

Each time step contains eight input features.

The LSTM processes the sequence and uses its final hidden representation to predict six sensor measurements for the next time step.

Conceptually:

```text
Historical Sensor Readings
          |
          v
   30 Time Steps
          |
          v
    LSTM Layer 1
          |
          v
    LSTM Layer 2
          |
          v
    Fully Connected Layer
          |
          v
 Predicted Sensor Readings
```

### Training Strategy

The training dataset is restricted to normal operating sequences.

A validation section is kept separate from the fitting period and is used for anomaly threshold calibration.

This reduces the risk of calibrating the threshold using the same samples used to fit the model.

The trained model is saved as:

`outputs/compressor_lstm.pth`

A baseline model is also retained:

`outputs/compressor_lstm_baseline.pth`

## 7. Forecast Evaluation

The trained model generates one-step-ahead predictions.

Actual and predicted sensor values are compared to understand forecasting performance.

### Evaluation Metrics

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted values.

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

**Root Mean Squared Error (RMSE)**

Measures the square root of the average squared prediction error.

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

The dashboard calculates these metrics for the selected compressor and sensor.

## 8. Anomaly Detection Methodology

The project uses prediction error as an anomaly score.

The LSTM predicts expected sensor measurements. When actual measurements differ significantly from predictions, the resulting prediction error may indicate abnormal behavior.

### Anomaly Score

The mean absolute error across the six standardized sensor predictions is calculated for each sample.

\[
AnomalyScore = \frac{1}{6}\sum_{j=1}^{6}|y_j-\hat{y}_j|
\]

Where:

- \(y_j\) represents the actual standardized sensor measurement.
- \(\hat{y}_j\) represents the predicted standardized sensor measurement.
- Six represents the number of target sensor variables.

### Threshold Calibration

The anomaly threshold is determined using the 99th percentile of prediction scores from normal validation samples.

\[
Threshold = Percentile(ValidationScores,99)
\]

The calibrated threshold is:

`0.707430`

### Classification Rule

```python
if anomaly_score > threshold:
    predicted_anomaly = 1
else:
    predicted_anomaly = 0
```

A prediction error above the threshold is classified as an anomaly.

## 9. Anomaly Detection Results

The final test evaluation contains 3,850 valid sequence samples.

### Test Dataset Distribution

| Category | Samples |
|---|---:|
| Normal | 3,400 |
| Anomalies | 450 |
| Total | 3,850 |

### Confusion Matrix

| | Predicted Normal | Predicted Anomaly |
|---|---:|---:|
| Actual Normal | 3,344 | 56 |
| Actual Anomaly | 0 | 450 |

### Performance Metrics

| Metric | Score |
|---|---:|
| Accuracy | 98.55% |
| Precision | 88.93% |
| Recall | 100.00% |
| F1-score | 94.14% |

### Interpretation

**Accuracy:** Approximately 98.55% of test samples were classified correctly.

**Precision:** Approximately 88.93% of predicted anomalies were actual anomalies.

**Recall:** All injected test anomalies were detected.

**F1-score:** The combined precision and recall score was approximately 94.14%.

The model produced 56 false-positive alerts and no false negatives on this synthetic test dataset.

These results should not be interpreted as guaranteed detection performance in a real industrial environment.

## 10. Visualization and Dashboard

The project includes three saved anomaly visualizations:

1. Anomaly score distribution.
2. Actual versus predicted anomaly count comparison.
3. Compressor-wise anomaly timeline.

The Streamlit dashboard contains five pages.

### Page 1: Plant Overview

Displays:

- Overall monitoring KPIs.
- Actual anomaly distribution.
- Actual versus predicted anomaly counts.
- Compressor-wise anomaly summaries.

### Page 2: Sensor Monitoring

Displays:

- Sensor trends.
- Actual anomaly markers.
- Predicted anomaly markers.
- Sensor statistics.

### Page 3: Anomaly Alerts

Displays:

- Calibrated anomaly threshold.
- Predicted anomaly records.
- Anomaly scores.
- Downloadable alert data.

### Page 4: Model Performance

Displays:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Anomaly score distribution.

### Page 5: Forecasting

Displays:

- Compressor selection.
- Sensor selection.
- Actual versus predicted readings.
- MAE and RMSE.
- Absolute error distribution.
- Forecast results table.
- CSV download.

The forecasting dashboard uses historical test-set predictions. It does not currently provide real-time sensor ingestion or live future forecasting.

## 11. Project Execution

### Environment Setup

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Check the environment:

```bash
python check_environment.py
```

### Execute the Pipeline

Run these commands from the project root:

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

### Launch Dashboard

```bash
streamlit run src/dashboard.py
```

## 12. Project Outputs

| Output | Description |
|---|---|
| `compressor_lstm.pth` | Final trained LSTM model |
| `compressor_lstm_baseline.pth` | Baseline model |
| `scaler.joblib` | Fitted StandardScaler |
| `forecast_predictions.csv` | Historical forecast predictions |
| `anomaly_results.csv` | Anomaly detection results |
| `anomaly_threshold.txt` | Calibrated threshold |
| `anomaly_score_distribution.png` | Score distribution |
| `anomaly_count_comparison.png` | Anomaly count comparison |
| `compressor_anomaly_timeline.png` | Compressor anomaly timeline |

## 13. Limitations

- The dataset is synthetically generated.
- The injected anomaly types are predefined.
- Real compressor behavior may be more complex.
- The model has not been validated on real industrial sensor data.
- The threshold may require recalibration for different machines and operating conditions.
- The current forecasting workflow produces historical one-step predictions rather than live future predictions.
- The dashboard does not currently connect to an industrial IoT or SCADA system.

## 14. Future Enhancements

Potential improvements include:

1. Integrating real industrial sensor datasets.
2. Implementing real-time sensor data ingestion.
3. Developing multi-step forecasting.
4. Adding automated maintenance alerts.
5. Experimenting with Transformer-based forecasting models.
6. Implementing model drift monitoring.
7. Adding explainable AI techniques for anomaly investigation.
8. Deploying the application to a cloud platform.
9. Building an API for external system integration.
10. Evaluating performance across different operating conditions.

## 15. Conclusion

This project demonstrates how deep learning can be applied to industrial time-series forecasting and anomaly detection.

An LSTM model is used to learn temporal patterns in compressor sensor measurements and predict expected operating values.

Prediction errors are then used to identify unusual behavior through a validation-calibrated anomaly threshold.

The project combines data preprocessing, feature engineering, neural network training, model evaluation, anomaly detection, visualization, and an interactive Streamlit dashboard.

It provides a practical foundation for further exploration of predictive maintenance systems using real industrial data.
