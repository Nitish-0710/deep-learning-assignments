# Assignment 6 — LSTM-Based Time-Series Forecasting

## Problem Statement

Develop an LSTM-based model for time-series forecasting using a stock price, weather, or sales dataset.

## Overview

This assignment implements and compares **LSTM-based time-series forecasting models** using two different approaches:

1. **Univariate forecasting** using the Airline Passengers dataset.
2. **Multivariate forecasting** using the Jena Climate dataset.

For both implementations, multiple look-back windows are evaluated to study how the amount of historical information provided to the LSTM affects forecasting performance.

The models are evaluated using:

- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**

The notebooks also generate prediction plots, metric comparisons, and training/validation loss curves.

## Repository Structure

```text
Assignment 6/
│
├── Assignment6_univariate.ipynb
├── Assignment6_multivariate.ipynb
├── README.md
│
├── data/
│   ├── airline-passengers.csv
│   └── README.md
│
└── results/
    │
    ├── multivariate/
    │   ├── all_windows_actual_vs_predicted.png
    │   ├── loss_24_step_window.png
    │   ├── loss_48_step_window.png
    │   ├── loss_72_step_window.png
    │   ├── prediction_24_step_window.png
    │   ├── prediction_48_step_window.png
    │   ├── prediction_72_step_window.png
    │   └── window_metric_comparison.png
    │
    └── univariate/
        ├── all_windows_actual_vs_predicted.png
        ├── loss_6_month_window.png
        ├── loss_12_month_window.png
        ├── loss_24_month_window.png
        ├── passenger_time_series.png
        ├── prediction_6_month_window.png
        ├── prediction_12_month_window.png
        ├── prediction_24_month_window.png
        └── window_metric_comparison.png
```

# 1. Univariate LSTM Forecasting

## Dataset

The univariate implementation uses the **Airline Passengers dataset**.

The dataset contains monthly airline passenger counts with:

- `Month` — observation date
- `Passengers` — number of passengers

The notebook uses `Passengers` as the single forecasting variable.

If the dataset is not available locally, the notebook downloads it automatically and stores it as:

```text
data/airline-passengers.csv
```

## Objective

The univariate experiment compares three historical look-back windows:

| Look-back | Historical Context         |
| --------: | -------------------------- |
|  6 months | Short-term context         |
| 12 months | One complete yearly cycle  |
| 24 months | Two complete yearly cycles |

For each window, the LSTM receives the previous observations and predicts the passenger count for the next month.

For example, with a 12-month look-back:

```text
Months 1–12  → Month 13
Months 2–13  → Month 14
Months 3–14  → Month 15
...
```

## Preprocessing

The notebook:

1. Converts the `Month` column to datetime.
2. Sorts observations chronologically.
3. Splits the dataset into:
   - 80% training
   - 20% testing

4. Fits a `MinMaxScaler` only on the training data.
5. Applies the scaler to the training and test data.
6. Creates overlapping time-series sequences.

## Model Architecture

The same architecture is used for all three look-back windows:

```text
Input
  ↓
LSTM (64 units)
  ↓
Dropout (0.2)
  ↓
Dense (32, ReLU)
  ↓
Dense (1)
```

### Training Configuration

- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Maximum epochs: 100
- Batch size: 16
- Validation split: 10%
- Shuffle: False
- Early stopping patience: 15
- Best weights restored using `restore_best_weights=True`

## Evaluation

Each model is evaluated using:

- MAE
- RMSE

Training time and number of epochs are also recorded.

## Generated Results

The univariate notebook generates:

- Passenger time-series plot
- MAE/RMSE comparison
- Combined actual vs predicted plot
- Individual prediction plots for 6, 12, and 24 months
- Training/validation loss plots for each look-back window

Results are stored in:

```text
results/univariate/
```

# 2. Multivariate LSTM Forecasting

## Dataset

The multivariate implementation uses the **Jena Climate Dataset**.

The dataset contains weather measurements recorded at regular time intervals at the Max Planck Institute for Biogeochemistry in Jena, Germany.

The dataset contains:

- 420,451 observations
- 15 columns
- 1 timestamp column
- 14 numerical weather variables

The notebook automatically downloads and extracts the dataset if it is not available locally.

The expected local file is:

```text
data/jena_climate_2009_2016.csv
```

The download source used by the notebook is:

```text
https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
```

Therefore, the Jena Climate CSV does not need to be manually included in the repository.

## Objective

The multivariate experiment uses multiple weather-related variables as input to predict the temperature at the next time step.

Three look-back windows are compared:

| Look-back | Historical Context |
| --------: | ------------------ |
|  24 steps | 4 hours            |
|  48 steps | 8 hours            |
|  72 steps | 12 hours           |

The Jena Climate observations are recorded at 10-minute intervals.

## Input Features

The model uses the following 14 weather variables:

```text
p (mbar)
T (degC)
Tpot (K)
Tdew (degC)
rh (%)
VPmax (mbar)
VPact (mbar)
VPdef (mbar)
sh (g/kg)
H2OC (mmol/mol)
rho (g/m**3)
wv (m/s)
max. wv (m/s)
wd (deg)
```

The prediction target is:

```text
T (degC)
```

The `Date Time` column is retained for chronological ordering and visualization and is not used as a numerical input feature.

## Preprocessing

The notebook:

1. Converts `Date Time` to datetime.
2. Sorts the observations chronologically.
3. Checks for missing values and dataset information.
4. Uses all 14 numerical weather variables as input features.
5. Uses `T (degC)` as the target.
6. Performs an 80/20 chronological train-test split.
7. Applies separate Min-Max scalers to the features and target.
8. Fits both scalers only on the training data.
9. Creates multivariate time-series sequences.

### Data Split

```text
Total observations:      420451
Training observations:   336360
Testing observations:     84091
```

### Sequence Shapes

| Look-back | Training Input     | Testing Input     |
| --------: | ------------------ | ----------------- |
|        24 | `(336336, 24, 14)` | `(84091, 24, 14)` |
|        48 | `(336312, 48, 14)` | `(84091, 48, 14)` |
|        72 | `(336288, 72, 14)` | `(84091, 72, 14)` |

Each input sequence therefore contains:

```text
look-back time steps × 14 features
```

## Model Architecture

The same architecture is used for all three multivariate experiments:

```text
Input (look_back, 14)
        ↓
LSTM (64 units)
        ↓
Dropout (0.2)
        ↓
Dense (32, ReLU)
        ↓
Dense (1)
```

### Training Configuration

- Optimizer: Adam
- Loss: Mean Squared Error (MSE)
- Metric: Mean Absolute Error (MAE)
- Maximum epochs: 100
- Batch size: 256
- Validation split: 10%
- Shuffle: False
- Early stopping patience: 15
- Best weights restored using `restore_best_weights=True`

The same architecture and training configuration are used for all three look-back windows so that the effect of sequence length can be compared.

## Evaluation

Each configuration is evaluated using:

- MAE
- RMSE
- Number of training epochs
- Training time

Predictions are inverse-transformed back to degrees Celsius before calculating MAE and RMSE.

## Recorded Results

The recorded experimental results are:

|    Look-back |   MAE (°C) |  RMSE (°C) | Epochs | Training Time |
| -----------: | ---------: | ---------: | -----: | ------------: |
|     24 steps |     1.9052 |     2.2893 |     16 |      316.11 s |
| **48 steps** | **1.0612** | **1.2735** |     16 |      605.49 s |
|     72 steps |     1.1704 |     1.4045 |     16 |      987.94 s |

For this recorded experiment, the **48-step look-back produced the lowest MAE and RMSE**.

This corresponds to approximately **8 hours of historical context**.

## Generated Results

The multivariate notebook generates:

- MAE/RMSE comparison
- Combined actual vs predicted temperature plot
- Individual prediction plots for 24, 48, and 72 steps
- Training/validation loss plots for each look-back window

Results are stored in:

```text
results/multivariate/
```

# 3. Comparison of the Two Approaches

| Aspect            | Univariate              | Multivariate                             |
| ----------------- | ----------------------- | ---------------------------------------- |
| Dataset           | Airline Passengers      | Jena Climate                             |
| Input variables   | 1                       | 14                                       |
| Target            | Passengers              | Temperature                              |
| Look-back windows | 6, 12, 24 months        | 24, 48, 72 steps                         |
| Prediction type   | Next month's passengers | Next time-step temperature               |
| Scaling           | Min-Max                 | Separate Min-Max for features and target |
| LSTM units        | 64                      | 64                                       |
| Dropout           | 0.2                     | 0.2                                      |
| Dense layer       | 32, ReLU                | 32, ReLU                                 |
| Output            | 1                       | 1                                        |
| Optimizer         | Adam                    | Adam                                     |
| Loss              | MSE                     | MSE                                      |
| Evaluation        | MAE, RMSE               | MAE, RMSE                                |

The univariate implementation demonstrates forecasting using the historical values of a single variable, while the multivariate implementation uses multiple weather measurements simultaneously to predict temperature.

# 4. Early Stopping

Both notebooks use early stopping:

```python
EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)
```

This monitors validation loss during training.

If validation loss does not improve for 15 consecutive epochs, training is stopped and the weights from the best validation-loss epoch are restored.

This helps prevent unnecessary training after validation performance stops improving.

# 5. Evaluation Metrics

## Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted values.

```text
MAE = mean(|actual - predicted|)
```

Lower MAE indicates smaller average prediction error.

## Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

```text
RMSE = sqrt(mean((actual - predicted)^2))
```

RMSE gives greater influence to larger prediction errors.

For both metrics:

```text
Lower value → Lower prediction error
```

# 6. Visualizations

Both notebooks generate visualizations for analyzing model performance.

### Univariate

```text
results/univariate/
```

Contains:

- Passenger time-series plot
- MAE/RMSE comparison
- Actual vs predicted comparison
- Individual prediction plots
- Training/validation loss plots

### Multivariate

```text
results/multivariate/
```

Contains:

- MAE/RMSE comparison
- Actual vs predicted temperature comparison
- Individual prediction plots
- Training/validation loss plots

# 7. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras
- Jupyter Notebook

# 8. Running the Assignment

## 1. Create and Activate the Environment

Create a Python virtual environment:

```bash
python -m venv deep-learning
```

Activate the environment on Windows:

```bash
deep-learning\Scripts\activate
```

## 2. Open the Assignment Directory

After cloning the repository, navigate to the Assignment 6 directory:

```bash
cd "Assignment 6"
```

## 3. Start Jupyter Notebook

```bash
jupyter notebook
```

## 4. Run Either Notebook

### Univariate

Open:

```text
Assignment6_univariate.ipynb
```

The notebook uses:

```text
data/airline-passengers.csv
```

If the dataset is not available locally, the notebook downloads it automatically.

### Multivariate

Open:

```text
Assignment6_multivariate.ipynb
```

The notebook automatically downloads and extracts the Jena Climate dataset if:

```text
data/jena_climate_2009_2016.csv
```

is not present.

# 9. Reproducibility

Both notebooks set the random seed to:

```python
SEED = 42
```

and seed:

```python
random
numpy
tensorflow
```

The experiments also preserve chronological ordering and use:

```python
shuffle=False
```

for model training.

# 10. Key Takeaways

- LSTMs can be used for both univariate and multivariate time-series forecasting.
- A look-back window determines how much historical information is provided to the model.
- The univariate experiment studies 6-, 12-, and 24-month historical windows.
- The multivariate experiment studies 24-, 48-, and 72-step historical windows.
- The multivariate model uses 14 weather-related features to predict the next temperature value.
- Chronological train-test splitting is used to preserve the temporal nature of the data.
- Scaling is fitted only on the training data to avoid test-data leakage through preprocessing.
- MAE and RMSE are used to evaluate forecasting performance.
- Early stopping is used to stop training when validation loss stops improving.
- For the recorded Jena Climate experiment, the 48-step look-back achieved the lowest MAE and RMSE.
