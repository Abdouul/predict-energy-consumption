# Regional Electricity Consumption Forecasting

This project predicts future electricity consumption for French regions using a PyTorch LSTM time series model and a Streamlit dashboard.

The demo also recommends low-consumption time slots, for example to charge an electric vehicle or run electrical appliances.

## Project Objective

The goal is to:

- collect electricity consumption data for French regions,
- add weather data from Open-Meteo,
- train a PyTorch deep learning model for time series forecasting,
- predict the next 24 hours of regional consumption,
- recommend the best low-consumption time window.

## Data Sources

- Electricity consumption: RTE / ODRÉ eCO2mix regional data.
- Weather: Open-Meteo historical and forecast APIs.

If an API request fails, the project generates realistic synthetic data. This keeps the demo working during an oral defense even without internet access.

## Project Structure

```text
app.py                     Streamlit dashboard
train.py                   Full regional training pipeline
quick_train.py             Fast training script for a demo model
requirements.txt           Python dependencies
data/raw_data.csv          Latest collected raw data
data/processed_data.csv    Data after feature engineering
models/energy_model.pth    Saved PyTorch LSTM model
models/scaler.pkl          Saved feature and target scalers
src/data_collection.py     RTE/Open-Meteo data collection
src/preprocessing.py       Cleaning, time features, lag features
src/sequence_loader.py     Converts rows into LSTM sequences
src/model.py               PyTorch LSTM architecture
src/training.py            Custom PyTorch training loop
src/recommendation.py      24-hour forecast and best time-window logic
src/inference.py           Extra inference pipeline utilities
```

## Model Architecture

The model is a PyTorch LSTM.

- Input sequence: last 24 hours.
- Target: next hour electricity consumption in MW.
- Features: weather, hour/day/month features, lag values, rolling averages, and region code.
- Output: one predicted consumption value.

For a 24-hour forecast, the app predicts one hour, adds that prediction to the history, then predicts the next hour. This is repeated until the selected horizon is reached.

## Recommendation Logic

After the app predicts the next hours, it tests all possible windows of the selected duration.

Example: if the user selects 3 hours, the app compares:

- hour 1 to hour 3,
- hour 2 to hour 4,
- hour 3 to hour 5,
- and so on.

The best recommendation is the window with the lowest average predicted consumption.

## Installation

```bash
pip install -r requirements.txt
```

If you use the included virtual environment on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train the Model

For a quick demo model, the script uses three regions, about three months of RTE data, and 20 epochs:

```powershell
.\.venv\Scripts\python.exe quick_train.py
```

For the full regional training script, the script uses all configured regions, about six months of RTE data, and 60 epochs:

```powershell
.\.venv\Scripts\python.exe train.py
```

Training saves:

- `models/energy_model.pth`
- `models/scaler.pkl`
- `data/raw_data.csv`
- `data/processed_data.csv`
- `data/evaluation_predictions.csv`

The evaluation file compares the true consumption values with the model predictions in MW. This is useful for checking whether the model is learning the real RTE patterns.

Latest quick training result:

- MAE: about 276 MW
- RMSE: about 379 MW
- MAPE: about 3.30%
- R2: about 0.966

## Run the Streamlit App

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## What to Explain During the Oral Defense

1. The data combines electricity consumption and weather by timestamp.
2. Time features help the model learn daily and weekly patterns.
3. Lag features give the model recent past consumption values.
4. LSTM is used because electricity consumption is time series data.
5. The recommendation is based on predicted low-consumption periods, not random rules.
6. The project uses PyTorch for the model and Streamlit for the demo.
7. The model is not forced to avoid evening hours. If it recommends a time slot, it is because the predicted consumption for that slot is low.
