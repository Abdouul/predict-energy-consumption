# Energy Consumption Forecasting Project

A real-time energy consumption forecasting system using LSTM deep learning, built with PyTorch and Streamlit.

## Project Overview

This project implements a complete deep learning pipeline for forecasting energy consumption in France. It uses historical data for training and real-time API data for inference.

### Key Features

- **Real-time Data**: Fetches live energy and weather data from APIs
- **LSTM Model**: Custom PyTorch LSTM with multiple layers and dropout
- **Live Dashboard**: Streamlit app with auto-refresh and real-time predictions
- **Complete Pipeline**: From data collection to prediction

## Project Structure

```
yolo_demo/
├── app.py                 # Streamlit dashboard
├── train.py               # Main training script
├── quick_train.py        # Quick training test
├── requirements.txt      # Python dependencies
├── models/
│   ├── energy_model.pth  # Trained model weights
│   └── scaler.pkl        # Data normalizer
├── data/
│   ├── raw_data.csv      # Raw collected data
│   └── processed_data.csv
└── src/
    ├── data_collection.py    # API data fetching
    ├── preprocessing.py     # Data cleaning & features
    ├── sequence_loader.py   # LSTM sequence creation
    ├── model.py             # LSTM architecture
    ├── training.py          # Custom training loop
    └── inference.py         # Real-time inference
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

```bash
python train.py
```

Or for quick training:

```bash
python quick_train.py
```

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser with:
- Real-time energy consumption display
- Weather data
- Next hour prediction
- Auto-refresh every 60 seconds

## Data Sources

- **Energy Data**: RTE Eco2Mix (French electricity grid)
- **Weather Data**: Open-Meteo API

## Model Architecture

- **Input**: 24 hours of historical data (24 timesteps)
- **Features**: Weather + time features + lag features
- **Architecture**: LSTM with 2 layers, 128 hidden units, dropout
- **Output**: Next hour's consumption prediction

## Explanation of Key Components

### 1. Data Collection (`src/data_collection.py`)
- Fetches real-time energy consumption from RTE API
- Fetches weather data from Open-Meteo API
- Generates synthetic data when APIs unavailable

### 2. Preprocessing (`src/preprocessing.py`)
- Cleans data and handles missing values
- Adds time-based features (hour, day, month, weekend)
- Adds cyclical encoding for temporal features
- Adds lag features (past 1, 2, 3, 6, 12, 24 hours)

### 3. Sequence Loader (`src/sequence_loader.py`)
- Converts tabular data to LSTM sequences
- Input: 24-hour window
- Output: Next hour prediction

### 4. Model (`src/model.py`)
- Multi-layer LSTM with dropout
- Fully connected output layer
- ~211K trainable parameters

### 5. Training (`src/training.py`)
- Custom PyTorch training loop (not sklearn)
- Adam optimizer with learning rate scheduler
- Early stopping to prevent overfitting

### 6. Inference (`src/inference.py`)
- Real-time prediction pipeline
- Fetches latest data and makes predictions

### 7. Streamlit App (`app.py`)
- Live dashboard with metrics
- Auto-refreshing graphs
- System status display

## For Oral Defense Preparation

### What to Explain:

1. **Why LSTM?**
   - Good for sequential data
   - Can learn long-term dependencies
   - Standard for time series forecasting

2. **Why 24-hour sequence?**
   - Captures daily patterns
   - Enough history for accurate predictions

3. **Why weather features?**
   - Temperature affects heating/cooling demand
   - Strong correlation with consumption

4. **Why lag features?**
   - Energy consumption is autocorrelated
   - Past values help predict future

5. **Custom training loop vs sklearn**
   - Shows understanding of deep learning fundamentals
   - More control over the process

## Technical Details

- **Framework**: PyTorch
- **UI**: Streamlit
- **Data**: Real-time APIs (RTE, Open-Meteo)
- **Model**: LSTM (2 layers, 128 hidden units)
- **Sequence Length**: 24 hours
- **Features**: 24 (weather + time + lags)

## Notes

- The RTE API may not be accessible, so the project uses synthetic data when needed
- The model is trained on synthetic data for demonstration
- In production, you would use real historical data from RTE

## Author

Student Project for Master's Program