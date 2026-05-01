"""
Forecast and recommendation helpers.

These functions are intentionally small so they are easy to explain:
1. predict the next hours one by one,
2. find the time window with the lowest average predicted consumption.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import torch

from src.preprocessing import add_lag_features, add_region_features, add_time_features, get_feature_columns


def _prepare_model_sequence(history, model, scalers):
    """Create the latest 24-hour input sequence for the PyTorch model."""
    df = history.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = add_time_features(df)
    df = add_region_features(df)
    df = add_lag_features(df)
    df = df.dropna()

    if len(df) < 24:
        return None

    feature_cols = get_feature_columns(df)
    expected_features = getattr(model, "input_size", len(feature_cols))

    # Old models were trained without region_code. New regional models include it.
    if len(feature_cols) != expected_features and "region_code" in feature_cols:
        feature_cols.remove("region_code")

    if len(feature_cols) != expected_features:
        print(f"Feature mismatch: expected {expected_features}, got {len(feature_cols)}")
        return None

    sequence = df[feature_cols].tail(24).values
    if scalers and scalers.get("feature_scaler") is not None:
        sequence = scalers["feature_scaler"].transform(sequence)

    return torch.FloatTensor(sequence).unsqueeze(0)


def predict_next_hours(model, scalers, recent_data, future_weather, region, horizon=24):
    """Predict regional consumption for the next horizon hours."""
    if model is None or len(recent_data) < 48:
        return pd.DataFrame()

    history = recent_data.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"])
    history = history.sort_values("timestamp")
    history["region"] = region

    weather = future_weather.copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])
    weather = weather.sort_values("timestamp")

    results = []
    last_time = history["timestamp"].max()

    for step in range(1, horizon + 1):
        next_time = last_time + timedelta(hours=step)
        weather_row = weather[
            (weather["timestamp"] >= next_time)
            & (weather["timestamp"] < next_time + timedelta(hours=1))
        ].head(1)

        if weather_row.empty:
            last_weather = history.tail(1).iloc[0]
            weather_values = {
                "temperature": last_weather["temperature"],
                "humidity": last_weather["humidity"],
                "wind_speed": last_weather["wind_speed"],
                "cloud_cover": last_weather["cloud_cover"],
            }
        else:
            weather_values = weather_row.iloc[0][["temperature", "humidity", "wind_speed", "cloud_cover"]].to_dict()

        model_input = _prepare_model_sequence(history, model, scalers)
        if model_input is None:
            break

        with torch.no_grad():
            prediction = model(model_input).item()

        if scalers and scalers.get("target_scaler") is not None:
            prediction = scalers["target_scaler"].inverse_transform([[prediction]])[0][0]

        prediction = float(max(prediction, 0))
        row = {
            "timestamp": next_time,
            "region": region,
            "predicted_consumption_mw": prediction,
            **weather_values,
        }
        results.append(row)

        history = pd.concat([
            history,
            pd.DataFrame([{
                "timestamp": next_time,
                "region": region,
                "consumption_mw": prediction,
                **weather_values,
            }])
        ], ignore_index=True)

    return pd.DataFrame(results)


def recommend_low_consumption_windows(forecast_df, duration_hours=3, top_n=3):
    """Return the lowest average consumption windows."""
    if forecast_df.empty or len(forecast_df) < duration_hours:
        return pd.DataFrame()

    rows = []
    for start in range(0, len(forecast_df) - duration_hours + 1):
        window = forecast_df.iloc[start:start + duration_hours]
        rows.append({
            "start_time": window["timestamp"].iloc[0],
            "end_time": window["timestamp"].iloc[-1] + timedelta(hours=1),
            "duration_hours": duration_hours,
            "average_consumption_mw": window["predicted_consumption_mw"].mean(),
            "minimum_consumption_mw": window["predicted_consumption_mw"].min(),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("average_consumption_mw")
        .head(top_n)
        .reset_index(drop=True)
    )
