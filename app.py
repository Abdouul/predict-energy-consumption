"""
Streamlit dashboard for regional electricity consumption forecasting.
"""

from datetime import datetime, timedelta
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from src.data_collection import (
    collect_all_data_for_region,
    fetch_weather_forecast,
    get_region_info,
    get_region_names,
)
from src.model import EnergyLSTM
from src.recommendation import predict_next_hours, recommend_low_consumption_windows


st.set_page_config(
    page_title="Regional Energy Forecast",
    page_icon=":zap:",
    layout="wide",
)


@st.cache_data(ttl=3600)
def load_recent_region_data(region, days):
    """Load recent data ending today for the selected region."""
    end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=days)
    return collect_all_data_for_region(region, start_date, end_date, use_synthetic=True)


@st.cache_data(ttl=1800)
def load_region_weather_forecast(region, horizon):
    """Load future weather for the selected region."""
    info = get_region_info(region)
    days = max(2, int(horizon / 24) + 1)
    return fetch_weather_forecast(
        datetime.now(),
        days=days,
        lat=info["lat"],
        lon=info["lon"],
    )


@st.cache_resource
def load_model_and_scalers(model_version):
    """Load the PyTorch model and saved scalers."""
    model_path = "models/energy_model.pth"
    scaler_path = "models/scaler.pkl"

    if not os.path.exists(model_path):
        return None, None, None

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config") or checkpoint.get("model_config") or {}

    model = EnergyLSTM(
        input_size=config.get("input_size", 24),
        hidden_size=config.get("hidden_size", 128),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.2),
        output_size=config.get("output_size", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scalers = None
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as file:
            scalers = pickle.load(file)

    metrics = checkpoint.get("metrics")
    return model, scalers, metrics


@st.cache_data
def load_evaluation_predictions():
    """Load saved validation results where actual and predicted values are aligned."""
    path = "data/evaluation_predictions.csv"

    if not os.path.exists(path):
        return pd.DataFrame()

    return pd.read_csv(path)


def draw_forecast_chart(forecast_df, recommendations):
    """Draw the future forecast and highlight the recommended low-consumption window."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        forecast_df["timestamp"],
        forecast_df["predicted_consumption_mw"],
        marker="o",
        label="Predicted consumption",
        color="#1f5f99",
    )

    if not recommendations.empty:
        best = recommendations.iloc[0]
        ax.axvspan(best["start_time"], best["end_time"], color="#b7e4c7", alpha=0.5, label="Best window")

    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption (MW)")
    ax.set_title("Future Predicted Consumption and Recommended Window")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    st.pyplot(fig)


def display_real_vs_predicted_comparison(real_consumption, predicted_consumption, recent_data, forecast):
    """Display model error using the latest real value and the first prediction."""
    st.subheader("Real Consumption vs Predicted Consumption")

    # Error is positive when the model predicts too high and negative when it predicts too low.
    error = predicted_consumption - real_consumption
    absolute_error = abs(error)
    error_percentage = (absolute_error / real_consumption) * 100

    # The status text explains the prediction quality in simple words.
    if error_percentage < 5:
        prediction_status = "Prediction accurate"
        status_color = "#15803d"
    elif predicted_consumption > real_consumption:
        prediction_status = "Model overestimates"
        status_color = "#f97316" if error_percentage <= 10 else "#dc2626"
    else:
        prediction_status = "Model underestimates"
        status_color = "#f97316" if error_percentage <= 10 else "#dc2626"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Real Consumption (MW)", f"{real_consumption:,.0f} MW")
    col2.metric("Predicted Consumption (MW)", f"{predicted_consumption:,.0f} MW")
    col3.metric("Error (MW)", f"{error:+,.0f} MW")
    col4.metric("Absolute Error (MW)", f"{absolute_error:,.0f} MW")
    col5.metric("Error Percentage (%)", f"{error_percentage:.2f}%")

    # The color gives a quick visual interpretation of the error level.
    st.markdown(
        f"""
        <div style="
            padding: 12px;
            border-radius: 6px;
            background-color: {status_color};
            color: white;
            font-weight: 700;
            width: fit-content;
            margin-bottom: 12px;
        ">
            {prediction_status}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if error_percentage > 10:
        st.warning(
            "High prediction error detected. The model may need retraining "
            "or the current consumption pattern is unusual."
        )

    st.info(
        "The error represents the difference between the real electricity "
        "consumption and the predicted consumption."
    )

def display_prediction_quality_graphs():
    """Display graphs where actual and predicted values use the same samples."""
    st.subheader("Actual vs Predicted Energy Consumption")
    st.write(
        "These graphs use validation data, where the real consumption and the "
        "predicted consumption are known for the same samples. This is the correct "
        "way to evaluate model quality."
    )

    evaluation = load_evaluation_predictions()
    required_columns = {"actual_consumption_mw", "predicted_consumption_mw"}

    if evaluation.empty or not required_columns.issubset(evaluation.columns):
        st.warning("No aligned actual/predicted evaluation data is available.")
        return

    # These rows come from validation data, so actual and predicted values match the same samples.
    df = evaluation.dropna(subset=list(required_columns)).copy()
    if df.empty:
        st.warning("Evaluation data does not contain valid values.")
        return

    df["sample"] = range(len(df))
    df["error_mw"] = df["actual_consumption_mw"] - df["predicted_consumption_mw"]

    # Main comparison: same samples, actual in solid blue, prediction in dashed orange.
    st.markdown("**1. Actual vs Predicted line chart**")
    st.caption("If the orange dashed line follows the blue line, the model follows the real trend.")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        df["sample"],
        df["actual_consumption_mw"],
        label="Actual consumption",
        color="#2563eb",
        linestyle="-",
        linewidth=1.8,
    )
    ax.plot(
        df["sample"],
        df["predicted_consumption_mw"],
        label="Predicted consumption",
        color="#f97316",
        linestyle="--",
        linewidth=1.8,
    )
    ax.set_title("Actual vs Predicted Energy Consumption")
    ax.set_xlabel("Validation sample")
    ax.set_ylabel("Consumption (MW)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    col_left, col_right = st.columns(2)

    with col_left:
        # Error curve: values close to zero mean better predictions.
        st.markdown("**2. Prediction error over time**")
        st.caption("Values close to zero mean the prediction is close to the real value.")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df["sample"], df["error_mw"], color="#dc2626", linewidth=1.6)
        ax.axhline(0, color="#111827", linestyle="--", linewidth=1)
        ax.set_title("Prediction Error Over Time")
        ax.set_xlabel("Validation sample")
        ax.set_ylabel("Error (MW)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col_right:
        # Scatter plot: points near the diagonal represent accurate predictions.
        st.markdown("**3. Actual vs Predicted scatter plot**")
        st.caption("Points close to the diagonal line represent accurate predictions.")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(
            df["actual_consumption_mw"],
            df["predicted_consumption_mw"],
            color="#f97316",
            alpha=0.65,
        )
        min_value = min(df["actual_consumption_mw"].min(), df["predicted_consumption_mw"].min())
        max_value = max(df["actual_consumption_mw"].max(), df["predicted_consumption_mw"].max())
        ax.plot([min_value, max_value], [min_value, max_value], color="#2563eb", linestyle="--")
        ax.set_title("Actual vs Predicted Scatter Plot")
        ax.set_xlabel("Actual consumption (MW)")
        ax.set_ylabel("Predicted consumption (MW)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    # Latest validation sample as a compact bar chart.
    st.markdown("**4. Latest validation sample comparison**")
    st.caption("This compares the last actual value and the last predicted value in the validation file.")
    latest = df.iloc[-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        ["Actual", "Predicted"],
        [latest["actual_consumption_mw"], latest["predicted_consumption_mw"]],
        color=["#2563eb", "#f97316"],
    )
    ax.set_title("Latest Consumption Comparison")
    ax.set_ylabel("Consumption (MW)")
    ax.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig)


def display_model_performance(metrics):
    """Show saved validation metrics as clear cards."""
    st.subheader("Model Performance")

    if not metrics:
        st.info("No validation metrics were found in the saved model.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{float(metrics.get('MAE', 0)):,.0f} MW")
    col2.metric("RMSE", f"{float(metrics.get('RMSE', 0)):,.0f} MW")
    col3.metric("MAPE", f"{float(metrics.get('MAPE', 0)):.2f}%")
    col4.metric("R2", f"{float(metrics.get('R2', 0)):.3f}")


def display_business_recommendation(best_window):
    """Explain the best low-consumption time slot in business language."""
    st.subheader("Business Recommendation")

    if best_window is None:
        st.info("No recommendation is available for the selected horizon.")
        return

    start_time = best_window["start_time"].strftime("%H:%M")
    end_time = best_window["end_time"].strftime("%H:%M")
    average_consumption = best_window["average_consumption_mw"]

    st.success(
        f"Recommended start time: {start_time} because the predicted average "
        f"consumption is the lowest during this window "
        f"({start_time} - {end_time}, about {average_consumption:,.0f} MW)."
    )


def main():
    st.title("Regional Electricity Consumption Forecast")
    st.caption("PyTorch LSTM demo using RTE regional consumption data and Open-Meteo weather data.")

    with st.sidebar:
        st.header("Prediction settings")
        region = st.selectbox("French region", get_region_names(), index=get_region_names().index("Ile-de-France"))
        horizon = st.slider("Prediction horizon (hours)", min_value=6, max_value=48, value=24, step=1)
        duration = st.slider("Usage duration (hours)", min_value=1, max_value=8, value=3, step=1)
        history_days = st.slider("Recent history used (days)", min_value=3, max_value=30, value=7, step=1)

    model_version = os.path.getmtime("models/energy_model.pth") if os.path.exists("models/energy_model.pth") else 0
    model, scalers, metrics = load_model_and_scalers(model_version)

    if model is None:
        st.error("No trained model found. Run `python quick_train.py` or `python train.py` first.")
        return

    with st.spinner("Loading regional data and forecasting..."):
        recent_data = load_recent_region_data(region, history_days)
        future_weather = load_region_weather_forecast(region, horizon)
        forecast = predict_next_hours(model, scalers, recent_data, future_weather, region, horizon)
        recommendations = recommend_low_consumption_windows(forecast, duration_hours=duration, top_n=3)

    if forecast.empty:
        st.error("The forecast could not be created. Try increasing the history window or retraining the model.")
        return

    current_consumption = recent_data.sort_values("timestamp")["consumption_mw"].iloc[-1]
    next_prediction = forecast["predicted_consumption_mw"].iloc[0]
    best_window = recommendations.iloc[0] if not recommendations.empty else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Region", region)
    col2.metric("Latest consumption", f"{current_consumption:,.0f} MW")
    col3.metric("Next hour prediction", f"{next_prediction:,.0f} MW", f"{next_prediction - current_consumption:+,.0f} MW")
    if best_window is not None:
        col4.metric("Best start time", best_window["start_time"].strftime("%H:%M"))
    else:
        col4.metric("Best start time", "Unavailable")

    tab_live, tab_recommendation, tab_evaluation, tab_data = st.tabs([
        "1. Live Forecast",
        "2. Recommendation",
        "3. Model Evaluation",
        "4. Forecast Table",
    ])

    with tab_live:
        st.write(
            "This section explains the current prediction. It compares the latest "
            "known consumption with the next-hour prediction, then shows the future "
            "forecast curve."
        )
        display_real_vs_predicted_comparison(current_consumption, next_prediction, recent_data, forecast)
        st.subheader("Future Forecast")
        st.caption(
            "The blue curve is the model forecast for the selected horizon. "
            "The green area is the recommended low-consumption window."
        )
        draw_forecast_chart(forecast, recommendations)

    with tab_recommendation:
        display_business_recommendation(best_window)

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Recommended Low-Consumption Time Slots")
            display = recommendations.copy()
            if not display.empty:
                display["start_time"] = display["start_time"].dt.strftime("%Y-%m-%d %H:%M")
                display["end_time"] = display["end_time"].dt.strftime("%Y-%m-%d %H:%M")
                display["average_consumption_mw"] = display["average_consumption_mw"].round(0)
                display["minimum_consumption_mw"] = display["minimum_consumption_mw"].round(0)
            st.dataframe(display, use_container_width=True, hide_index=True)

        with right:
            st.subheader("How to Explain It")
            st.write(
                "The app tests every possible time window with the selected duration. "
                "The best window is the one with the lowest average predicted consumption."
            )
            st.write(f"Selected horizon: {horizon} hours")
            st.write(f"Selected usage duration: {duration} hours")
            st.write(f"Model input features: {model.input_size}")

    with tab_evaluation:
        display_model_performance(metrics)
        display_prediction_quality_graphs()

    with tab_data:
        st.subheader("Forecast Values")
        table = forecast[["timestamp", "predicted_consumption_mw", "temperature", "humidity"]].copy()
        table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        table["predicted_consumption_mw"] = table["predicted_consumption_mw"].round(0)
        st.dataframe(table, use_container_width=True, hide_index=True)

        # The CSV download lets the user keep the forecast for reporting or analysis.
        csv_data = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download forecast as CSV",
            data=csv_data,
            file_name=f"forecast_{region}.csv",
            mime="text/csv",
        )

    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
