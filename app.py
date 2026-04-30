"""
Streamlit Live Dashboard
========================

This is the main Streamlit application that provides a real-time
dashboard for energy consumption forecasting.

Features:
- Real-time energy consumption display
- Weather data display
- Next hour prediction
- Auto-refresh every X seconds
- Historical + predicted consumption graph
- Clean, professional UI

Author: Student
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5
    }
    .stMetric {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
    .metric-delta {
        font-size: 14px;
    }
    h1 {
        color: #1e3a5f;
    }
    h2 {
        color: #2c5282;
    }
    .stAlert {
        background-color: #fff;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA FETCHING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_current_energy():
    """Fetch current energy data."""
    from src.data_collection import get_realtime_energy, simulate_realtime_energy
    
    try:
        data = get_realtime_energy()
        if data is None:
            raise Exception("No data")
        return data
    except:
        return simulate_realtime_energy()


@st.cache_data(ttl=300)
def fetch_current_weather():
    """Fetch current weather data."""
    from src.data_collection import get_realtime_weather, simulate_realtime_weather
    
    try:
        data = get_realtime_weather()
        if data is None:
            raise Exception("No data")
        return data
    except:
        return simulate_realtime_weather()


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_historical_data(days=2):
    """Fetch historical data for the graph."""
    from src.data_collection import collect_all_data
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        data = collect_all_data(start_date, end_date)
        return data
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        # Generate synthetic data
        return generate_synthetic_history(days)


def generate_synthetic_history(days):
    """Generate synthetic historical data."""
    timestamps = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
    
    records = []
    for ts in timestamps:
        hour = ts.hour
        dayofweek = ts.dayofweek
        
        # Base consumption with patterns
        base = 55000
        if 6 <= hour < 10:
            factor = 0.9
        elif 10 <= hour < 18:
            factor = 1.1
        elif 18 <= hour < 22:
            factor = 1.15
        else:
            factor = 0.85
        
        weekend_factor = 0.92 if dayofweek >= 5 else 1.0
        
        consumption = base * factor * weekend_factor * np.random.uniform(0.97, 1.03)
        
        records.append({
            'timestamp': ts,
            'consumption_mw': consumption,
            'temperature': 15 + np.random.randn() * 5,
            'humidity': 70 + np.random.randn() * 10,
            'wind_speed': 10 + np.random.randn() * 5,
            'cloud_cover': 50 + np.random.randn() * 20
        })
    
    return pd.DataFrame(records)


# ============================================================================
# MODEL AND PREDICTION
# ============================================================================

def load_model_for_inference():
    """Load the trained model."""
    import torch
    from src.model import EnergyLSTM
    
    model_path = "models/energy_model.pth"
    
    if not os.path.exists(model_path):
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint.get('config', {})
        
        model = EnergyLSTM(
            input_size=config.get('input_size', 15),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2),
            output_size=1
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def make_prediction(model, recent_data):
    """Make a prediction using the model."""
    import torch
    
    if model is None or len(recent_data) < 24:
        return None
    
    try:
        # Get feature columns
        feature_cols = [
            'consumption_mw', 'temperature', 'humidity', 'wind_speed', 'cloud_cover',
            'hour', 'dayofweek', 'month', 'dayofyear', 'is_weekend',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'dayofweek_sin', 'dayofweek_cos'
        ]
        
        # Add lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            feature_cols.append(f'lag_{lag}h')
        
        # Add rolling features
        for window in [6, 12, 24]:
            feature_cols.append(f'rolling_mean_{window}h')
        
        # Get last 24 hours
        df = recent_data.tail(24).copy()
        
        # Add time features
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['dayofyear'] = df['timestamp'].dt.dayofyear
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Add lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'lag_{lag}h'] = df['consumption_mw'].shift(lag)
        
        # Add rolling features
        for window in [6, 12, 24]:
            df[f'rolling_mean_{window}h'] = df['consumption_mw'].shift(1).rolling(window).mean()
        
        # Drop NaN rows
        df = df.dropna()
        
        if len(df) < 1:
            return None
        
        # Get features
        features = df[feature_cols].values
        
        # Convert to tensor
        x = torch.FloatTensor(features).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            prediction = model(x)
        
        return prediction.item()
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None


# ============================================================================
# UI COMPONENTS
# ============================================================================

def display_metrics(current_energy, current_weather, prediction):
    """Display the main metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="⚡ Current Consumption",
            value=f"{current_energy['consumption_mw']:,.0f} MW",
            delta=None
        )
    
    with col2:
        temp = current_weather.get('temperature', 0)
        st.metric(
            label="🌡️ Temperature",
            value=f"{temp:.1f}°C",
            delta=None
        )
    
    with col3:
        humidity = current_weather.get('humidity', 0)
        st.metric(
            label="💧 Humidity",
            value=f"{humidity:.0f}%",
            delta=None
        )
    
    with col4:
        if prediction is not None:
            delta = prediction - current_energy['consumption_mw']
            st.metric(
                label="🔮 Next Hour Prediction",
                value=f"{prediction:,.0f} MW",
                delta=f"{delta:+,.0f} MW"
            )
        else:
            st.metric(
                label="🔮 Next Hour Prediction",
                value="-- MW",
                delta="Model not loaded"
            )


def display_consumption_graph(historical_data, prediction=None):
    """Display the consumption graph."""
    st.subheader("📊 Energy Consumption Over Time")
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Plot historical data
    ax.plot(historical_data['timestamp'], 
            historical_data['consumption_mw'],
            label='Actual Consumption',
            color='#2c5282',
            linewidth=1.5)
    
    # Plot prediction if available
    if prediction is not None:
        last_time = historical_data['timestamp'].iloc[-1]
        next_time = last_time + timedelta(hours=1)
        
        ax.scatter([next_time], [prediction], 
                  color='red', s=100, marker='*', 
                  label='Predicted Next Hour', zorder=5)
        
        # Add annotation
        ax.annotate(f'{prediction:,.0f} MW',
                   xy=(next_time, prediction),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, color='red')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Consumption (MW)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    fig.autofmt_xdate()
    
    st.pyplot(fig)


def display_weather_details(weather_data):
    """Display detailed weather information."""
    st.subheader("🌤️ Weather Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Wind Speed**: {weather_data.get('wind_speed', 0):.1f} km/h")
    
    with col2:
        st.info(f"**Cloud Cover**: {weather_data.get('cloud_cover', 0):.0f}%")
    
    with col3:
        source = weather_data.get('source', 'Unknown')
        st.info(f"**Data Source**: {source}")


def display_system_status():
    """Display system status information."""
    st.subheader("📋 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ Data Collection: Active")
    
    with col2:
        model_exists = os.path.exists("models/energy_model.pth")
        if model_exists:
            st.success("✅ Model: Loaded")
        else:
            st.warning("⚠️ Model: Not trained")
    
    with col3:
        st.success("✅ Dashboard: Running")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main Streamlit application."""
    
    # Title
    st.title("⚡ Real-Time Energy Consumption Forecasting")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    refresh_interval = st.sidebar.slider(
        "Auto-refresh interval (seconds)",
        min_value=10,
        max_value=300,
        value=60,
        step=10
    )
    
    show_history_days = st.sidebar.slider(
        "History display (days)",
        min_value=1,
        max_value=7,
        value=2
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This dashboard shows real-time energy consumption "
        "and predictions for France using an LSTM deep learning model."
    )
    
    # Fetch data
    with st.spinner("Fetching current data..."):
        current_energy = fetch_current_energy()
        current_weather = fetch_current_weather()
        historical_data = fetch_historical_data(days=show_history_days)
    
    # Load model
    model = load_model_for_inference()
    
    # Make prediction
    prediction = None
    if model is not None and len(historical_data) >= 24:
        prediction = make_prediction(model, historical_data)
    
    # Display metrics
    display_metrics(current_energy, current_weather, prediction)
    
    st.markdown("---")
    
    # Display graphs and details
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        display_consumption_graph(historical_data, prediction)
    
    with col_right:
        display_weather_details(current_weather)
    
    st.markdown("---")
    
    # Display status
    display_system_status()
    
    # Auto-refresh
    st.markdown(f"""
    <meta http-equiv="refresh" content="{refresh_interval}">
    """, unsafe_allow_html=True)
    
    # Last update time
    st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()