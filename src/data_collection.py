"""
Data Collection Module
======================

This module handles fetching energy consumption data and weather data
from various APIs. It provides both historical data for training and
real-time data for inference.

Data Sources:
- Energy: RTE Eco2Mix (French electricity grid operator)
- Weather: Open-Meteo (free weather API)

Author: Student
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# RTE Eco2Mix API - French electricity consumption data
# Base URL for RTE data
RTE_BASE_URL = "https://opendata.reseaux-energies.fr/api/v2"

# Open-Meteo API - Free weather data (no API key required)
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1"

# France coordinates (Paris as reference for national consumption)
FRANCE_LAT = 48.8566
FRANCE_LON = 2.3522

# Output directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ============================================================================
# PART 1: REAL-TIME ENERGY DATA FETCHING
# ============================================================================

def get_realtime_energy():
    """
    Fetch REAL-TIME energy consumption from RTE Eco2Mix API.
    
    This function calls the live API to get current electricity consumption
    data for France. No local files are used - all data comes from the API.
    
    Returns:
        dict: Contains timestamp, consumption (MW), and source info
    
    WHY DO WE DO THIS?
    - For real-time inference, we need live data
    - RTE provides up-to-the-minute consumption data
    - This is the "live" part of our live pipeline
    
    Example output:
    {
        'timestamp': '2024-01-15T14:30:00',
        'consumption_mw': 65400,
        'source': 'RTE Eco2Mix'
    }
    """
    try:
        # RTE provides real-time data via their API endpoint
        # The endpoint gives consumption data updated every 5-15 minutes
        url = f"{RTE_BASE_URL}/catalog/datasets/consommation-nationale/records"
        
        # Parameters to get the most recent data
        params = {
            "limit": 1,  # Get only the latest record
            "sort": "-date_utc"  # Sort by date descending
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('records') and len(data['records']) > 0:
            record = data['records'][0]['fields']
            return {
                'timestamp': record.get('date_utc'),
                'consumption_mw': record.get('consumption'),
                'source': 'RTE Eco2Mix'
            }
        
        return None
        
    except Exception as e:
        print(f"Error fetching real-time energy data: {e}")
        # Fallback: generate simulated data for demo purposes
        # In production, you would handle this differently
        return simulate_realtime_energy()


def simulate_realtime_energy():
    """
    Generate simulated real-time energy data for demonstration.
    
    This is used when the API is unavailable or for testing purposes.
    It simulates realistic French electricity consumption patterns.
    
    Returns:
        dict: Simulated energy data
    
    WHY DO WE DO THIS?
    - API might be down
    - For testing the pipeline
    - For demonstration when offline
    """
    import random
    
    now = datetime.now()
    hour = now.hour
    
    # Base consumption varies by time of day
    # France typically has ~50-70 GW base consumption
    base = 55000  # MW
    
    # Time-of-day pattern (higher during day, lower at night)
    if 6 <= hour < 10:
        factor = 0.9  # Morning ramp-up
    elif 10 <= hour < 18:
        factor = 1.1  # Daytime peak
    elif 18 <= hour < 22:
        factor = 1.15  # Evening peak
    elif 22 <= hour < 6:
        factor = 0.85  # Night low
    
    # Add some randomness (±5%)
    variation = random.uniform(-0.05, 0.05)
    consumption = base * factor * (1 + variation)
    
    return {
        'timestamp': now.isoformat(),
        'consumption_mw': round(consumption, 2),
        'source': 'Simulated (API unavailable)'
    }


# ============================================================================
# PART 2: HISTORICAL ENERGY DATA FETCHING
# ============================================================================

def fetch_historical_energy(start_date, end_date):
    """
    Fetch HISTORICAL energy consumption data for training.
    
    This downloads historical data from RTE for model training.
    We need past data to train our LSTM model.
    
    Args:
        start_date: Start date for historical data (datetime)
        end_date: End date for historical data (datetime)
    
    Returns:
        pd.DataFrame: Historical energy consumption data
    
    WHY DO WE DO THIS?
    - LSTM needs training data
    - We need to learn patterns from past consumption
    - Historical data teaches the model daily/weekly patterns
    """
    print(f"Fetching historical energy data from {start_date.date()} to {end_date.date()}")
    
    all_records = []
    
    # RTE API has limits, so we fetch in chunks
    current_date = start_date
    while current_date < end_date:
        try:
            # Calculate end of this chunk (max 7 days to avoid API limits)
            chunk_end = min(current_date + timedelta(days=7), end_date)
            
            url = f"{RTE_BASE_URL}/catalog/datasets/consommation-nationale/records"
            params = {
                "limit": 5000,  # Max records per request
                "refine": f"date_utc:[{current_date.isoformat()} TO {chunk_end.isoformat()}]"
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('records'):
                for record in data['records']:
                    fields = record['fields']
                    all_records.append({
                        'timestamp': fields.get('date_utc'),
                        'consumption_mw': fields.get('consumption')
                    })
            
            current_date = chunk_end
            time.sleep(0.5)  # Be nice to the API
            
        except Exception as e:
            print(f"Error fetching chunk starting {current_date.date()}: {e}")
            current_date += timedelta(days=7)
    
    df = pd.DataFrame(all_records)
    
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates()
    
    return df


def generate_synthetic_historical_energy(start_date, end_date):
    """
    Generate synthetic historical energy data when API is unavailable.
    
    This creates realistic-looking energy consumption data for training
    when the RTE API cannot be accessed. It simulates:
    - Daily patterns (higher during day, lower at night)
    - Weekly patterns (lower on weekends)
    - Seasonal patterns (higher in winter/summer)
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
    
    Returns:
        pd.DataFrame: Synthetic hourly energy data
    
    WHY DO WE DO THIS?
    - RTE API might be down or rate-limited
    - For demonstration and testing
    - Ensures the project works even without API access
    """
    import random
    
    print(f"Generating synthetic historical energy data...")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    
    # Generate hourly timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
    
    records = []
    for ts in timestamps:
        hour = ts.hour
        dayofweek = ts.dayofweek
        month = ts.month
        
        # Base consumption
        base = 55000  # MW
        
        # Time of day factor
        if 6 <= hour < 10:
            time_factor = 0.9
        elif 10 <= hour < 18:
            time_factor = 1.1
        elif 18 <= hour < 22:
            time_factor = 1.15
        else:
            time_factor = 0.85
        
        # Weekend factor (lower on weekends)
        weekend_factor = 0.92 if dayofweek >= 5 else 1.0
        
        # Seasonal factor (higher in winter/summer for heating/cooling)
        if month in [12, 1, 2]:
            season_factor = 1.15  # Winter heating
        elif month in [6, 7, 8]:
            season_factor = 1.1   # Summer cooling
        else:
            season_factor = 1.0
        
        # Calculate consumption
        consumption = base * time_factor * weekend_factor * season_factor
        
        # Add random noise (±3%)
        noise = random.uniform(-0.03, 0.03)
        consumption *= (1 + noise)
        
        records.append({
            'timestamp': ts,
            'consumption_mw': round(consumption, 2)
        })
    
    df = pd.DataFrame(records)
    print(f"Generated {len(df)} hourly records")
    
    return df


# ============================================================================
# PART 3: REAL-TIME WEATHER DATA FETCHING
# ============================================================================

def get_realtime_weather(lat=FRANCE_LAT, lon=FRANCE_LON):
    """
    Fetch REAL-TIME weather data from Open-Meteo API.
    
    Open-Meteo is a free weather API that doesn't require an API key.
    We fetch current weather conditions that affect energy consumption.
    
    Args:
        lat: Latitude (default: Paris)
        lon: Longitude (default: Paris)
    
    Returns:
        dict: Current weather data
    
    WHY DO WE DO THIS?
    - Weather strongly affects energy consumption
    - Cold weather = more heating = higher consumption
    - Hot weather = more cooling = higher consumption
    - This data is used for accurate predictions
    
    Example output:
    {
        'timestamp': '2024-01-15T14:00:00',
        'temperature': 8.5,
        'humidity': 82,
        'wind_speed': 15.2,
        'cloud_cover': 45
    }
    """
    try:
        url = f"{OPEN_METEO_BASE_URL}/forecast"
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
            "timezone": "Europe/Paris"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        current = data.get('current', {})
        
        return {
            'timestamp': datetime.now().isoformat(),
            'temperature': current.get('temperature_2m'),
            'humidity': current.get('relative_humidity_2m'),
            'wind_speed': current.get('wind_speed_10m'),
            'cloud_cover': current.get('cloud_cover'),
            'source': 'Open-Meteo'
        }
        
    except Exception as e:
        print(f"Error fetching real-time weather: {e}")
        return simulate_realtime_weather()


def simulate_realtime_weather():
    """
    Generate simulated weather data for demonstration.
    
    Returns realistic weather data when API is unavailable.
    """
    import random
    
    now = datetime.now()
    month = now.month
    
    # Base temperature by month (Paris climate)
    if month in [12, 1, 2]:
        base_temp = 5
    elif month in [3, 4, 5]:
        base_temp = 12
    elif month in [6, 7, 8]:
        base_temp = 20
    else:
        base_temp = 14
    
    # Add time-of-day variation
    hour = now.hour
    if 6 <= hour < 12:
        temp_variation = (hour - 6) * 0.5
    elif 12 <= hour < 20:
        temp_variation = 3 - (hour - 12) * 0.3
    else:
        temp_variation = -2
    
    temperature = base_temp + temp_variation + random.uniform(-2, 2)
    
    return {
        'timestamp': now.isoformat(),
        'temperature': round(temperature, 1),
        'humidity': random.randint(60, 90),
        'wind_speed': round(random.uniform(5, 25), 1),
        'cloud_cover': random.randint(20, 80),
        'source': 'Simulated'
    }


# ============================================================================
# PART 4: HISTORICAL WEATHER DATA FETCHING
# ============================================================================

def fetch_historical_weather(start_date, end_date, lat=FRANCE_LAT, lon=FRANCE_LON):
    """
    Fetch HISTORICAL weather data for training.
    
    We need past weather data to train our model to understand
    how weather affects energy consumption.
    
    Args:
        start_date: Start date (datetime)
        end_date: End date (datetime)
        lat: Latitude
        lon: Longitude
    
    Returns:
        pd.DataFrame: Historical weather data
    
    WHY DO WE DO THIS?
    - Weather features improve prediction accuracy
    - The model learns correlation between weather and consumption
    - Historical weather is needed to match with historical energy data
    """
    print(f"Fetching historical weather data...")
    
    try:
        # Open-Meteo historical API
        url = f"{OPEN_METEO_BASE_URL}/forecast"
        
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,cloud_cover",
            "timezone": "Europe/Paris"
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        hourly = data.get('hourly', {})
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly.get('time')),
            'temperature': hourly.get('temperature_2m'),
            'humidity': hourly.get('relative_humidity_2m'),
            'wind_speed': hourly.get('wind_speed_10m'),
            'cloud_cover': hourly.get('cloud_cover')
        })
        
        return df
        
    except Exception as e:
        print(f"Error fetching historical weather: {e}")
        return generate_synthetic_historical_weather(start_date, end_date)


def generate_synthetic_historical_weather(start_date, end_date):
    """
    Generate synthetic historical weather data.
    
    Creates realistic weather patterns for training when API unavailable.
    """
    import random
    
    print(f"Generating synthetic historical weather data...")
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
    
    records = []
    for ts in timestamps:
        month = ts.month
        hour = ts.hour
        
        # Base temperature by month
        if month in [12, 1, 2]:
            base_temp = 5
        elif month in [3, 4, 5]:
            base_temp = 12
        elif month in [6, 7, 8]:
            base_temp = 20
        else:
            base_temp = 14
        
        # Time of day variation
        if 6 <= hour < 12:
            temp_var = (hour - 6) * 0.5
        elif 12 <= hour < 20:
            temp_var = 3 - (hour - 12) * 0.3
        else:
            temp_var = -2
        
        temperature = base_temp + temp_var + random.uniform(-2, 2)
        
        records.append({
            'timestamp': ts,
            'temperature': round(temperature, 1),
            'humidity': random.randint(60, 90),
            'wind_speed': round(random.uniform(5, 25), 1),
            'cloud_cover': random.randint(20, 80)
        })
    
    df = pd.DataFrame(records)
    print(f"Generated {len(df)} hourly weather records")
    
    return df


# ============================================================================
# PART 5: DATA MERGING
# ============================================================================

def merge_energy_weather(energy_df, weather_df):
    """
    Merge energy and weather datasets on timestamp.
    
    This combines both data sources into a single dataset
    that our model can use for training.
    
    Args:
        energy_df: Energy consumption DataFrame
        weather_df: Weather DataFrame
    
    Returns:
        pd.DataFrame: Merged dataset
    
    WHY DO WE DO THIS?
    - We need both energy AND weather features
    - The model learns how weather affects consumption
    - Merging on timestamp ensures alignment
    """
    # Ensure timestamps are datetime
    energy_df['timestamp'] = pd.to_datetime(energy_df['timestamp'])
    weather_df['timestamp'] = pd.to_datetime(weather_df['timestamp'])
    
    # Merge on timestamp (inner join to keep only matching records)
    merged = pd.merge_asof(
        energy_df.sort_values('timestamp'),
        weather_df.sort_values('timestamp'),
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta('1h')  # Within 1 hour tolerance
    )
    
    # Drop any rows with missing values
    merged = merged.dropna()
    
    print(f"Merged dataset: {len(merged)} records")
    
    return merged


# ============================================================================
# PART 6: MAIN DATA COLLECTION FUNCTION
# ============================================================================

def collect_all_data(start_date=None, end_date=None, use_synthetic=True):
    """
    Main function to collect all data for the project.
    
    This is the entry point for data collection. It fetches both
    energy and weather data and merges them together.
    
    Args:
        start_date: Start date for historical data (default: 30 days ago)
        end_date: End date for historical data (default: today)
        use_synthetic: If True, use synthetic data when API unavailable
    
    Returns:
        pd.DataFrame: Complete merged dataset
    
    WHY DO WE DO THIS?
    - Provides a simple interface for data collection
    - Handles both API and synthetic data
    - Returns a ready-to-use merged dataset
    """
    # Default to last 30 days if not specified
    if end_date is None:
        end_date = datetime.now()
    if start_date is None:
        start_date = end_date - timedelta(days=30)
    
    # Try to fetch real data, fall back to synthetic if needed
    try:
        energy_df = fetch_historical_energy(start_date, end_date)
        if len(energy_df) < 10:  # If too few records
            raise Exception("Insufficient energy data")
    except Exception as e:
        print(f"Could not fetch real energy data: {e}")
        if use_synthetic:
            energy_df = generate_synthetic_historical_energy(start_date, end_date)
        else:
            raise
    
    try:
        weather_df = fetch_historical_weather(start_date, end_date)
        if len(weather_df) < 10:
            raise Exception("Insufficient weather data")
    except Exception as e:
        print(f"Could not fetch real weather data: {e}")
        if use_synthetic:
            weather_df = generate_synthetic_historical_weather(start_date, end_date)
        else:
            raise
    
    # Merge the datasets
    merged_df = merge_energy_weather(energy_df, weather_df)
    
    return merged_df


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Collection Module")
    print("=" * 60)
    
    # Test real-time functions
    print("\n1. Testing real-time energy data...")
    energy = get_realtime_energy()
    print(f"   Result: {energy}")
    
    print("\n2. Testing real-time weather data...")
    weather = get_realtime_weather()
    print(f"   Result: {weather}")
    
    # Test historical data collection
    print("\n3. Testing historical data collection...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    data = collect_all_data(start_date, end_date)
    print(f"   Collected {len(data)} records")
    print(f"   Columns: {list(data.columns)}")
    print(f"\n   Sample data:")
    print(data.head())
    
    print("\n" + "=" * 60)
    print("Data Collection Module Test Complete!")
    print("=" * 60)