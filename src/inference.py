"""
Real-Time Inference Pipeline
=============================

This module implements the live prediction pipeline that:
1. Fetches real-time data from APIs
2. Maintains a sliding window of the last 24 hours
3. Loads the trained model
4. Makes predictions for the next hour

This is the core of the "live" part of the project.

Author: Student
"""

import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import pickle


# ============================================================================
# PART 1: REAL-TIME DATA MANAGER
# ============================================================================

class RealTimeDataManager:
    """
    Manages real-time data fetching and storage.
    
    This class:
    - Fetches data from APIs periodically
    - Maintains a sliding window of recent data
    - Provides data for model input
    
    WHY DO WE NEED THIS?
    ===================
    - For live predictions, we need recent data
    - The model needs the last 24 hours as input
    - We fetch new data and maintain a buffer
    """
    
    def __init__(self, sequence_length=24):
        """
        Initialize the data manager.
        
        Args:
            sequence_length: Number of hours to keep in buffer
        """
        self.sequence_length = sequence_length
        self.data_buffer = []
        self.last_fetch = None
        
        print(f"Initialized RealTimeDataManager with buffer size {sequence_length}")
    
    def fetch_and_update(self):
        """
        Fetch latest data from APIs and update buffer.
        
        This should be called periodically (e.g., every 5 minutes)
        to keep the data fresh.
        
        Returns:
            bool: True if data was fetched successfully
        """
        try:
            # Import data collection functions
            from src.data_collection import get_realtime_energy, get_realtime_weather
            
            # Fetch real-time data
            energy_data = get_realtime_energy()
            weather_data = get_realtime_weather()
            
            if energy_data is None:
                print("Warning: Could not fetch energy data")
                return False
            
            # Create a record
            timestamp = pd.to_datetime(energy_data['timestamp'])
            
            record = {
                'timestamp': timestamp,
                'consumption_mw': energy_data['consumption_mw'],
                'temperature': weather_data.get('temperature', 15),
                'humidity': weather_data.get('humidity', 70),
                'wind_speed': weather_data.get('wind_speed', 10),
                'cloud_cover': weather_data.get('cloud_cover', 50)
            }
            
            # Add to buffer
            self.data_buffer.append(record)
            
            # Keep only the last sequence_length records
            if len(self.data_buffer) > self.sequence_length:
                self.data_buffer = self.data_buffer[-self.sequence_length:]
            
            self.last_fetch = datetime.now()
            
            print(f"Fetched data: consumption={record['consumption_mw']:.0f} MW, "
                  f"temp={record['temperature']:.1f}°C")
            
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def get_latest_sequence(self, feature_cols):
        """
        Get the most recent sequence for prediction.
        
        Args:
            feature_cols: List of feature column names
        
        Returns:
            np.array: Shape (sequence_length, num_features)
        """
        if len(self.data_buffer) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} records, "
                           f"have {len(self.data_buffer)}")
        
        # Convert to DataFrame
        df = pd.DataFrame(self.data_buffer)
        
        # Add time features
        df = self._add_time_features(df)
        
        # Get feature values
        features = df[feature_cols].values
        
        return features
    
    def _add_time_features(self, df):
        """
        Add time-based features to the data.
        
        This must match the preprocessing from training.
        """
        df = df.copy()
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['dayofyear'] = df['timestamp'].dt.dayofyear
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        return df
    
    def get_current_data(self):
        """
        Get the most recent data point.
        
        Returns:
            dict: Current data record
        """
        if not self.data_buffer:
            return None
        return self.data_buffer[-1]
    
    def get_buffer_size(self):
        """
        Get the current buffer size.
        
        Returns:
            int: Number of records in buffer
        """
        return len(self.data_buffer)
    
    def is_ready(self):
        """
        Check if we have enough data for prediction.
        
        Returns:
            bool: True if buffer is full
        """
        return len(self.data_buffer) >= self.sequence_length


# ============================================================================
# PART 2: INFERENCE PIPELINE
# ============================================================================

class InferencePipeline:
    """
    Complete inference pipeline for real-time predictions.
    
    This class:
    - Loads the trained model
    - Manages data fetching
    - Makes predictions
    - Returns results
    
    WHY DO WE NEED THIS?
    ===================
    - Separates inference logic from training
    - Easy to deploy in production
    - Can be used by the Streamlit app
    """
    
    def __init__(self, model_path, scaler_path=None, sequence_length=24):
        """
        Initialize the inference pipeline.
        
        Args:
            model_path: Path to the trained model
            scaler_path: Path to the fitted scaler (optional)
            sequence_length: Length of input sequences
        """
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.eval()  # Set to evaluation mode
        
        # Load scaler
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            print(f"Loading scaler from: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # Initialize data manager
        self.data_manager = RealTimeDataManager(sequence_length)
        
        # Feature columns (must match training)
        self.feature_cols = self._get_feature_cols()
        
        print(f"\nInference Pipeline initialized!")
        print(f"  Device: {self.device}")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Features: {len(self.feature_cols)}")
    
    def _load_model(self, model_path):
        """
        Load the trained model.
        """
        # This project stores local training metadata in the checkpoint.
        # PyTorch 2.6+ defaults to weights_only=True, which rejects that format.
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get model config
        config = checkpoint.get('config', {})
        
        # Create model
        from src.model import EnergyLSTM
        
        model = EnergyLSTM(
            input_size=config.get('input_size', 15),
            hidden_size=config.get('hidden_size', 128),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.2),
            output_size=1
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def _get_feature_cols(self):
        """
        Get the feature columns used during training.
        
        This must match the preprocessing exactly.
        """
        # These are the features we used in training
        # Update this list based on your actual features
        base_features = [
            'consumption_mw', 'temperature', 'humidity', 'wind_speed', 'cloud_cover',
            'hour', 'dayofweek', 'month', 'dayofyear', 'is_weekend',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'dayofweek_sin', 'dayofweek_cos'
        ]
        
        # Add lag features if used in training
        lag_features = [f'lag_{lag}h' for lag in [1, 2, 3, 6, 12, 24]]
        rolling_features = [f'rolling_mean_{window}h' for window in [6, 12, 24]]
        
        all_features = base_features + lag_features + rolling_features
        
        return all_features
    
    def initialize_with_historical(self, historical_df):
        """
        Initialize the buffer with historical data.
        
        This is useful when starting the pipeline - we can
        pre-populate with recent data instead of waiting.
        
        Args:
            historical_df: DataFrame with historical data
        """
        print("Initializing with historical data...")
        
        # Take the last sequence_length records
        recent = historical_df.tail(self.sequence_length).copy()
        
        # Convert to records
        self.data_manager.data_buffer = recent.to_dict('records')
        
        print(f"  Loaded {len(self.data_manager.data_buffer)} historical records")
    
    def predict_next_hour(self):
        """
        Make a prediction for the next hour's consumption.
        
        Returns:
            dict: Prediction results
        """
        # Check if we have enough data
        if not self.data_manager.is_ready():
            return {
                'success': False,
                'error': f'Need {self.sequence_length} hours of data, '
                         f'have {self.data_manager.get_buffer_size()}',
                'prediction': None
            }
        
        try:
            # Get the latest sequence
            sequence = self.data_manager.get_latest_sequence(self.feature_cols)
            
            # Apply scaler if available
            if self.scaler is not None:
                original_shape = sequence.shape
                flat = sequence.reshape(-1, sequence.shape[-1])
                normalized = self.scaler.transform(flat)
                sequence = normalized.reshape(original_shape)
            
            # Convert to tensor
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                prediction = self.model(x)
            
            # Convert to numpy
            pred_value = prediction.cpu().item()
            
            # Get current data for reference
            current = self.data_manager.get_current_data()
            
            return {
                'success': True,
                'prediction': pred_value,
                'current_consumption': current['consumption_mw'],
                'timestamp': current['timestamp'].isoformat() if hasattr(current['timestamp'], 'isoformat') else str(current['timestamp']),
                'temperature': current.get('temperature'),
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None
            }
    
    def run_prediction_cycle(self):
        """
        Run a complete prediction cycle:
        1. Fetch latest data
        2. Make prediction
        3. Return results
        
        Returns:
            dict: Prediction results
        """
        # Fetch new data
        success = self.data_manager.fetch_and_update()
        
        if not success:
            return {
                'success': False,
                'error': 'Failed to fetch data',
                'prediction': None
            }
        
        # Make prediction
        result = self.predict_next_hour()
        
        return result
    
    def get_status(self):
        """
        Get the current status of the pipeline.
        
        Returns:
            dict: Status information
        """
        return {
            'buffer_size': self.data_manager.get_buffer_size(),
            'is_ready': self.data_manager.is_ready(),
            'last_fetch': self.data_manager.last_fetch,
            'device': str(self.device),
            'model_loaded': self.model is not None
        }


# ============================================================================
# PART 3: BATCH PREDICTION (FOR TESTING)
# ============================================================================

def predict_batch(model, data_loader, device):
    """
    Make predictions on a batch of data.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader with input data
        device: Device to run on
    
    Returns:
        np.array: Predictions
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().cpu().numpy())
    
    return np.array(predictions)


# ============================================================================
# PART 4: MAIN INFERENCE FUNCTION
# ============================================================================

def run_inference(model_path, num_predictions=5):
    """
    Run inference multiple times (for testing/demo).
    
    Args:
        model_path: Path to trained model
        num_predictions: Number of predictions to make
    
    Returns:
        list: List of prediction results
    """
    print("=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)
    
    # Create pipeline
    pipeline = InferencePipeline(model_path)
    
    # Initialize with some data
    print("\nInitializing data buffer...")
    from src.data_collection import collect_all_data
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)
    
    historical = collect_all_data(start_date, end_date)
    pipeline.initialize_with_historical(historical)
    
    # Run predictions
    results = []
    for i in range(num_predictions):
        print(f"\nPrediction {i+1}/{num_predictions}")
        
        result = pipeline.run_prediction_cycle()
        
        if result['success']:
            print(f"  Current: {result['current_consumption']:.0f} MW")
            print(f"  Predicted next hour: {result['prediction']:.0f} MW")
            print(f"  Temperature: {result['temperature']}°C")
        else:
            print(f"  Error: {result['error']}")
        
        results.append(result)
    
    return results


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Inference Pipeline")
    print("=" * 60)
    
    # Test data manager
    print("\n1. Testing RealTimeDataManager...")
    dm = RealTimeDataManager(sequence_length=24)
    
    # Simulate adding data
    for i in range(30):
        record = {
            'timestamp': datetime.now() - timedelta(hours=30-i),
            'consumption_mw': 55000 + np.random.randn() * 1000,
            'temperature': 15 + np.random.randn() * 3,
            'humidity': 70 + np.random.randn() * 10,
            'wind_speed': 10 + np.random.randn() * 3,
            'cloud_cover': 50 + np.random.randn() * 20
        }
        dm.data_buffer.append(record)
    
    print(f"  Buffer size: {dm.get_buffer_size()}")
    print(f"  Is ready: {dm.is_ready()}")
    
    # Test feature extraction
    print("\n2. Testing feature extraction...")
    df = pd.DataFrame(dm.data_buffer)
    df = dm._add_time_features(df)
    print(f"  Features: {list(df.columns)}")
    
    print("\n" + "=" * 60)
    print("Inference Pipeline Test Complete!")
    print("=" * 60)
