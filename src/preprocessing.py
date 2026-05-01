"""
Data Preprocessing Module
==========================

This module handles:
1. Data cleaning and validation
2. Feature engineering (time-based features)
3. Data normalization
4. Train/test splitting

Author: Student
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime

REGION_CODES = {
    "Auvergne-Rhone-Alpes": 0,
    "Bourgogne-Franche-Comte": 1,
    "Bretagne": 2,
    "Centre-Val de Loire": 3,
    "Grand Est": 4,
    "Hauts-de-France": 5,
    "Ile-de-France": 6,
    "Normandie": 7,
    "Nouvelle-Aquitaine": 8,
    "Occitanie": 9,
    "Pays de la Loire": 10,
    "Provence-Alpes-Cote d'Azur": 11,
}


# ============================================================================
# PART 1: DATA CLEANING
# ============================================================================

def clean_data(df):
    """
    Clean and validate the input data.
    
    This function:
    - Removes duplicate timestamps
    - Handles missing values
    - Removes outliers
    - Validates data ranges
    
    Args:
        df: Raw DataFrame with energy and weather data
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    
    WHY DO WE DO THIS?
    - Raw data often has issues (duplicates, missing values)
    - ML models need clean, consistent data
    - This ensures data quality before processing
    """
    print("Cleaning data...")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Remove duplicates based on timestamp, and region when available.
    initial_len = len(df)
    duplicate_cols = ['timestamp']
    if 'region' in df.columns:
        duplicate_cols.append('region')
    df = df.drop_duplicates(subset=duplicate_cols, keep='first')
    if len(df) < initial_len:
        print(f"  Removed {initial_len - len(df)} duplicate records")
    
    sort_cols = ['timestamp']
    if 'region' in df.columns:
        sort_cols = ['region', 'timestamp']
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
    # Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.dropna()
    if missing_before > 0:
        print(f"  Removed {missing_before} rows with missing values")
    
    # Validate consumption values. Regional values are lower than national ones.
    if 'consumption_mw' in df.columns:
        if 'region' in df.columns:
            valid_mask = (df['consumption_mw'] > 100) & (df['consumption_mw'] < 30000)
        else:
            valid_mask = (df['consumption_mw'] > 30000) & (df['consumption_mw'] < 90000)
        invalid_count = (~valid_mask).sum()
        if invalid_count > 0:
            print(f"  Warning: {invalid_count} rows with unusual consumption values")
            df = df[valid_mask]
    
    print(f"  Cleaned data: {len(df)} records remaining")
    
    return df


def add_region_features(df):
    """Convert the region name into a simple numeric feature."""
    df = df.copy()
    if 'region' in df.columns:
        df['region_code'] = df['region'].map(REGION_CODES).fillna(6).astype(int)
    return df


# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================

def add_time_features(df):
    """
    Add time-based features to the dataset.
    
    This creates features that capture temporal patterns in energy consumption:
    - Hour of day (0-23)
    - Day of week (0-6)
    - Month (1-12)
    - Is weekend flag
    - Day of year (for seasonal patterns)
    
    These features help the model learn:
    - Daily patterns (higher during business hours)
    - Weekly patterns (lower on weekends)
    - Seasonal patterns (higher in winter/summer)
    
    Args:
        df: DataFrame with timestamp column
    
    Returns:
        pd.DataFrame: DataFrame with new time features
    
    WHY DO WE DO THIS?
    - Energy consumption has strong time patterns
    - The model can learn these patterns from time features
    - This is domain knowledge encoded as features
    """
    print("Adding time-based features...")
    
    df = df.copy()
    
    # Extract time components from timestamp
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['dayofyear'] = df['timestamp'].dt.dayofyear
    
    # Weekend flag (Saturday=5, Sunday=6)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Cyclical encoding for hour (optional but recommended)
    # This captures that 23:00 is close to 00:00
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Cyclical encoding for day of week
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    print(f"  Added features: hour, dayofweek, month, is_weekend, cyclical encodings")
    
    return df


def add_lag_features(df, target_col='consumption_mw', lags=[1, 2, 3, 6, 12, 24]):
    """
    Add lag features (previous values) to the dataset.
    
    Lag features capture the temporal dependency in energy consumption.
    For example, today's consumption depends on yesterday's values.
    
    Args:
        df: DataFrame with target column
        target_col: Name of the target column
        lags: List of lag periods (in hours)
    
    Returns:
        pd.DataFrame: DataFrame with lag features
    
    WHY DO WE DO THIS?
    - Energy consumption is autocorrelated
    - Past values help predict future values
    - LSTM will use these to learn temporal patterns
    
    Example:
    - lag_1h: consumption from 1 hour ago
    - lag_24h: consumption from 24 hours ago (same hour yesterday)
    """
    print(f"Adding lag features for {target_col}...")
    
    df = df.copy()
    groups = [df]
    if 'region' in df.columns:
        groups = [group for _, group in df.groupby('region', sort=False)]

    frames = []
    for group in groups:
        group = group.copy()
        for lag in lags:
            col_name = f'lag_{lag}h'
            group[col_name] = group[target_col].shift(lag)
            print(f"  Added: {col_name}")
        
        # Rolling statistics (moving averages)
        for window in [6, 12, 24]:
            col_name = f'rolling_mean_{window}h'
            group[col_name] = group[target_col].shift(1).rolling(window=window).mean()
            print(f"  Added: {col_name}")
        frames.append(group)
    
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# PART 3: DATA NORMALIZATION
# ============================================================================

class DataNormalizer:
    """
    A class to handle data normalization and inverse transformation.
    
    This is important because:
    - Neural networks work better with normalized data
    - We need to remember the scaler to inverse-transform predictions
    - This class saves the scaler parameters for later use
    """
    
    def __init__(self, scaler_type='standard'):
        """
        Initialize the normalizer.
        
        Args:
            scaler_type: 'standard' (z-score) or 'minmax' (0-1 range)
        
        WHY DO WE DO THIS?
        - Different scalers work better for different data
        - StandardScaler: good for data that looks like a bell curve
        - MinMaxScaler: good for bounded data
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.fitted = False
    
    def fit(self, data):
        """
        Fit the scaler to the data.
        
        This learns the statistics (mean, std, min, max) from training data.
        IMPORTANT: Only fit on training data, not all data!
        
        Args:
            data: numpy array or DataFrame to fit on
        """
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.scaler.fit(data)
        self.fitted = True
        
        print(f"Fitted {self.scaler_type} scaler")
        if self.scaler_type == 'standard':
            print(f"  Mean: {self.scaler.mean_[:3]}... (showing first 3)")
            print(f"  Std: {self.scaler.scale_[:3]}...")
        else:
            print(f"  Min: {self.scaler.data_min_[:3]}...")
            print(f"  Max: {self.scaler.data_max_[:3]}...")
    
    def transform(self, data):
        """
        Transform data using the fitted scaler.
        
        Args:
            data: Data to transform
        
        Returns:
            Transformed data
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted yet! Call fit() first.")
        return self.scaler.transform(data)
    
    def inverse_transform(self, data):
        """
        Inverse transform (denormalize) data.
        
        This is used to convert predictions back to original scale.
        
        Args:
            data: Normalized data to transform back
        
        Returns:
            Data in original scale
        """
        if not self.fitted:
            raise ValueError("Scaler not fitted yet!")
        return self.scaler.inverse_transform(data)
    
    def fit_transform(self, data):
        """
        Fit and transform in one step.
        
        Args:
            data: Training data
        
        Returns:
            Transformed data
        """
        self.fit(data)
        return self.transform(data)


# ============================================================================
# PART 4: TRAIN/TEST SPLITTING
# ============================================================================

def split_data(df, train_ratio=0.8):
    """
    Split data into training and testing sets.
    
    We use time-based splitting (not random) because:
    - Time series data shouldn't be shuffled
    - We train on past data, predict future data
    - This mimics real-world deployment
    
    Args:
        df: DataFrame to split
        train_ratio: Ratio of data for training (0.0 to 1.0)
    
    Returns:
        tuple: (train_df, test_df)
    
    WHY DO WE DO THIS?
    - We need separate data for training and evaluation
    - Time-based split is appropriate for time series
    - Typically use 80% train, 20% test
    """
    print(f"Splitting data with {train_ratio*100}% train / {(1-train_ratio)*100}% test...")
    
    if 'region' in df.columns:
        train_parts = []
        test_parts = []
        for _, group in df.groupby('region', sort=False):
            split_idx = int(len(group) * train_ratio)
            train_parts.append(group.iloc[:split_idx].copy())
            test_parts.append(group.iloc[split_idx:].copy())
        train_df = pd.concat(train_parts, ignore_index=True)
        test_df = pd.concat(test_parts, ignore_index=True)
    else:
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    
    print(f"  Training set: {len(train_df)} records")
    print(f"  Test set: {len(test_df)} records")
    print(f"  Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"  Test period: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    return train_df, test_df


# ============================================================================
# PART 5: MAIN PREPROCESSING FUNCTION
# ============================================================================

def preprocess_data(df, add_lags=True):
    """
    Main preprocessing pipeline.
    
    This is the entry point that applies all preprocessing steps:
    1. Clean data
    2. Add time features
    3. Add lag features (optional)
    4. Handle any remaining NaN values
    
    Args:
        df: Raw DataFrame from data collection
        add_lags: Whether to add lag features
    
    Returns:
        pd.DataFrame: Fully preprocessed DataFrame
    
    WHY DO WE DO THIS?
    - Provides a simple interface for preprocessing
    - Ensures consistent preprocessing for train and test
    """
    print("=" * 60)
    print("Starting Data Preprocessing")
    print("=" * 60)
    
    # Step 1: Clean data
    df = clean_data(df)
    
    # Step 2: Add time features
    df = add_time_features(df)

    # Optional region code for regional models
    df = add_region_features(df)
    
    # Step 3: Add lag features
    if add_lags:
        df = add_lag_features(df)
    
    # Step 4: Handle NaN from lag features
    # (First few rows won't have lag values)
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"Removing {nan_count} NaN values from lag features...")
        df = df.dropna()
    
    print("=" * 60)
    print("Preprocessing Complete!")
    print(f"Final dataset: {len(df)} records, {len(df.columns)} features")
    print("=" * 60)
    
    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_feature_columns(df):
    """
    Get list of feature columns for model training.
    
    Args:
        df: Preprocessed DataFrame
    
    Returns:
        list: Column names to use as features
    """
    # Exclude timestamp and target
    exclude = ['timestamp', 'consumption_mw', 'region']
    features = [col for col in df.columns if col not in exclude]
    
    return features


def get_target_column():
    """
    Get the target column name.
    
    Returns:
        str: Name of the target column
    """
    return 'consumption_mw'


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Preprocessing Module")
    print("=" * 60)
    
    # Create sample data
    from datetime import timedelta
    
    print("\n1. Creating sample data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
    
    sample_data = pd.DataFrame({
        'timestamp': timestamps,
        'consumption_mw': np.random.uniform(45000, 75000, len(timestamps)),
        'temperature': np.random.uniform(5, 25, len(timestamps)),
        'humidity': np.random.uniform(50, 90, len(timestamps)),
        'wind_speed': np.random.uniform(5, 20, len(timestamps)),
        'cloud_cover': np.random.uniform(0, 100, len(timestamps))
    })
    
    print(f"   Created {len(sample_data)} sample records")
    
    # Test preprocessing
    print("\n2. Testing preprocessing pipeline...")
    processed = preprocess_data(sample_data)
    
    print(f"\n   Processed data shape: {processed.shape}")
    print(f"   Features: {list(processed.columns)}")
    
    # Test normalizer
    print("\n3. Testing DataNormalizer...")
    normalizer = DataNormalizer(scaler_type='standard')
    
    target_data = processed[['consumption_mw']].values
    normalized = normalizer.fit_transform(target_data)
    
    print(f"   Original mean: {target_data.mean():.2f}")
    print(f"   Normalized mean: {normalized.mean():.4f}")
    
    # Test inverse transform
    restored = normalizer.inverse_transform(normalized)
    print(f"   Restored mean: {restored.mean():.2f}")
    
    # Test train/test split
    print("\n4. Testing train/test split...")
    train_df, test_df = split_data(processed, train_ratio=0.8)
    print(f"   Train: {len(train_df)}, Test: {len(test_df)}")
    
    print("\n" + "=" * 60)
    print("Data Preprocessing Module Test Complete!")
    print("=" * 60)
