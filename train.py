"""
Main Training Script
=====================

This script orchestrates the complete training pipeline:
1. Collect data
2. Preprocess data
3. Create sequences
4. Train model
5. Save model

Run this script to train the model:
    python train.py

Author: Student
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

# Import our modules
from src.data_collection import collect_all_data
from src.preprocessing import preprocess_data, split_data, DataNormalizer, get_feature_columns, get_target_column
from src.sequence_loader import create_sequences_from_df, create_data_loaders
from src.model import create_model, print_model_summary
from src.training import TrainingConfig, train_model


# ============================================================================
# CONFIGURATION
# ============================================================================

class ProjectConfig:
    """Project configuration."""
    
    # Data collection
    data_days = 30  # Days of historical data to collect
    
    # Sequence parameters
    sequence_length = 24  # Use last 24 hours to predict next hour
    
    # Model parameters
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    
    # Training parameters
    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001
    train_ratio = 0.8
    
    # Output
    model_dir = "models"
    data_dir = "data"


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    
    print("=" * 70)
    print("ENERGY CONSUMPTION FORECASTING - MODEL TRAINING")
    print("=" * 70)
    
    # Create config
    config = ProjectConfig()
    
    # Create output directories
    os.makedirs(config.model_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    
    # =========================================================================
    # STEP 1: DATA COLLECTION
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA COLLECTION")
    print("=" * 70)
    
    print(f"\nCollecting {config.data_days} days of historical data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=config.data_days)
    
    # Collect data (uses API or generates synthetic if unavailable)
    df = collect_all_data(start_date, end_date, use_synthetic=True)
    
    print(f"\nCollected {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Save raw data
    raw_data_path = os.path.join(config.data_dir, "raw_data.csv")
    df.to_csv(raw_data_path, index=False)
    print(f"Saved raw data to: {raw_data_path}")
    
    # =========================================================================
    # STEP 2: DATA PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: DATA PREPROCESSING")
    print("=" * 70)
    
    # Preprocess data
    df_processed = preprocess_data(df, add_lags=True)
    
    print(f"\nProcessed data shape: {df_processed.shape}")
    print(f"Features: {list(df_processed.columns)}")
    
    # Save processed data
    processed_data_path = os.path.join(config.data_dir, "processed_data.csv")
    df_processed.to_csv(processed_data_path, index=False)
    print(f"Saved processed data to: {processed_data_path}")
    
    # =========================================================================
    # STEP 3: TRAIN/TEST SPLIT
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN/TEST SPLIT")
    print("=" * 70)
    
    # Split data
    train_df, test_df = split_data(df_processed, train_ratio=config.train_ratio)
    
    print(f"\nTraining set: {len(train_df)} records")
    print(f"Test set: {len(test_df)} records")
    
    # =========================================================================
    # STEP 4: CREATE SEQUENCES
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: CREATE SEQUENCES FOR LSTM")
    print("=" * 70)
    
    # Get feature and target columns
    feature_cols = get_feature_columns(df_processed)
    target_col = get_target_column()
    
    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols[:5]):
        print(f"  {i+1}. {col}")
    print(f"  ... and {len(feature_cols)-5} more")
    
    print(f"\nTarget column: {target_col}")
    
    # Create sequences for training set
    X_train, y_train = create_sequences_from_df(
        train_df, feature_cols, target_col, 
        sequence_length=config.sequence_length
    )
    
    # Create sequences for test set
    X_test, y_test = create_sequences_from_df(
        test_df, feature_cols, target_col,
        sequence_length=config.sequence_length
    )
    
    print(f"\nTraining sequences: {X_train.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    # =========================================================================
    # STEP 5: NORMALIZE DATA
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: DATA NORMALIZATION")
    print("=" * 70)
    
    # Fit normalizer on training data only
    normalizer = DataNormalizer(scaler_type='standard')
    
    # Reshape for scaler (flatten sequence dimension)
    n_samples_train = X_train.shape[0]
    n_features = X_train.shape[2]
    X_train_flat = X_train.reshape(-1, n_features)
    
    normalizer.fit(X_train_flat)
    
    # Transform training data
    X_train_flat_norm = normalizer.transform(X_train_flat)
    X_train_norm = X_train_flat_norm.reshape(n_samples_train, config.sequence_length, n_features)
    
    # Transform test data
    n_samples_test = X_test.shape[0]
    X_test_flat = X_test.reshape(-1, n_features)
    X_test_flat_norm = normalizer.transform(X_test_flat)
    X_test_norm = X_test_flat_norm.reshape(n_samples_test, config.sequence_length, n_features)
    
    # Normalize targets
    target_normalizer = DataNormalizer(scaler_type='standard')
    y_train_norm = target_normalizer.fit_transform(y_train.reshape(-1, 1)).squeeze()
    y_test_norm = target_normalizer.transform(y_test.reshape(-1, 1)).squeeze()
    
    print(f"Normalized training data: {X_train_norm.shape}")
    print(f"Normalized test data: {X_test_norm.shape}")
    
    # Save scalers
    import pickle
    scaler_path = os.path.join(config.model_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump({'feature_scaler': normalizer, 'target_scaler': target_normalizer}, f)
    print(f"Saved scalers to: {scaler_path}")
    
    # =========================================================================
    # STEP 6: CREATE DATA LOADERS
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: CREATE DATA LOADERS")
    print("=" * 70)
    
    train_loader, test_loader = create_data_loaders(
        X_train_norm, y_train_norm,
        X_test_norm, y_test_norm,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # =========================================================================
    # STEP 7: CREATE MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: CREATE LSTM MODEL")
    print("=" * 70)
    
    input_size = len(feature_cols)
    
    model = create_model(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout
    )
    
    print_model_summary(model, input_size, config.sequence_length)
    
    # =========================================================================
    # STEP 8: TRAIN MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: TRAIN MODEL")
    print("=" * 70)
    
    # Create training config
    training_config = TrainingConfig()
    training_config.input_size = input_size
    training_config.hidden_size = config.hidden_size
    training_config.num_layers = config.num_layers
    training_config.dropout = config.dropout
    training_config.batch_size = config.batch_size
    training_config.num_epochs = config.num_epochs
    training_config.learning_rate = config.learning_rate
    training_config.patience = 10
    
    # Train
    model, history, metrics = train_model(
        model,
        train_loader,
        test_loader,
        config=training_config,
        save_path=os.path.join(config.model_dir, "energy_model.pth")
    )
    
    # =========================================================================
    # STEP 9: FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    print("\n📊 Final Metrics:")
    print(f"  MAE: {metrics['MAE']:.2f} MW")
    print(f"  RMSE: {metrics['RMSE']:.2f} MW")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²: {metrics['R2']:.4f}")
    
    print("\n📁 Output files:")
    print(f"  Model: {config.model_dir}/energy_model.pth")
    print(f"  Scalers: {config.model_dir}/scaler.pkl")
    print(f"  Raw data: {config.data_dir}/raw_data.csv")
    print(f"  Processed data: {config.data_dir}/processed_data.csv")
    
    print("\n🚀 Next steps:")
    print("  1. Run the Streamlit dashboard: streamlit run app.py")
    print("  2. The model will make real-time predictions")
    print("  3. Dashboard auto-refreshes every 60 seconds")
    
    print("\n" + "=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()