"""
Quick Training Test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_collection import collect_regional_dataset
from src.preprocessing import preprocess_data, split_data, DataNormalizer, get_feature_columns, get_target_column
from src.sequence_loader import create_sequences_from_df, create_data_loaders
from src.model import create_model
from src.training import TrainingConfig, train_model
from datetime import datetime, timedelta

# Quick training test
print('Quick training test...')

# Collect several months of real RTE data for a few regions.
# RTE consolidated regional data is historical, so we use a known available period.
end_date = datetime(2026, 1, 31)
start_date = end_date - timedelta(days=90)
regions = ["Ile-de-France", "Auvergne-Rhone-Alpes", "Occitanie"]
df = collect_regional_dataset(regions, start_date, end_date, use_synthetic=True)
print(f'Collected {len(df)} records')
df.to_csv('data/raw_data.csv', index=False)

# Preprocess
df_processed = preprocess_data(df, add_lags=True)
print(f'Processed {len(df_processed)} records')
df_processed.to_csv('data/processed_data.csv', index=False)

# Split
train_df, test_df = split_data(df_processed, train_ratio=0.8)

# Get features
feature_cols = get_feature_columns(df_processed)
target_col = get_target_column()
print(f'Features: {len(feature_cols)}')

# Create sequences
X_train, y_train = create_sequences_from_df(train_df, feature_cols, target_col, 24)
X_test, y_test = create_sequences_from_df(test_df, feature_cols, target_col, 24)
print(f'Sequences: train={X_train.shape}, test={X_test.shape}')

# Normalize
normalizer = DataNormalizer(scaler_type='standard')
n_samples = X_train.shape[0]
n_features = X_train.shape[2]
X_train_flat = X_train.reshape(-1, n_features)
normalizer.fit(X_train_flat)
X_train_flat_norm = normalizer.transform(X_train_flat)
X_train_norm = X_train_flat_norm.reshape(n_samples, 24, n_features)

n_samples_test = X_test.shape[0]
X_test_flat = X_test.reshape(-1, n_features)
X_test_flat_norm = normalizer.transform(X_test_flat)
X_test_norm = X_test_flat_norm.reshape(n_samples_test, 24, n_features)

target_normalizer = DataNormalizer(scaler_type='standard')
y_train_norm = target_normalizer.fit_transform(y_train.reshape(-1, 1)).squeeze()
y_test_norm = target_normalizer.transform(y_test.reshape(-1, 1)).squeeze()

# Data loaders
train_loader, test_loader = create_data_loaders(X_train_norm, y_train_norm, X_test_norm, y_test_norm, 32)

# Model
model = create_model(input_size=len(feature_cols), hidden_size=64, num_layers=1, dropout=0.1)
print(f'Model created with {model.get_num_parameters()} parameters')

# Train
config = TrainingConfig()
config.input_size = len(feature_cols)
config.hidden_size = 64
config.num_layers = 1
config.dropout = 0.1
config.num_epochs = 20
config.batch_size = 32
config.patience = 5
config.min_delta = 0.0001

model, history, metrics = train_model(
    model,
    train_loader,
    test_loader,
    config=config,
    save_path='models/energy_model.pth',
    target_scaler=target_normalizer,
    predictions_path='data/evaluation_predictions.csv'
)

import pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump({'feature_scaler': normalizer, 'target_scaler': target_normalizer}, f)

print(f'Final metrics: MAE={metrics["MAE"]:.2f}, RMSE={metrics["RMSE"]:.2f}')
print('Training complete!')
