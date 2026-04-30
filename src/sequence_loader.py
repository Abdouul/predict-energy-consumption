"""
Sequence Data Loader for LSTM
==============================

This module creates sequences of data for training the LSTM model.

For LSTM, we need to transform our tabular data into sequences:
- Input: past 24 hours of data (24 timesteps)
- Output: next 1 hour's consumption (1 prediction)

Author: Student
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


# ============================================================================
# PART 1: LSTM SEQUENCE CREATION
# ============================================================================

class EnergyDataset(Dataset):
    """
    PyTorch Dataset for energy consumption forecasting.
    
    This converts our DataFrame into PyTorch tensors that the LSTM
    can process in batches.
    
    WHAT IS A DATASET?
    - A Dataset is PyTorch's way of organizing data for training
    - It provides a way to access individual samples by index
    - The DataLoader will use this to create batches
    
    WHY DO WE NEED THIS?
    - PyTorch requires data in this format
    - Handles batching, shuffling, etc.
    - Efficient memory usage with lazy loading
    """
    
    def __init__(self, features, targets, sequence_length=24):
        """
        Initialize the dataset.
        
        Args:
            features: numpy array of feature data (samples, timesteps, features)
            targets: numpy array of target values (samples,)
            sequence_length: Number of timesteps in each input sequence
        
        WHY DO WE DO THIS?
        - Store data as numpy arrays for efficient access
        - sequence_length defines how many past hours we look at
        """
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        print(f"Created EnergyDataset with {len(features)} samples")
        print(f"  Sequence length: {sequence_length}")
        print(f"  Feature shape: {features.shape if len(features) > 0 else 'N/A'}")
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        This is required by PyTorch.
        """
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Index of the sample
        
        Returns:
            tuple: (input_sequence, target_value)
        
        WHY DO WE DO THIS?
        - Called by DataLoader when getting a batch
        - Returns one input sequence and its corresponding target
        """
        # Get the features (shape: sequence_length x num_features)
        x = self.features[idx]
        
        # Get the target (scalar)
        y = self.targets[idx]
        
        # Convert to PyTorch tensors
        x = torch.FloatTensor(x)
        y = torch.tensor(y, dtype=torch.float32)
        
        return x, y


def create_sequences(features, targets, sequence_length=24):
    """
    Create sequences from the data for LSTM training.
    
    This transforms the data into the format LSTM expects:
    - Each sample contains sequence_length timesteps
    - The target is the value at the next timestep
    
    Args:
        features: DataFrame or array of feature data
        targets: Array of target values
        sequence_length: Number of past hours to use as input
    
    Returns:
        tuple: (X_sequences, y_targets)
    
    WHY DO WE DO THIS?
    - LSTM needs sequential input
    - We use past 24 hours to predict next hour
    - This creates sliding windows over the data
    
    Example:
    - If we have 100 hours of data and sequence_length=24
    - We get 100-24 = 76 sequences
    - Each sequence has 24 timesteps
    """
    print(f"Creating sequences with length {sequence_length}...")
    
    X = []
    y = []
    
    # Convert to numpy if DataFrame
    if isinstance(features, pd.DataFrame):
        features = features.values
    if isinstance(targets, pd.DataFrame):
        targets = targets.values
    
    # Create sliding windows
    for i in range(len(features) - sequence_length):
        # Input: sequence of features from i to i+sequence_length-1
        X.append(features[i:i + sequence_length])
        
        # Target: value at i+sequence_length (next hour)
        y.append(targets[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"  Created {len(X)} sequences")
    print(f"  X shape: {X.shape} (samples x timesteps x features)")
    print(f"  y shape: {y.shape} (samples)")
    
    return X, y


def create_sequences_from_df(df, feature_cols, target_col, sequence_length=24):
    """
    Create sequences directly from a DataFrame.
    
    This is a convenience function that handles everything:
    - Extracts features and target
    - Creates sequences
    - Returns ready-to-use data
    
    Args:
        df: Preprocessed DataFrame
        feature_cols: List of column names for features
        target_col: Name of target column
        sequence_length: Number of past hours to use
    
    Returns:
        tuple: (X_sequences, y_targets)
    
    WHY DO WE DO THIS?
    - Simple interface for creating sequences
    - Works with the preprocessed DataFrame directly
    """
    print(f"Creating sequences from DataFrame...")
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Target: {target_col}")
    
    # Extract features and target
    features = df[feature_cols].values
    targets = df[target_col].values
    
    # Create sequences
    X, y = create_sequences(features, targets, sequence_length)
    
    return X, y


# ============================================================================
# PART 2: DATA LOADER CREATION
# ============================================================================

def create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32, shuffle=True):
    """
    Create PyTorch DataLoaders for training and testing.
    
    DataLoader wraps a Dataset and provides:
    - Batching: Groups samples into batches
    - Shuffling: Randomizes order each epoch (training only)
    - Parallel loading: Uses multiple workers for efficiency
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle training data
    
    Returns:
        tuple: (train_loader, test_loader)
    
    WHY DO WE DO THIS?
    - DataLoader handles batching automatically
    - Makes training code cleaner
    - Enables parallel data loading
    """
    print(f"Creating DataLoaders with batch_size={batch_size}...")
    
    # Create Datasets
    train_dataset = EnergyDataset(X_train, y_train)
    test_dataset = EnergyDataset(X_test, y_test)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0  # Set to >0 for parallel loading
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for testing
        num_workers=0
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    return train_loader, test_loader


# ============================================================================
# PART 3: SEQUENCE UTILITY FUNCTIONS
# ============================================================================

def get_latest_sequence(df, feature_cols, sequence_length=24):
    """
    Get the most recent sequence for prediction.
    
    This is used for real-time inference - we need the last 24 hours
    to make a prediction for the next hour.
    
    Args:
        df: DataFrame with recent data
        feature_cols: List of feature column names
        sequence_length: Number of hours to use
    
    Returns:
        np.array: Shape (sequence_length, num_features)
    
    WHY DO WE DO THIS?
    - For prediction, we need the most recent data
    - This extracts the last 24 hours from the DataFrame
    - Used by the real-time inference pipeline
    """
    if len(df) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} records, got {len(df)}")
    
    # Get the last sequence_length rows
    features = df[feature_cols].values
    sequence = features[-sequence_length:]
    
    return sequence


def prepare_input_for_prediction(sequence, scaler=None):
    """
    Prepare a sequence for model prediction.
    
    This normalizes the sequence if a scaler is provided.
    
    Args:
        sequence: numpy array of shape (sequence_length, num_features)
        scaler: Optional fitted scaler
    
    Returns:
        torch.Tensor: Ready for model input
    
    WHY DO WE DO THIS?
    - Model expects normalized input
    - This ensures the input is in the right format
    """
    # Apply scaler if provided
    if scaler is not None:
        # Flatten, transform, reshape
        original_shape = sequence.shape
        flat = sequence.reshape(-1, sequence.shape[-1])
        normalized = scaler.transform(flat)
        sequence = normalized.reshape(original_shape)
    
    # Convert to tensor
    x = torch.FloatTensor(sequence)
    
    # Add batch dimension (model expects batch x sequence x features)
    x = x.unsqueeze(0)
    
    return x


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Sequence Data Loader")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    n_samples = 200
    
    # Simulate features: 10 features
    np.random.seed(42)
    features = np.random.randn(n_samples, 10)
    
    # Target is related to features
    targets = 50000 + features[:, 0] * 1000 + features[:, 1] * 500 + np.random.randn(n_samples) * 200
    
    print(f"   Features shape: {features.shape}")
    print(f"   Targets shape: {targets.shape}")
    
    # Test sequence creation
    print("\n2. Testing sequence creation...")
    sequence_length = 24
    X, y = create_sequences(features, targets, sequence_length)
    
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    # Test Dataset
    print("\n3. Testing EnergyDataset...")
    dataset = EnergyDataset(X, y, sequence_length)
    
    # Get a sample
    sample_x, sample_y = dataset[0]
    print(f"   Sample X shape: {sample_x.shape}")
    print(f"   Sample y shape: {sample_y.shape}")
    
    # Test DataLoader
    print("\n4. Testing DataLoader...")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Get a batch
    batch_x, batch_y = next(iter(loader))
    print(f"   Batch X shape: {batch_x.shape}")
    print(f"   Batch y shape: {batch_y.shape}")
    
    # Test train/test split
    print("\n5. Testing train/test split...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(X_train, y_train, X_test, y_test, batch_size=32)
    
    print("\n" + "=" * 60)
    print("Sequence Data Loader Test Complete!")
    print("=" * 60)