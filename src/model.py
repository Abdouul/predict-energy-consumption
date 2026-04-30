"""
LSTM Model for Energy Consumption Forecasting
==============================================

This module defines the LSTM neural network architecture for
time series forecasting.

Author: Student
"""

import torch
import torch.nn as nn


# ============================================================================
# PART 1: LSTM MODEL ARCHITECTURE
# ============================================================================

class EnergyLSTM(nn.Module):
    """
    LSTM Model for Energy Consumption Forecasting.
    
    This is a multi-layer LSTM with dropout for regularization.
    
    ARCHITECTURE EXPLANATION:
    =========================
    
    1. Input Layer:
       - Takes sequence of features (past 24 hours)
       - Each timestep has multiple features (weather, time, lags)
    
    2. LSTM Layers:
       - LSTM (Long Short-Term Memory) is a type of RNN
       - Can learn long-term dependencies in time series
       - Multiple layers allow learning complex patterns
       - Dropout prevents overfitting between layers
    
    3. Fully Connected Layer:
       - Converts LSTM output to prediction
       - Single output neuron for consumption value
    
    WHY LSTM?
    =========
    - Good for sequential data with temporal patterns
    - Can remember important information from far in the past
    - Handles variable-length sequences well
    - Standard choice for time series forecasting
    """
    
    def __init__(
        self,
        input_size,          # Number of features at each timestep
        hidden_size=128,     # Number of hidden units per LSTM layer
        num_layers=2,        # Number of LSTM layers
        dropout=0.2,        # Dropout probability between layers
        output_size=1        # Number of output values (1 for next hour)
    ):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features per timestep
            hidden_size: Number of hidden units in each LSTM
            num_layers: Number of LSTM layers stacked
            dropout: Dropout probability (0 = no dropout)
            output_size: Number of output values
        
        WHY DO WE NEED THESE PARAMETERS?
        ================================
        - input_size: Depends on your features (weather + time + lags)
        - hidden_size: Larger = more capacity, but slower and may overfit
        - num_layers: More layers = more complex patterns, but harder to train
        - dropout: Prevents overfitting (remembering training data too well)
        - output_size: 1 for single prediction, could be more for multi-step
        """
        super(EnergyLSTM, self).__init__()
        
        # Store configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Define the LSTM layers
        # batch_first=True means input is (batch, sequence, features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Only apply dropout if >1 layer
            batch_first=True
        )
        
        # Define the fully connected output layer
        # Takes the last hidden state and outputs the prediction
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights (optional but can help)
        self._init_weights()
        
        print(f"Created EnergyLSTM model:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Dropout: {dropout}")
        print(f"  Output size: {output_size}")
    
    def _init_weights(self):
        """
        Initialize model weights using Xavier initialization.
        
        This helps with training stability and convergence.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'fc' in name:
                    # Xavier initialization for linear layers
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                # Small positive bias for LSTM gates
                nn.init.constant_(param, 0.01)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        This is called when we pass data through the model.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_size)
        
        Returns:
            Output tensor of shape (batch, output_size)
        
        WHAT HAPPENS INSIDE?
        ====================
        1. LSTM processes the entire sequence
           - Each timestep's hidden state depends on previous
           - Final hidden state contains information from all timesteps
        
        2. Fully connected layer transforms LSTM output to prediction
           - Takes the last timestep's hidden state
           - Outputs the predicted consumption value
        """
        # Forward through LSTM
        # output: (batch, sequence, hidden_size)
        # hidden: tuple of (h_n, c_n) where h_n is final hidden state
        output, (hidden, cell) = self.lstm(x)
        
        # Take the output from the last timestep
        # This contains information from the entire sequence
        last_output = output[:, -1, :]
        
        # Pass through fully connected layer
        prediction = self.fc(last_output)
        
        return prediction
    
    def get_num_parameters(self):
        """
        Calculate the total number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# PART 2: MODEL SUMMARY
# ============================================================================

def print_model_summary(model, input_size, sequence_length=24):
    """
    Print a summary of the model architecture.
    
    Args:
        model: The PyTorch model
        input_size: Number of input features
        sequence_length: Length of input sequences
    """
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    print(f"\nLayer-by-layer breakdown:")
    
    # Create a dummy input to trace
    dummy_input = torch.randn(1, sequence_length, input_size)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"  (batch_size, sequence_length, num_features)")
    
    # Forward pass to see intermediate shapes
    with torch.no_grad():
        output = model.forward(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"  (batch_size, output_size)")
    
    print("\n" + "=" * 60)
    print("PARAMETER DETAILS")
    print("=" * 60)
    
    total_params = 0
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"  {name}: {list(param.shape)} ({num_params:,} params)")
    
    print(f"\nTotal trainable parameters: {total_params:,}")
    print("=" * 60)


# ============================================================================
# PART 3: SAVE/LOAD FUNCTIONS
# ============================================================================

def save_model(model, filepath):
    """
    Save the trained model to a file.
    
    Args:
        model: PyTorch model
        filepath: Path to save the model
    
    WHY DO WE DO THIS?
    ==================
    - Training takes time, we want to save the result
    - Can load later for inference
    - Model includes architecture and weights
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_layers': model.num_layers,
            'dropout': model.dropout,
            'output_size': 1
        }
    }, filepath)
    
    print(f"Model saved to: {filepath}")


def load_model(filepath, device=None):
    """
    Load a trained model from a file.
    
    Args:
        filepath: Path to the saved model
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        EnergyLSTM: Loaded model
    
    WHY DO WE DO THIS?
    ==================
    - Load trained model for inference
    - Don't need to retrain every time
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Recreate model with same architecture
    config = checkpoint['model_config']
    model = EnergyLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        output_size=config['output_size']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from: {filepath}")
    
    return model


# ============================================================================
# PART 4: MODEL CREATION HELPER
# ============================================================================

def create_model(input_size, hidden_size=128, num_layers=2, dropout=0.2):
    """
    Create a new LSTM model with default settings.
    
    This is a convenience function for creating models.
    
    Args:
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_layers: Number of LSTM layers
        dropout: Dropout probability
    
    Returns:
        EnergyLSTM: New model instance
    """
    model = EnergyLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        output_size=1
    )
    
    return model


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing LSTM Model")
    print("=" * 60)
    
    # Create a model
    print("\n1. Creating model...")
    input_size = 15  # Example: 15 features
    model = create_model(input_size, hidden_size=128, num_layers=2, dropout=0.2)
    
    # Print model summary
    print("\n2. Model summary:")
    print_model_summary(model, input_size, sequence_length=24)
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    dummy_input = torch.randn(4, 24, input_size)  # batch=4, seq=24, features=15
    
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Sample predictions: {output.squeeze().tolist()}")
    
    # Test save/load
    print("\n4. Testing save/load...")
    save_path = "test_model.pth"
    save_model(model, save_path)
    loaded_model = load_model(save_path)
    
    # Verify outputs match
    with torch.no_grad():
        original_output = model(dummy_input)
        loaded_output = loaded_model(dummy_input)
    
    if torch.allclose(original_output, loaded_output):
        print("   ✓ Save/load successful - outputs match!")
    else:
        print("   ✗ Warning: outputs don't match")
    
    print("\n" + "=" * 60)
    print("LSTM Model Test Complete!")
    print("=" * 60)