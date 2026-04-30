"""
Training Module with Custom Training Loop
============================================

This module implements a complete custom training loop in PyTorch.
This is NOT using high-level APIs like sklearn - we write everything
from scratch to demonstrate understanding.

Author: Student
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


# ============================================================================
# PART 1: TRAINING CONFIGURATION
# ============================================================================

class TrainingConfig:
    """
    Configuration class for training parameters.
    
    This keeps all training hyperparameters in one place.
    """
    
    def __init__(self):
        # Model parameters
        self.input_size = None  # Set based on data
        self.hidden_size = 128
        self.num_layers = 2
        self.dropout = 0.2
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 50
        
        # Early stopping
        self.patience = 10  # epochs to wait before early stop
        self.min_delta = 50  # minimum improvement to count
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Output
        self.model_dir = "models"
        self.results_dir = "results"
        
    def __repr__(self):
        return f"""
TrainingConfig:
  Model: hidden_size={self.hidden_size}, num_layers={self.num_layers}
  Training: epochs={self.num_epochs}, batch_size={self.batch_size}, lr={self.learning_rate}
  Early stopping: patience={self.patience}, min_delta={self.min_delta}
  Device: {self.device}
"""


# ============================================================================
# PART 2: CUSTOM TRAINING LOOP
# ============================================================================

class Trainer:
    """
    Custom training loop for the LSTM model.
    
    This implements a complete training pipeline from scratch:
    1. Initialize model and optimizer
    2. Loop through epochs
    3. Loop through batches
    4. Forward pass
    5. Calculate loss
    6. Backward pass
    7. Update weights
    8. Evaluate on validation set
    9. Save best model
    
    WHY CUSTOM LOOP?
    ================
    - Shows understanding of deep learning fundamentals
    - More control than using high-level APIs
    - Can customize any part of the process
    - Better for learning/education
    """
    
    def __init__(self, model, config):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            config: TrainingConfig object
        """
        self.model = model
        self.config = config
        
        # Move model to device
        self.model.to(config.device)
        
        # Loss function - MSE for regression
        self.criterion = nn.MSELoss()
        
        # Optimizer - Adam is a good default choice
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Learning rate scheduler - reduces LR when loss plateaus
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        This is the core training loop - processes all batches once.
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()  # Set to training mode
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)
            
            # ============================================================
            # STEP 1: FORWARD PASS
            # ============================================================
            # Pass data through the model
            # This computes: input -> hidden -> output
            outputs = self.model(inputs)
            
            # ============================================================
            # STEP 2: CALCULATE LOSS
            # ============================================================
            # Compare predictions to actual values
            loss = self.criterion(outputs.squeeze(), targets)
            
            # ============================================================
            # STEP 3: BACKWARD PASS
            # ============================================================
            # Clear gradients from previous iteration
            self.optimizer.zero_grad()
            
            # Compute gradients (backpropagation)
            loss.backward()
            
            # ============================================================
            # STEP 4: UPDATE WEIGHTS
            # ============================================================
            # Update model parameters using gradients
            self.optimizer.step()
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            float: Average validation loss
        """
        self.model.eval()  # Set to evaluation mode
        total_loss = 0
        num_batches = 0
        
        # Don't compute gradients during validation
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move to device
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs.squeeze(), targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader):
        """
        Main training loop.
        
        This trains the model for multiple epochs, tracking progress
        and saving the best model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        
        Returns:
            dict: Training history
        """
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Device: {self.config.device}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("=" * 60)
        
        start_time = datetime.now()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = datetime.now()
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 40)
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            epoch_time = (datetime.now() - epoch_start).total_seconds()
            self.history['epoch_times'].append(epoch_time)
            
            # Print progress
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # ============================================================
            # LEARNING RATE SCHEDULER
            # ============================================================
            # Adjust learning rate based on validation loss
            self.scheduler.step(val_loss)
            
            # ============================================================
            # EARLY STOPPING CHECK
            # ============================================================
            if val_loss < self.best_val_loss - self.config.min_delta:
                # Improvement found!
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_without_improvement = 0
                print(f"  ✓ New best model! Val loss: {val_loss:.4f}")
            else:
                # No improvement
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
                
                # Check for early stopping
                if self.epochs_without_improvement >= self.config.patience:
                    print(f"\nEarly stopping triggered!")
                    print(f"No improvement for {self.config.patience} epochs")
                    break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nRestored best model with val loss: {self.best_val_loss:.4f}")
        
        total_time = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 60)
        print(f"TRAINING COMPLETE!")
        print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 60)
        
        return self.history
    
    def plot_training_history(self, save_path=None):
        """
        Plot the training history.
        
        Args:
            save_path: Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot epoch times
        ax2.plot(self.history['epoch_times'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Epoch Training Time')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Training plot saved to: {save_path}")
        
        plt.show()


# ============================================================================
# PART 3: EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        dict: Dictionary of metrics
    
    WHY DO WE NEED THESE?
    =====================
    - Loss tells us how well the model is training
    - But other metrics help understand real-world performance
    - MAE: Average error in same units as data
    - RMSE: Penalizes large errors more
    - MAPE: Percentage error (scale-independent)
    - R²: How much variance is explained (0-1)
    """
    # Convert to numpy if tensors
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    # R² Score (Coefficient of Determination)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }


def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for evaluation data
        device: Device to run on
    
    Returns:
        tuple: (predictions, actuals, metrics)
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Store predictions and targets
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Convert to arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    metrics = calculate_metrics(all_targets, all_preds)
    
    return all_preds, all_targets, metrics


# ============================================================================
# PART 4: MAIN TRAINING FUNCTION
# ============================================================================

def train_model(
    model,
    train_loader,
    val_loader,
    config=None,
    save_path=None
):
    """
    Main function to train the model.
    
    This is the entry point for training.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: TrainingConfig (optional)
        save_path: Path to save the trained model
    
    Returns:
        tuple: (trained_model, history, metrics)
    """
    # Create default config if not provided
    if config is None:
        config = TrainingConfig()
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    preds, targets, metrics = evaluate_model(model, val_loader, config.device)
    
    print("\nValidation Metrics:")
    print(f"  MAE: {metrics['MAE']:.2f} MW")
    print(f"  RMSE: {metrics['RMSE']:.2f} MW")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  R²: {metrics['R2']:.4f}")
    
    # Save model if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_size': config.input_size,
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'dropout': config.dropout
            },
            'metrics': metrics,
            'history': history
        }, save_path)
        print(f"\nModel saved to: {save_path}")
    
    return model, history, metrics


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Training Module")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample data...")
    torch.manual_seed(42)
    
    # Simple model for testing
    from src.model import create_model
    
    input_size = 15
    model = create_model(input_size, hidden_size=64, num_layers=1, dropout=0.1)
    
    # Create dummy data loaders
    from src.sequence_loader import create_sequences
    
    # Generate random data
    X = np.random.randn(500, 24, input_size)
    y = np.random.randn(500) * 1000 + 50000
    
    # Split
    split_idx = 400
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create data loaders
    from torch.utils.data import DataLoader, TensorDataset
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Test training
    print("\n2. Testing training loop...")
    config = TrainingConfig()
    config.input_size = input_size
    config.num_epochs = 5
    config.batch_size = 32
    
    trainer = Trainer(model, config)
    history = trainer.train(train_loader, val_loader)
    
    # Test evaluation
    print("\n3. Testing evaluation...")
    preds, targets, metrics = evaluate_model(model, val_loader, config.device)
    print(f"  MAE: {metrics['MAE']:.2f}")
    print(f"  RMSE: {metrics['RMSE']:.2f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    print("\n" + "=" * 60)
    print("Training Module Test Complete!")
    print("=" * 60)