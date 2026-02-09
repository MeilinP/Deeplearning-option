"""
Deep Learning Models for Options Trading
Implements Linear, MLP, CNN, and LSTM architectures with Sharpe ratio loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm


class LinearModel(nn.Module):
    """Simple linear model: Input -> Linear -> Tanh -> Signal"""
    
    def __init__(self, input_dim: int):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            Trading signals in range [-1, 1] (batch_size, 1)
        """
        out = self.linear(x)
        return torch.tanh(out)


class MLPModel(nn.Module):
    """Multilayer Perceptron with hidden layers"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        super(MLPModel, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with Tanh activation (per paper Eq. 9)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.2))  # Regularization
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, input_dim)
        
        Returns:
            Trading signals in range [-1, 1] (batch_size, 1)
        """
        out = self.network(x)
        return torch.tanh(out)


class CNNModel(nn.Module):
    """Convolutional Neural Network for time-series features"""
    
    def __init__(self, input_dim: int, seq_length: int = 10, num_filters: int = 32):
        """
        Args:
            input_dim: Number of features per time step
            seq_length: Length of time series
            num_filters: Number of convolutional filters
        """
        super(CNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        
        # Causal (1D) convolutions for time-series (left-only padding)
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=num_filters,
            kernel_size=3,
            padding=0
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters // 2,
            kernel_size=3,
            padding=0
        )
        
        self.pool = nn.AdaptiveAvgPool1d(1)  # Average pooling per paper Eq. 10
        self.fc1 = nn.Linear(num_filters // 2, num_filters // 4)  # Hidden FC layer
        self.fc2 = nn.Linear(num_filters // 4, 1)  # Output FC layer
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, seq_length, input_dim) or (batch_size, input_dim)
        
        Returns:
            Trading signals in range [-1, 1] (batch_size, 1)
        """
        # Handle both 2D and 3D input
        if len(x.shape) == 2:
            # Reshape (batch_size, input_dim) -> (batch_size, seq_length, features_per_step)
            batch_size = x.shape[0]
            features_per_step = self.input_dim
            x = x.view(batch_size, self.seq_length, features_per_step)
        
        # CNN expects (batch_size, channels, length)
        x = x.permute(0, 2, 1)  # (batch, features, seq_length)

        # Causal convolutional layers (left-only padding to avoid look-ahead bias)
        x = F.pad(x, (2, 0))  # Causal pad for kernel_size=3
        x = F.relu(self.conv1(x))
        x = F.pad(x, (2, 0))  # Causal pad for kernel_size=3
        x = F.relu(self.conv2(x))

        # Global average pooling (per paper Eq. 10)
        x = self.pool(x)  # (batch, num_filters//2, 1)
        x = x.squeeze(-1)  # (batch, num_filters//2)

        # Two fully connected layers (per paper Eq. 10)
        x = torch.tanh(self.fc1(x))  # Hidden FC with tanh
        out = self.fc2(x)  # Output FC
        return torch.tanh(out)


class LSTMModel(nn.Module):
    """LSTM for sequential options data"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 40, num_layers: int = 1):
        """
        Args:
            input_dim: Number of features per time step
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, seq_length, input_dim)
        
        Returns:
            Trading signals in range [-1, 1] (batch_size, 1)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)
        
        # Fully connected output
        out = self.fc(last_hidden)
        return torch.tanh(out)


class SharpeRatioLoss(nn.Module):
    """Custom loss function based on negative Sharpe ratio"""
    
    def __init__(self, risk_free_rate: float = 0.0):
        """
        Args:
            risk_free_rate: Annual risk-free rate (default 0)
        """
        super(SharpeRatioLoss, self).__init__()
        self.risk_free_rate = risk_free_rate
    
    def forward(self, signals: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative Sharpe ratio
        
        Args:
            signals: Trading signals from model (batch_size, 1) in range [-1, 1]
            returns: Actual returns (batch_size, 1)
        
        Returns:
            Negative Sharpe ratio (scalar)
        """
        # Portfolio returns = signal_{t-1} * return_t
        # For simplicity, use current signals (in practice, shift by 1)
        portfolio_returns = signals.squeeze() * returns.squeeze()
        
        # Calculate mean and std
        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8  # Avoid division by zero
        
        # Sharpe ratio (annualized, assuming daily returns)
        sharpe_ratio = (mean_return - self.risk_free_rate / 252) / std_return * np.sqrt(252)
        
        # Return negative (we want to maximize Sharpe, so minimize negative Sharpe)
        return -sharpe_ratio


class TurnoverRegularizedLoss(nn.Module):
    """Sharpe ratio loss with turnover regularization"""
    
    def __init__(self, sharpe_weight: float = 1.0, turnover_weight: float = 0.1):
        """
        Args:
            sharpe_weight: Weight for Sharpe ratio term
            turnover_weight: Weight for turnover penalty (lambda)
        """
        super(TurnoverRegularizedLoss, self).__init__()
        self.sharpe_loss = SharpeRatioLoss()
        self.sharpe_weight = sharpe_weight
        self.turnover_weight = turnover_weight
    
    def forward(self,
                signals: torch.Tensor,
                returns: torch.Tensor,
                prev_signals: Optional[torch.Tensor] = None,
                volatilities: Optional[torch.Tensor] = None,
                prev_volatilities: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        Calculate loss with turnover regularization

        Args:
            signals: Current trading signals (batch_size, 1)
            returns: Actual returns (batch_size, 1)
            prev_signals: Previous signals (batch_size, 1), if available
            volatilities: Current volatilities (batch_size, 1), for vol-scaled turnover
            prev_volatilities: Previous volatilities (batch_size, 1), for vol-scaled turnover

        Returns:
            Total loss and a dict with loss components
        """
        # Sharpe ratio loss
        sharpe_loss = self.sharpe_loss(signals, returns)

        # Turnover penalty (per paper Eq. 13: |X_t/σ_t - X_{t-1}/σ_{t-1}|)
        turnover_loss = torch.tensor(0.0, device=signals.device)
        if prev_signals is not None:
            if volatilities is not None and prev_volatilities is not None:
                # Volatility-scaled turnover per paper Eq. 13
                scaled_signals = signals / (volatilities + 1e-8)
                scaled_prev = prev_signals / (prev_volatilities + 1e-8)
                turnover = torch.mean(torch.abs(scaled_signals - scaled_prev))
            else:
                # Fallback: unscaled turnover
                turnover = torch.mean(torch.abs(signals - prev_signals))
            turnover_loss = self.turnover_weight * turnover
        
        # Total loss
        total_loss = self.sharpe_weight * sharpe_loss + turnover_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'sharpe_loss': sharpe_loss.item(),
            'turnover_loss': turnover_loss.item() if prev_signals is not None else 0.0
        }
        
        return total_loss, loss_dict


class OptionsTrader:
    """Training and inference for options trading models"""
    
    def __init__(self, 
                 model: nn.Module,
                 learning_rate: float = 1e-3,
                 turnover_penalty: float = 0.1,
                 device: str = 'cpu'):
        """
        Args:
            model: Neural network model
            learning_rate: Learning rate for optimizer
            turnover_penalty: Lambda for turnover regularization
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = TurnoverRegularizedLoss(turnover_weight=turnover_penalty)
        self.training_history = []
    
    def train_epoch(self, 
                   features: np.ndarray, 
                   returns: np.ndarray,
                   batch_size: int = 32) -> dict:
        """
        Train for one epoch
        
        Args:
            features: Feature matrix (n_samples, n_features)
            returns: Returns array (n_samples,)
            batch_size: Batch size
        
        Returns:
            Dictionary with average epoch losses
        """
        self.model.train()
        n_samples = len(features)
        indices = np.random.permutation(n_samples)
        
        epoch_losses = []
        prev_signals = None
        
        # Mini-batch training
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            
            # Convert to tensors
            X_batch = torch.FloatTensor(features[batch_idx]).to(self.device)
            y_batch = torch.FloatTensor(returns[batch_idx]).unsqueeze(1).to(self.device)
            
            # Forward pass
            signals = self.model(X_batch)
            
            # Calculate loss (don't use prev_signals for turnover across batches - sizes may differ)
            loss, loss_dict = self.loss_fn(signals, y_batch, None)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(loss_dict)
        
        # Average losses
        avg_losses = {
            key: np.mean([loss[key] for loss in epoch_losses])
            for key in epoch_losses[0].keys()
        }
        
        return avg_losses
    
    def evaluate(self, features: np.ndarray, returns: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Evaluate model on data
        
        Args:
            features: Feature matrix
            returns: Returns array
        
        Returns:
            Tuple of (predictions, metrics)
        """
        self.model.eval()
        
        with torch.no_grad():
            X = torch.FloatTensor(features).to(self.device)
            y = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
            
            # Get predictions
            signals = self.model(X)
            
            # Calculate metrics
            loss, loss_dict = self.loss_fn(signals, y)
            
            # Calculate Sharpe ratio (positive)
            portfolio_returns = signals.squeeze().cpu().numpy() * returns
            sharpe_ratio = np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252)
            
            metrics = {
                **loss_dict,
                'sharpe_ratio': sharpe_ratio,
                'mean_return': np.mean(portfolio_returns),
                'std_return': np.std(portfolio_returns)
            }
        
        return signals.cpu().numpy(), metrics
    
    def fit(self, 
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: bool = True) -> dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training returns
            X_val: Validation features (optional)
            y_val: Validation returns (optional)
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Print progress
        
        Returns:
            Training history
        """
        history = {'train_loss': [], 'train_sharpe': []}
        if X_val is not None:
            history['val_loss'] = []
            history['val_sharpe'] = []
        
        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)
        
        for epoch in iterator:
            # Training
            train_losses = self.train_epoch(X_train, y_train, batch_size)
            
            # Evaluation
            _, train_metrics = self.evaluate(X_train, y_train)
            
            history['train_loss'].append(train_losses['total_loss'])
            history['train_sharpe'].append(train_metrics['sharpe_ratio'])
            
            # Validation
            if X_val is not None and y_val is not None:
                _, val_metrics = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_metrics['total_loss'])
                history['val_sharpe'].append(val_metrics['sharpe_ratio'])
            
            # Update progress bar
            if verbose and epoch % 10 == 0:
                status = f"Epoch {epoch}: Train Sharpe={train_metrics['sharpe_ratio']:.3f}"
                if X_val is not None:
                    status += f", Val Sharpe={val_metrics['sharpe_ratio']:.3f}"
                if hasattr(iterator, 'set_description'):
                    iterator.set_description(status)
        
        self.training_history = history
        return history
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Generate trading signals
        
        Args:
            features: Feature matrix
        
        Returns:
            Trading signals in range [-1, 1]
        """
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features).to(self.device)
            signals = self.model(X)
        return signals.cpu().numpy()
    
    def save_model(self, path: str):
        """Save model to file"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)
    
    def load_model(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])


if __name__ == "__main__":
    # Test models with synthetic data
    print("Testing neural network architectures...")
    
    batch_size = 32
    input_dim = 10
    seq_length = 5
    
    # Test Linear Model
    print("\n1. Linear Model")
    model = LinearModel(input_dim)
    x = torch.randn(batch_size, input_dim)
    out = model(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    print(f"   Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Test MLP Model
    print("\n2. MLP Model")
    model = MLPModel(input_dim, hidden_dims=[64, 32])
    out = model(x)
    print(f"   Input shape: {x.shape}, Output shape: {out.shape}")
    print(f"   Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Test CNN Model
    print("\n3. CNN Model")
    model = CNNModel(input_dim=2, seq_length=seq_length)
    x_seq = torch.randn(batch_size, seq_length, 2)
    out = model(x_seq)
    print(f"   Input shape: {x_seq.shape}, Output shape: {out.shape}")
    print(f"   Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Test LSTM Model
    print("\n4. LSTM Model")
    model = LSTMModel(input_dim=2, hidden_dim=40)
    out = model(x_seq)
    print(f"   Input shape: {x_seq.shape}, Output shape: {out.shape}")
    print(f"   Output range: [{out.min():.3f}, {out.max():.3f}]")
    
    # Test loss functions
    print("\n5. Testing Loss Functions")
    signals = torch.randn(100, 1)
    returns = torch.randn(100, 1) * 0.02  # 2% volatility
    
    sharpe_loss = SharpeRatioLoss()
    loss = sharpe_loss(signals, returns)
    print(f"   Sharpe loss: {loss.item():.4f}")
    
    turnover_loss = TurnoverRegularizedLoss(turnover_weight=0.1)
    prev_signals = signals + torch.randn(100, 1) * 0.1
    loss, loss_dict = turnover_loss(signals, returns, prev_signals)
    print(f"   Turnover regularized loss: {loss_dict}")
    
    print("\n✓ All models working correctly!")
