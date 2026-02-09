"""
Deep Learning Models for Options Trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from tqdm import tqdm


class LinearModel(nn.Module):

    def __init__(self, input_dim: int):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        return torch.tanh(out)


class MLPModel(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        super(MLPModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.network(x)
        return torch.tanh(out)


class CNNModel(nn.Module):

    def __init__(self, input_dim: int, seq_length: int = 10, num_filters: int = 32):
        super(CNNModel, self).__init__()

        self.input_dim = input_dim
        self.seq_length = seq_length

        # Causal 1D convolutions (no padding; we pad manually in forward)
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(num_filters, num_filters // 2, kernel_size=3, padding=0)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(num_filters // 2, num_filters // 4)
        self.fc2 = nn.Linear(num_filters // 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            x = x.view(batch_size, self.seq_length, self.input_dim)

        x = x.permute(0, 2, 1)  # (batch, features, seq_length)

        # Causal padding: pad left only to avoid look-ahead
        x = F.pad(x, (2, 0))
        x = F.relu(self.conv1(x))
        x = F.pad(x, (2, 0))
        x = F.relu(self.conv2(x))

        x = self.pool(x).squeeze(-1)  # (batch, num_filters//2)

        x = torch.tanh(self.fc1(x))
        out = self.fc2(x)
        return torch.tanh(out)


class LSTMModel(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = 40, num_layers: int = 1):
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
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)
        out = self.fc(last_hidden)
        return torch.tanh(out)


class SharpeRatioLoss(nn.Module):

    def __init__(self, risk_free_rate: float = 0.0):
        super(SharpeRatioLoss, self).__init__()
        self.risk_free_rate = risk_free_rate

    def forward(self, signals: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        portfolio_returns = signals.squeeze() * returns.squeeze()

        mean_return = torch.mean(portfolio_returns)
        std_return = torch.std(portfolio_returns) + 1e-8

        sharpe_ratio = (mean_return - self.risk_free_rate / 252) / std_return * np.sqrt(252)

        return -sharpe_ratio


class TurnoverRegularizedLoss(nn.Module):

    def __init__(self, sharpe_weight: float = 1.0, turnover_weight: float = 0.1):
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

        sharpe_loss = self.sharpe_loss(signals, returns)

        turnover_loss = torch.tensor(0.0, device=signals.device)
        if prev_signals is not None:
            if volatilities is not None and prev_volatilities is not None:
                # Vol-scaled turnover: |X_t/sigma_t - X_{t-1}/sigma_{t-1}|
                scaled_signals = signals / (volatilities + 1e-8)
                scaled_prev = prev_signals / (prev_volatilities + 1e-8)
                turnover = torch.mean(torch.abs(scaled_signals - scaled_prev))
            else:
                turnover = torch.mean(torch.abs(signals - prev_signals))
            turnover_loss = self.turnover_weight * turnover

        total_loss = self.sharpe_weight * sharpe_loss + turnover_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'sharpe_loss': sharpe_loss.item(),
            'turnover_loss': turnover_loss.item() if prev_signals is not None else 0.0
        }

        return total_loss, loss_dict


class OptionsTrader:

    def __init__(self,
                 model: nn.Module,
                 learning_rate: float = 1e-3,
                 turnover_penalty: float = 0.1,
                 device: str = 'cpu'):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.loss_fn = TurnoverRegularizedLoss(turnover_weight=turnover_penalty)
        self.training_history = []

    def train_epoch(self,
                   features: np.ndarray,
                   returns: np.ndarray,
                   batch_size: int = 32) -> dict:
        self.model.train()
        n_samples = len(features)
        indices = np.random.permutation(n_samples)

        epoch_losses = []
        prev_signals = None

        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]

            X_batch = torch.FloatTensor(features[batch_idx]).to(self.device)
            y_batch = torch.FloatTensor(returns[batch_idx]).unsqueeze(1).to(self.device)

            signals = self.model(X_batch)
            loss, loss_dict = self.loss_fn(signals, y_batch, None)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss_dict)

        avg_losses = {
            key: np.mean([loss[key] for loss in epoch_losses])
            for key in epoch_losses[0].keys()
        }

        return avg_losses

    def evaluate(self, features: np.ndarray, returns: np.ndarray) -> Tuple[np.ndarray, dict]:
        self.model.eval()

        with torch.no_grad():
            X = torch.FloatTensor(features).to(self.device)
            y = torch.FloatTensor(returns).unsqueeze(1).to(self.device)

            signals = self.model(X)
            loss, loss_dict = self.loss_fn(signals, y)

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

        history = {'train_loss': [], 'train_sharpe': []}
        if X_val is not None:
            history['val_loss'] = []
            history['val_sharpe'] = []

        iterator = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in iterator:
            train_losses = self.train_epoch(X_train, y_train, batch_size)
            _, train_metrics = self.evaluate(X_train, y_train)

            history['train_loss'].append(train_losses['total_loss'])
            history['train_sharpe'].append(train_metrics['sharpe_ratio'])

            if X_val is not None and y_val is not None:
                _, val_metrics = self.evaluate(X_val, y_val)
                history['val_loss'].append(val_metrics['total_loss'])
                history['val_sharpe'].append(val_metrics['sharpe_ratio'])

            if verbose and epoch % 10 == 0:
                status = f"Epoch {epoch}: Train Sharpe={train_metrics['sharpe_ratio']:.3f}"
                if X_val is not None:
                    status += f", Val Sharpe={val_metrics['sharpe_ratio']:.3f}"
                if hasattr(iterator, 'set_description'):
                    iterator.set_description(status)

        self.training_history = history
        return history

    def predict(self, features: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            X = torch.FloatTensor(features).to(self.device)
            signals = self.model(X)
        return signals.cpu().numpy()

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])


if __name__ == "__main__":
    print("Testing neural network architectures...")

    batch_size = 32
    input_dim = 10
    seq_length = 5

    print("\n1. Linear Model")
    model = LinearModel(input_dim)
    x = torch.randn(batch_size, input_dim)
    out = model(x)
    print(f"   Input: {x.shape}, Output: {out.shape}, Range: [{out.min():.3f}, {out.max():.3f}]")

    print("\n2. MLP Model")
    model = MLPModel(input_dim, hidden_dims=[64, 32])
    out = model(x)
    print(f"   Input: {x.shape}, Output: {out.shape}, Range: [{out.min():.3f}, {out.max():.3f}]")

    print("\n3. CNN Model")
    model = CNNModel(input_dim=2, seq_length=seq_length)
    x_seq = torch.randn(batch_size, seq_length, 2)
    out = model(x_seq)
    print(f"   Input: {x_seq.shape}, Output: {out.shape}, Range: [{out.min():.3f}, {out.max():.3f}]")

    print("\n4. LSTM Model")
    model = LSTMModel(input_dim=2, hidden_dim=40)
    out = model(x_seq)
    print(f"   Input: {x_seq.shape}, Output: {out.shape}, Range: [{out.min():.3f}, {out.max():.3f}]")

    print("\n5. Loss Functions")
    signals = torch.randn(100, 1)
    returns = torch.randn(100, 1) * 0.02

    sharpe_loss = SharpeRatioLoss()
    loss = sharpe_loss(signals, returns)
    print(f"   Sharpe loss: {loss.item():.4f}")

    turnover_loss = TurnoverRegularizedLoss(turnover_weight=0.1)
    prev_signals = signals + torch.randn(100, 1) * 0.1
    loss, loss_dict = turnover_loss(signals, returns, prev_signals)
    print(f"   Turnover regularized loss: {loss_dict}")

    print("\nAll models working correctly!")
