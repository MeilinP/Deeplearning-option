# Deep Learning for Options Trading

Implementation of "Deep Learning for Options Trading: An End-to-End Approach" ([arXiv:2407.21791](https://arxiv.org/abs/2407.21791))

## Overview

This project implements an end-to-end deep learning framework for options trading that learns directly from market data without requiring explicit option pricing models. The approach uses neural networks to map market features to optimal trading signals, optimized for risk-adjusted performance via Sharpe ratio maximization.

**Data**: Works with **2 years of historical options data** from Polygon.io Starter plan (while the original paper used 10+ years of S&P 100 data, the methodology applies to any timeframe).

## Features

- **Real Market Data**: Integration with Polygon.io API for historical options data
- **Multiple Architectures**: Linear, MLP, CNN, and LSTM models
- **Custom Loss Functions**: Sharpe ratio optimization with turnover regularization
- **Comprehensive Backtesting**: Realistic simulation with transaction costs and slippage
- **Feature Engineering**: Automated computation of moneyness, time to maturity, implied volatility

## Installation

1. Clone or download this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your Polygon.io API key in `.env`:

```bash
cp .env.example .env
# Edit .env and add your API key
```

## Project Structure

```
Deeplearning-option/
├── data_utils.py           # Data fetching and feature engineering
├── options_trading_dl.py   # Neural network models and training
├── backtesting.py          # Backtesting framework
├── demo.ipynb             # Interactive demonstration
├── requirements.txt        # Python dependencies
├── .env                   # API credentials (not in git)
└── README.md              # This file
```

## Quick Start

### 1. Test with Synthetic Data

```python
from data_utils import generate_synthetic_options_data
from options_trading_dl import LinearModel, OptionsTrader
from backtesting import Backtester

# Generate synthetic data
X, returns = generate_synthetic_options_data(n_samples=1000, n_features=10)

# Split data
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = returns[:split_idx], returns[split_idx:]

# Train model
model = LinearModel(input_dim=10)
trader = OptionsTrader(model, learning_rate=1e-3, turnover_penalty=0.1)
trader.fit(X_train, y_train, epochs=50)

# Generate signals and backtest
signals = trader.predict(X_test).squeeze()
backtester = Backtester(initial_capital=100000, transaction_cost=0.001)
metrics = backtester.run(signals, y_test)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Total Return: {metrics['total_return']*100:.2f}%")
```

### 2. Use Real Market Data

```python
from data_utils import PolygonOptionsDataLoader

# Initialize data loader
loader = PolygonOptionsDataLoader()

# Fetch options data
options_df = loader.get_options_chain(
    ticker="AAPL",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

# Process and train (see demo.ipynb for full example)
```

## Neural Network Architectures

### 1. Linear Model
Simple baseline: `Input → Linear → Tanh → Signal`

### 2. MLP (Multilayer Perceptron)
Deep network: `Input → Hidden Layers → Output → Tanh → Signal`

### 3. CNN (Convolutional Neural Network)
For time-series features: `Input → Conv Layers → Pooling → Dense → Tanh → Signal`

### 4. LSTM (Long Short-Term Memory)
For sequential data: `Input → LSTM Layers → Dense → Tanh → Signal`

All models output trading signals in the range [-1, 1] where:
- +1 = maximum long position
- 0 = no position
- -1 = maximum short position

## Loss Function

The models optimize a custom loss combining:

1. **Negative Sharpe Ratio**: Maximize risk-adjusted returns
   ```
   Loss_sharpe = -mean(returns) / std(returns) * sqrt(252)
   ```

2. **Turnover Regularization**: Penalize excessive trading
   ```
   Loss_turnover = λ * mean(|signal_t - signal_{t-1}|)
   ```

Total loss: `Loss = Loss_sharpe + Loss_turnover`

## Backtesting

The backtesting framework includes:
- Transaction costs (default 0.1%)
- Slippage simulation
- Performance metrics: Sharpe ratio, max drawdown, total return
- Comparison with baseline strategies

## Results Interpretation

Key metrics to evaluate:
- **Sharpe Ratio**: Risk-adjusted performance (higher is better, >1 is good)
- **Total Return**: Overall profit/loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Turnover**: Trading frequency (lower reduces costs)

## Paper Citation

```bibtex
@article{tan2024deep,
  title={Deep Learning for Options Trading: An End-To-End Approach},
  author={Tan, Wee Ling and Roberts, Stephen and Zohren, Stefan},
  journal={arXiv preprint arXiv:2407.21791},
  year={2024}
}
```

## Disclaimer

This implementation is for educational and research purposes only. It is not financial advice. Trading options involves substantial risk and may not be suitable for all investors.

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or pull request with improvements.
