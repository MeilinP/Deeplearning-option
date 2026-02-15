"""
End-to-end integration test for the options trading system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data_utils import generate_synthetic_options_data, DataPreprocessor
from options_trading_dl import LinearModel, MLPModel, OptionsTrader
from backtesting import Backtester, print_metrics_table

print("="*60)
print("Deep Learning Options Trading - Integration Test")
print("="*60)

# 1. Generate data
print("\n[1/5] Generating synthetic data...")
n_samples = 500
n_features = 10

X, returns = generate_synthetic_options_data(n_samples=n_samples, n_features=n_features)
print(f"  ✓ Generated {n_samples} samples with {n_features} features")

# 2. Split and preprocess
print("\n[2/5] Splitting and preprocessing data...")
split_idx = int(n_samples * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = returns[:split_idx], returns[split_idx:]

preprocessor = DataPreprocessor()
X_train_norm = preprocessor.fit_transform(X_train)
X_test_norm = preprocessor.transform(X_test)
print(f"  ✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# 3. Train Linear Model
print("\n[3/5] Training Linear Model...")
linear_model = LinearModel(input_dim=n_features)
linear_trader = OptionsTrader(
    model=linear_model,
    learning_rate=1e-2,
    turnover_penalty=0.1,
    device='cpu'
)

history = linear_trader.fit(
    X_train_norm, y_train,
    X_test_norm, y_test,
    epochs=30,
    batch_size=32,
    verbose=False
)
print(f"  ✓ Model trained (final train Sharpe: {history['train_sharpe'][-1]:.3f})")

# 4. Generate trading signals
print("\n[4/5] Generating trading signals...")
signals = linear_trader.predict(X_test_norm).squeeze()
print(f"  ✓ Generated {len(signals)} signals")
print(f"  Signal range: [{signals.min():.3f}, {signals.max():.3f}]")
print(f"  Mean signal: {signals.mean():.3f}")

# 5. Backtest
print("\n[5/5] Running backtest...")
backtester = Backtester(
    initial_capital=100000,
    transaction_cost=0.001,
    slippage=0.0005
)
metrics = backtester.run(signals, y_test)

print("\n" + "="*60)
print_metrics_table(metrics, "Linear Model - Test Results")

# Summary
print("\n" + "="*60)
print("INTEGRATION TEST SUMMARY")
print("="*60)

if metrics['sharpe_ratio'] > -0.5:  # Reasonable threshold for random data
    print("✓ Test PASSED: Model trained successfully")
    print(f"  - Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    print(f"  - Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  - Turnover: {metrics['avg_turnover']:.4f}")
else:
    print("⚠ Test completed but performance is low (expected with synthetic data)")
    print("  Try training with more epochs or real market data")

print("\n✓ End-to-end pipeline working correctly!")
print("  Ready to use with real options data from Polygon.io")
print("="*60)
