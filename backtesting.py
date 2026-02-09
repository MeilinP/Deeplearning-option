"""
Backtesting framework for options trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

sns.set_style("darkgrid")


class Backtester:

    def __init__(self,
                 initial_capital: float = 100000.0,
                 transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self.portfolio_values = []
        self.positions = []
        self.trades = []

    def run(self,
            signals: np.ndarray,
            returns: np.ndarray,
            dates: Optional[np.ndarray] = None) -> Dict:
        n_samples = len(signals)

        if dates is None:
            dates = np.arange(n_samples)

        portfolio_value = self.initial_capital
        position = 0.0
        cash = self.initial_capital

        portfolio_values = [portfolio_value]
        positions = [position]
        portfolio_returns = []
        turnover_history = []

        for t in range(n_samples):
            target_position = signals[t]

            turnover = abs(target_position - position)
            turnover_history.append(turnover)

            cost = turnover * self.transaction_cost * portfolio_value
            cost += turnover * self.slippage * portfolio_value

            position = target_position

            portfolio_return = position * returns[t] - cost / portfolio_value
            portfolio_returns.append(portfolio_return)

            portfolio_value *= (1 + portfolio_return)

            portfolio_values.append(portfolio_value)
            positions.append(position)

        portfolio_returns = np.array(portfolio_returns)

        metrics = {
            'total_return': (portfolio_value - self.initial_capital) / self.initial_capital,
            'sharpe_ratio': self._calculate_sharpe(portfolio_returns),
            'max_drawdown': self._calculate_max_drawdown(portfolio_values),
            'volatility': np.std(portfolio_returns) * np.sqrt(252),
            'mean_return': np.mean(portfolio_returns) * 252,
            'total_turnover': np.sum(turnover_history),
            'avg_turnover': np.mean(turnover_history),
            'final_value': portfolio_value,
            'n_trades': np.sum(np.array(turnover_history) > 0.01)
        }

        self.portfolio_values = portfolio_values
        self.positions = positions
        self.returns = portfolio_returns
        self.dates = dates

        return metrics

    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        if len(returns) == 0:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        return (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        values = np.array(portfolio_values)
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        return np.min(drawdown)

    def plot_results(self, title: str = "Backtest Results", figsize: Tuple[int, int] = (15, 10)):
        fig, axes = plt.subplots(3, 1, figsize=figsize)

        axes[0].plot(self.dates, self.portfolio_values[1:], label='Portfolio Value', linewidth=2)
        axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        axes[0].set_title(f'{title} - Portfolio Value Over Time')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.dates, self.returns, label='Daily Returns', alpha=0.7)
        axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1].set_title('Daily Returns')
        axes[1].set_ylabel('Return')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(self.dates, self.positions[1:], label='Position Size', linewidth=2)
        axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[2].set_title('Trading Signals / Positions')
        axes[2].set_ylabel('Position')
        axes[2].set_xlabel('Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class BaselineStrategies:

    @staticmethod
    def buy_and_hold(n_samples: int) -> np.ndarray:
        return np.ones(n_samples)

    @staticmethod
    def random_trading(n_samples: int, seed: int = 42) -> np.ndarray:
        np.random.seed(seed)
        return np.random.uniform(-1, 1, n_samples)

    @staticmethod
    def momentum(returns: np.ndarray, lookback: int = 20) -> np.ndarray:
        signals = np.zeros(len(returns))
        for i in range(lookback, len(returns)):
            recent_return = np.mean(returns[i-lookback:i])
            signals[i] = np.tanh(recent_return * 50)
        return signals


def compare_strategies(strategies: Dict[str, np.ndarray],
                      returns: np.ndarray,
                      transaction_cost: float = 0.001,
                      dates: Optional[np.ndarray] = None) -> pd.DataFrame:
    results = []

    for name, signals in strategies.items():
        backtester = Backtester(transaction_cost=transaction_cost)
        metrics = backtester.run(signals, returns, dates)
        metrics['strategy'] = name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df.set_index('strategy')

    return df


def plot_strategy_comparison(comparison_df: pd.DataFrame, figsize: Tuple[int, int] = (15, 6)):
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    comparison_df['sharpe_ratio'].plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Sharpe Ratio Comparison')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    comparison_df['total_return'].plot(kind='bar', ax=axes[1], color='green')
    axes[1].set_title('Total Return Comparison')
    axes[1].set_ylabel('Total Return')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    comparison_df['max_drawdown'].plot(kind='bar', ax=axes[2], color='crimson')
    axes[2].set_title('Maximum Drawdown Comparison')
    axes[2].set_ylabel('Max Drawdown')
    axes[2].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    return fig


def print_metrics_table(metrics: Dict, title: str = "Backtest Metrics"):
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")

    print(f"\n{'Metric':<25} {'Value':>15}")
    print(f"{'-'*50}")

    formatted_metrics = {
        'Total Return': f"{metrics['total_return']*100:>14.2f}%",
        'Sharpe Ratio': f"{metrics['sharpe_ratio']:>15.3f}",
        'Annualized Return': f"{metrics['mean_return']*100:>14.2f}%",
        'Annualized Volatility': f"{metrics['volatility']*100:>14.2f}%",
        'Max Drawdown': f"{metrics['max_drawdown']*100:>14.2f}%",
        'Total Turnover': f"{metrics['total_turnover']:>15.2f}",
        'Average Turnover': f"{metrics['avg_turnover']:>15.4f}",
        'Number of Trades': f"{int(metrics['n_trades']):>15}",
        'Final Portfolio Value': f"${metrics['final_value']:>14,.2f}",
    }

    for metric, value in formatted_metrics.items():
        print(f"{metric:<25} {value}")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    print("Testing backtesting framework...")

    np.random.seed(42)
    n_samples = 252
    returns = np.random.randn(n_samples) * 0.02
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')

    print("\n1. Buy & Hold")
    signals = BaselineStrategies.buy_and_hold(n_samples)
    backtester = Backtester(initial_capital=100000, transaction_cost=0.001)
    metrics = backtester.run(signals, returns, dates)
    print_metrics_table(metrics, "Buy & Hold Strategy")

    print("\n2. Random Trading")
    signals = BaselineStrategies.random_trading(n_samples)
    backtester = Backtester(initial_capital=100000, transaction_cost=0.001)
    metrics = backtester.run(signals, returns, dates)
    print_metrics_table(metrics, "Random Trading Strategy")

    print("\n3. Momentum")
    signals = BaselineStrategies.momentum(returns, lookback=20)
    backtester = Backtester(initial_capital=100000, transaction_cost=0.001)
    metrics = backtester.run(signals, returns, dates)
    print_metrics_table(metrics, "Momentum Strategy")

    print("\n4. Strategy Comparison")
    strategies = {
        'Buy & Hold': BaselineStrategies.buy_and_hold(n_samples),
        'Random': BaselineStrategies.random_trading(n_samples),
        'Momentum': BaselineStrategies.momentum(returns)
    }

    comparison = compare_strategies(strategies, returns, transaction_cost=0.001, dates=dates)
    print("\nComparison Table:")
    print(comparison[['sharpe_ratio', 'total_return', 'max_drawdown', 'volatility']].round(3))

    print("\nBacktesting framework working correctly!")
