"""
Data utilities for fetching and processing options data from Polygon.io
"""

import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from polygon import RESTClient
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class PolygonOptionsDataLoader:

    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data_cache"):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("Polygon API key not found. Set POLYGON_API_KEY environment variable.")

        self.client = RESTClient(self.api_key)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_options_chain(self,
                         ticker: str,
                         start_date: str,
                         end_date: str,
                         option_type: str = "both",
                         force_refresh: bool = False) -> pd.DataFrame:
        cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}_{option_type}.pkl"

        if cache_file.exists() and not force_refresh:
            print(f"Loading cached data for {ticker}")
            with open(cache_file, 'rb') as f:
                cached_df = pickle.load(f)
                if not cached_df.empty:
                    return cached_df
                else:
                    print(f"Cached data for {ticker} is empty. Re-fetching...")

        print(f"Fetching options data for {ticker} from {start_date} to {end_date}")

        options_data = []

        try:
            contract_results = self.client.list_options_contracts(
                underlying_ticker=ticker,
                expiration_date_gte=start_date,
                expiration_date_lte=end_date,
                limit=1000
            )

            has_results = False
            for contract in tqdm(contract_results, desc=f"Processing {ticker} options"):
                has_results = True
                if option_type != "both":
                    if contract.contract_type.lower() != option_type:
                        continue

                options_data.append({
                    'ticker': contract.ticker,
                    'underlying_ticker': contract.underlying_ticker,
                    'strike_price': contract.strike_price,
                    'expiration_date': contract.expiration_date,
                    'contract_type': contract.contract_type,
                })

            if not has_results:
                print(f"No options contracts found for {ticker} in range {start_date} to {end_date}.")
                print("Note: Dates in list_options_contracts refer to expiration dates.")

        except Exception as e:
            print(f"Error fetching options data: {e}")

        df = pd.DataFrame(options_data)

        if not df.empty:
            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)
            print(f"Cached {len(df)} contracts to {cache_file}")

        return df

    def get_option_prices(self,
                         option_ticker: str,
                         start_date: str,
                         end_date: str) -> pd.DataFrame:
        cache_file = self.cache_dir / f"prices_{option_ticker}_{start_date}_{end_date}.pkl"

        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        try:
            aggs = self.client.get_aggs(
                ticker=option_ticker,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date
            )

            prices_data = []
            for agg in aggs:
                prices_data.append({
                    'date': datetime.fromtimestamp(agg.timestamp / 1000),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })

            df = pd.DataFrame(prices_data)

            with open(cache_file, 'wb') as f:
                pickle.dump(df, f)

            return df

        except Exception as e:
            print(f"Error fetching price data: {e}")
            return pd.DataFrame()


class OptionsFeatureEngineering:

    @staticmethod
    def compute_moneyness(strike_price: float, spot_price: float) -> float:
        """log(strike / spot)"""
        if spot_price <= 0:
            return 0.0
        return np.log(strike_price / spot_price)

    @staticmethod
    def compute_time_to_maturity(expiration_date: datetime, current_date: datetime) -> float:
        days = (expiration_date - current_date).days
        return days / 365.0

    @staticmethod
    def compute_returns(prices: np.ndarray) -> np.ndarray:
        return np.diff(prices) / prices[:-1]

    @staticmethod
    def compute_volatility(returns: np.ndarray, window: int = 20) -> float:
        """EWMA standard deviation"""
        if len(returns) == 0:
            return 0.0
        series = pd.Series(returns)
        ewm_std = series.ewm(span=window, min_periods=1).std()
        result = ewm_std.iloc[-1]
        return result if not np.isnan(result) else np.std(returns)

    @staticmethod
    def create_straddle_features(call_data: Dict, put_data: Dict, spot_price: float) -> Dict:
        strike = call_data.get('strike_price', 0)
        expiration = call_data.get('expiration_date')
        current_date = datetime.now()

        features = {
            'moneyness': OptionsFeatureEngineering.compute_moneyness(strike, spot_price),
            'time_to_maturity': OptionsFeatureEngineering.compute_time_to_maturity(
                expiration, current_date
            ) if expiration else 0.0,
            'strike_price': strike,
            'spot_price': spot_price,
            'call_price': call_data.get('price', 0),
            'put_price': put_data.get('price', 0),
            'straddle_cost': call_data.get('price', 0) + put_data.get('price', 0),
        }

        if 'implied_volatility' in call_data:
            features['call_iv'] = call_data['implied_volatility']
        if 'implied_volatility' in put_data:
            features['put_iv'] = put_data['implied_volatility']

        return features


class DataPreprocessor:

    def __init__(self):
        self.feature_mean = None
        self.feature_std = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0) + 1e-8
        return (X - self.feature_mean) / self.feature_std

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        return (X - self.feature_mean) / self.feature_std

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("Preprocessor not fitted.")
        return X * self.feature_std + self.feature_mean


def train_test_split_temporal(data: pd.DataFrame,
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]
    test_df = data.iloc[val_end:]

    return train_df, val_df, test_df


def generate_synthetic_options_data(n_samples: int = 1000,
                                    n_features: int = 10,
                                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)

    X = np.random.randn(n_samples, n_features)

    # Returns have some correlation to first few features
    weights = np.random.randn(min(5, n_features))
    base_returns = X[:, :len(weights)] @ weights
    noise = np.random.randn(n_samples) * 0.5
    returns = base_returns + noise

    returns = returns / (np.std(returns) + 1e-8) * 0.02  # ~2% daily vol

    return X, returns


if __name__ == "__main__":
    print("Testing synthetic data generation...")
    X, returns = generate_synthetic_options_data(n_samples=100)
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    print(f"Returns shape: {returns.shape}")
    print(f"Returns mean: {np.mean(returns):.4f}, std: {np.std(returns):.4f}")

    print("\nTesting feature engineering...")
    call_data = {
        'strike_price': 100,
        'price': 5.0,
        'expiration_date': datetime.now() + timedelta(days=30),
        'implied_volatility': 0.25
    }
    put_data = {
        'strike_price': 100,
        'price': 4.5,
        'expiration_date': datetime.now() + timedelta(days=30),
        'implied_volatility': 0.23
    }

    features = OptionsFeatureEngineering.create_straddle_features(
        call_data, put_data, spot_price=102
    )
    print("Straddle features:", features)

    print("\nData utilities working correctly!")
