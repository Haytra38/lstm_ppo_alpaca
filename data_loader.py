import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()


class DataLoader:
    def __init__(self, api_key=None, api_secret=None, base_url=None):
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.api_secret = api_secret or os.environ.get('ALPACA_API_SECRET')
        self.base_url = base_url or os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.api = None

    def _ensure_api(self):
        if self.api is None:
            if not self.api_key or not self.api_secret:
                raise RuntimeError('ALPACA_API_KEY/ALPACA_API_SECRET manquants')
            self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url, api_version='v2')

    def get_historical_data(self, symbol, start, end, timeframe):
        try:
            self._ensure_api()
            is_crypto = any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'LTC', 'XRP', 'BCH', 'EOS', 'TRX']) or '-USD' in symbol.upper() or '/USD' in symbol.upper()
            if is_crypto:
                barset = self.api.get_crypto_bars(symbol, timeframe, start=start, end=end, limit=None).df
            else:
                barset = self.api.get_bars(symbol, timeframe, start=start, end=end, limit=None).df
            if barset.empty:
                print(f"Aucune donnée renvoyée ({symbol}, {timeframe}, {start} -> {end})")
                return pd.DataFrame()
            barset = barset.reset_index()
            df = pd.DataFrame({
                'Time': barset['timestamp'],
                'Open': barset['open'],
                'High': barset['high'],
                'Low': barset['low'],
                'Close': barset['close'],
                'Volume': barset['volume'] if 'volume' in barset.columns else 0
            })
            df.set_index('Time', inplace=True)
            if df.isnull().any().any():
                df = df.fillna(method='ffill').fillna(method='bfill')
            return df
        except Exception as e:
            print(f"Erreur lors de l'appel API Alpaca: {e}")
            return pd.DataFrame()

    def get_stock_data(self, symbol, period='1y', interval='1m'):
        try:
            self._ensure_api()
            end_date = datetime.now()
            if period == '1d':
                start_date = end_date - timedelta(days=1)
            elif period == '2d':
                start_date = end_date - timedelta(days=2)
            elif period == '3d':
                start_date = end_date - timedelta(days=3)
            elif period == '5d':
                start_date = end_date - timedelta(days=5)
            elif period == '7d':
                start_date = end_date - timedelta(days=7)
            elif period == '1mo':
                start_date = end_date - timedelta(days=30)
            elif period == '3mo':
                start_date = end_date - timedelta(days=90)
            elif period == '6mo':
                start_date = end_date - timedelta(days=180)
            elif period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '2y':
                start_date = end_date - timedelta(days=730)
            elif period == '5y':
                start_date = end_date - timedelta(days=1825)
            else:
                start_date = end_date - timedelta(days=7)
            alpaca_interval = interval
            if interval == '1m':
                alpaca_interval = '1Min'
            elif interval == '5m':
                alpaca_interval = '5Min'
            elif interval == '15m':
                alpaca_interval = '15Min'
            elif interval == '30m':
                alpaca_interval = '30Min'
            elif interval in ('1h', '60m'):
                alpaca_interval = '1Hour'
            elif interval == '1d':
                alpaca_interval = '1Day'
            is_crypto = any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'LTC', 'XRP', 'BCH', 'EOS', 'TRX']) or '-USD' in symbol.upper() or '/USD' in symbol.upper()
            if is_crypto:
                barset = self.api.get_crypto_bars(
                    symbol,
                    alpaca_interval,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    limit=None
                ).df
            else:
                barset = self.api.get_bars(
                    symbol,
                    alpaca_interval,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    limit=None
                ).df
            if barset.empty:
                print(f"Aucune donnée renvoyée ({symbol}, {alpaca_interval}, {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')})")
                return pd.DataFrame()
            barset = barset.reset_index()
            df = pd.DataFrame({
                'Open': barset['open'],
                'High': barset['high'],
                'Low': barset['low'],
                'Close': barset['close'],
                'Volume': barset['volume'] if 'volume' in barset.columns else 0
            })
            df.index = pd.to_datetime(barset['timestamp'])
            return df
        except Exception as e:
            print(f"Erreur lors de l'appel API Alpaca: {e}")
            return pd.DataFrame()

    def load_stock_data(self, ticker, start_date, end_date, use_alpaca=True, interval='1Min'):
        df = self.get_historical_data(ticker, start_date, end_date, interval)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        # Harmoniser l'ordre des colonnes
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        return df[cols]

    def get_popular_stocks(self):
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
            'META', 'NVDA', 'NFLX', 'AMD', 'INTC',
            'BABA', 'CRM', 'ORCL', 'ADBE', 'PYPL',
            'UBER', 'LYFT', 'SPOT', 'TWTR', 'SNAP',
            'SPY', 'QQQ', 'IWM', 'GLD', 'SLV'
        ]

    def get_available_data_files(self, data_dir='data'):
        if not os.path.exists(data_dir):
            return []
        csv_files = []
        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                csv_files.append(file)
        return sorted(csv_files)

    def load_data_from_file(self, filename, data_dir='data'):
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            return None
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return None
            if 'Volume' not in df.columns:
                df['Volume'] = 0
            return df
        except Exception:
            return None


def get_historical_data(symbol, start, end, timeframe):
    return DataLoader().get_historical_data(symbol, start, end, timeframe)


def get_stock_data(symbol, period='1y', interval='1m'):
    return DataLoader().get_stock_data(symbol, period, interval)


def get_popular_stocks():
    return DataLoader().get_popular_stocks()


def get_available_data_files():
    return DataLoader().get_available_data_files()


def load_data_from_file(filename):
    return DataLoader().load_data_from_file(filename)

if __name__ == "__main__":
     symbol = "BTC/USD"
     timeframe = "1Min"
     start_date = "2021-01-01"
     end_date = "2025-11-18"

     api_key_env = os.environ.get("ALPACA_API_KEY")
     api_secret_env = os.environ.get("ALPACA_API_SECRET")
     placeholders = {"your_alpaca_api_key_here", "your_alpaca_api_secret_here", ""}
     if api_key_env in placeholders or api_secret_env in placeholders or api_key_env is None or api_secret_env is None:
         print("ALPACA_API_KEY/ALPACA_API_SECRET non définis ou invalides")
         raise SystemExit(1)

     dl = DataLoader()
     df = dl.get_historical_data(symbol, start_date, end_date, timeframe)

     if df is None or df.empty:
         print("Aucune donnée récupérée via l'API Alpaca")
         raise SystemExit(2)

     os.makedirs("data", exist_ok=True)
     output_path = f"data/{symbol.replace('/', '-')}{timeframe}_{start_date}_{end_date}.csv"
     df.to_csv(output_path, index=True)
     print(output_path)