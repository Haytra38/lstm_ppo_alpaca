import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


# Informations d'identification API Alpaca - CHARGEMENT SÉCURISÉ
API_KEY = os.environ.get('ALPACA_API_KEY', 'PKWDXUMR64LL03LXOZFN')  # Fallback pour compatibilité
API_SECRET = os.environ.get('ALPACA_API_SECRET', 'RLtrY3Idh1vQqNizfUJRoJprRJQ9uijfIoLLW4XA')  # Fallback pour compatibilité
BASE_URL = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')

# Initialiser l'API Alpaca
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

def get_historical_data(symbol, start, end, timeframe):
    """
    Récupère les données historiques depuis l'API Alpaca avec gestion d'erreurs.
    
    Args:
        symbol: Symbole de l'actif
        start: Date de début
        end: Date de fin  
        timeframe: Intervalle de temps
        
    Returns:
        pd.DataFrame: DataFrame avec les données OHLCV
    """
    try:
        is_crypto = any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'LTC', 'XRP', 'BCH', 'EOS', 'TRX']) or '-USD' in symbol.upper() or '/USD' in symbol.upper()
        
        if is_crypto:
            barset = api.get_crypto_bars(symbol, timeframe, start=start, end=end, limit=None).df
        else:
            barset = api.get_bars(symbol, timeframe, start=start, end=end, limit=None).df
            
        if barset.empty:
            print(f"⚠️ Aucune donnée récupérée pour {symbol} entre {start} et {end}")
            return pd.DataFrame()
            
        barset = barset.reset_index()
        df = pd.DataFrame({
            'Time': barset['timestamp'],
            'Open': barset['open'],
            'High': barset['high'],
            'Low': barset['low'],
            'Close': barset['close'],
            'Volume': barset.get('volume', 0)  # Volume avec valeur par défaut
        })
        df.set_index('Time', inplace=True)
        
        # Validation des données
        if df.isnull().any().any():
            print(f"⚠️ Données manquantes détectées pour {symbol}")
            df = df.fillna(method='ffill').fillna(method='bfill')  # Remplissage des valeurs manquantes
            
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des données pour {symbol}: {str(e)}")
        return pd.DataFrame()  # Retourner DataFrame vide en cas d'erreur



def get_stock_data(symbol, period='1y', interval='1m'):
    """
    Récupère les données historiques d'une action via l'API Alpaca
    
    Args:
        symbol (str): Symbole de l'action (ex: 'AAPL')
        period (str): Période des données ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval (str): Intervalle des données ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
        pd.DataFrame: DataFrame avec colonnes ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    try:
        # Convertir la période en dates de début et fin
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
            start_date = end_date - timedelta(days=7)  # Par défaut 7 jours
        
        # Convertir l'intervalle Alpaca
        alpaca_interval = interval
        if interval == '1m':
            alpaca_interval = '1Min'
        elif interval == '5m':
            alpaca_interval = '5Min'
        elif interval == '15m':
            alpaca_interval = '15Min'
        elif interval == '30m':
            alpaca_interval = '30Min'
        elif interval == '1h' or interval == '60m':
            alpaca_interval = '1Hour'
        elif interval == '1d':
            alpaca_interval = '1Day'
        
        # Détecter si c'est une cryptomonnaie
        is_crypto = any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'LTC', 'XRP', 'BCH', 'EOS', 'TRX']) or '-USD' in symbol.upper() or '/USD' in symbol.upper()
        
        # Récupérer les données via Alpaca
        if is_crypto:
            barset = api.get_crypto_bars(
                symbol,
                alpaca_interval,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                limit=None
            ).df
        else:
            barset = api.get_bars(
                symbol,
                alpaca_interval,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                limit=None
            ).df
        
        if barset.empty:
            print(f"Aucune donnée trouvée pour {symbol}")
            return pd.DataFrame()
        
        barset = barset.reset_index()
        
        # Créer le DataFrame au format attendu
        df = pd.DataFrame({
            'Open': barset['open'],
            'High': barset['high'],
            'Low': barset['low'],
            'Close': barset['close'],
            'Volume': barset['volume'] if 'volume' in barset.columns else 0
        })
        
        # Définir l'index temporel
        df.index = pd.to_datetime(barset['timestamp'])
        
        return df
        
    except Exception as e:
        print(f"Erreur lors de la récupération des données pour {symbol}: {e}")
        return pd.DataFrame()

def get_popular_stocks():
    """
    Retourne une liste d'actions populaires pour l'entraînement
    
    Returns:
        list: Liste des symboles d'actions populaires
    """
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'META', 'NVDA', 'NFLX', 'AMD', 'INTC',
        'BABA', 'CRM', 'ORCL', 'ADBE', 'PYPL',
        'UBER', 'LYFT', 'SPOT', 'TWTR', 'SNAP',
        'SPY', 'QQQ', 'IWM', 'GLD', 'SLV'
    ]

def get_available_data_files():
    """
    Retourne la liste des fichiers CSV disponibles dans le dossier data/
    
    Returns:
        list: Liste des noms de fichiers CSV disponibles
    """
    import os
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        return []
    
    csv_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            csv_files.append(file)
    
    return sorted(csv_files)

def load_data_from_file(filename):
    """
    Charge les données depuis un fichier CSV dans le dossier data/
    
    Args:
        filename (str): Nom du fichier CSV à charger
    
    Returns:
        pd.DataFrame: DataFrame avec les données chargées ou None si erreur
    """
    import os
    
    file_path = os.path.join("data", filename)
    
    if not os.path.exists(file_path):
        print(f"❌ Fichier non trouvé: {file_path}")
        return None
    
    try:
        # Charger le CSV avec l'index temporel
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Vérifier que les colonnes nécessaires sont présentes
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Colonnes manquantes dans {filename}: {missing_columns}")
            return None
        
        # Ajouter la colonne Volume si elle n'existe pas
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        
        print(f"✅ Données chargées depuis {filename}: {len(df)} périodes")
        print(f"📈 Période: {df.index[0]} à {df.index[-1]}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement de {filename}: {e}")
        return None

# Exemple d'utilisation (commenté pour éviter l'exécution automatique)
if __name__ == "__main__":
     # Définir les paramètres pour récupérer les données
     symbol = "BTC/USD"
     timeframe = "1Min"  # Choisir parmi "minute", "1M", "5M", "15M", "day", "1D"
     start_date = "2021-01-01"  # Date de début des données
     end_date = "2025-11-14"  # Date de fin des données
     
     # Récuperation des données
     df = get_historical_data(symbol, start_date, end_date, timeframe)
     
     # sauvegarde des donnés dans fichier .csv
     nom_sauvegarde = f"data/{symbol.replace('/', '-')}{timeframe}_{start_date}_{end_date}.csv"
     df.to_csv(nom_sauvegarde, index=True)
     print (df)
