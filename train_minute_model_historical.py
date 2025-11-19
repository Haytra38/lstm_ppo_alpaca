import numpy as np
import pandas as pd
import os
import json
import argparse
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
from data_loader import get_stock_data, get_popular_stocks, get_available_data_files, load_data_from_file
from trade_history_tracker import TradeHistoryTracker


class MinuteTradingEnvHistorical(gym.Env):
    """
    Environnement de trading optimisé pour les données minute avec analyse historique des 30 périodes précédentes
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, initial_balance=1000.0, lookback_periods=30):
        super(MinuteTradingEnvHistorical, self).__init__()
        
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.lookback_periods = lookback_periods
        self.current_step = lookback_periods  # Commencer après la période de lookback
        self.max_steps = len(df) - 1
        
        # Espace d'actions: 0=Hold, 1=Long, 2=Short
        self.action_space = spaces.Discrete(3)
        
        # Espace d'observation étendu pour inclure les données historiques
        # 5 features OHLCV par période * lookback_periods (uniquement les prix OHLCV)
        obs_size = 5 * lookback_periods
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Variables de trading - Système de positions multiples
        self.balance = initial_balance
        self.max_position_percentage = 0.10  # 10% du capital par position
        self.max_positions = 10  # Maximum 10 positions simultanées
        
        # Liste des positions ouvertes
        self.positions = []  # Chaque position: {'type': 1/-1, 'size': float, 'entry_price': float, 'stop_loss': float, 'take_profit': float, 'id': int}
        self.next_position_id = 1
        
        # Variables de compatibilité (pour les métriques)
        self.position = 0  # Sera calculé dynamiquement
        self.position_size = 0  # Sera calculé dynamiquement
        self.entry_price = 0  # Prix moyen des positions
        self.stop_loss = 0
        self.take_profit = 0
        
        # Statistiques
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
        self.peak_balance = initial_balance
        
        # Historique pour le calcul des métriques
        self.balance_history = []
        self.action_history = []
        self.price_history = []
        
        print(f"🔍 Environnement initialisé avec analyse historique de {lookback_periods} périodes")
        print(f"📊 Taille de l'observation: {obs_size} features")
        print(f"📈 Données disponibles: {len(df)} périodes")
        print(f"🎯 Période d'entraînement: {len(df) - lookback_periods} steps")
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Réinitialiser les variables - commencer à lookback_periods pour avoir des données historiques complètes
        self.current_step = self.lookback_periods
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        
        # Réinitialiser les statistiques
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
        self.peak_balance = self.initial_balance
        
        # Réinitialiser l'historique
        self.balance_history = [self.initial_balance]
        self.action_history = []
        self.price_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """
        Retourne l'observation avec les données OHLCV des 30 périodes précédentes
        L'agent commence ses observations à la période 30 pour avoir 30 vraies valeurs historiques
        """
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1
        
        # S'assurer qu'on a au moins lookback_periods de données historiques
        if self.current_step < self.lookback_periods:
            # Retourner une observation vide si on n'a pas assez de données
            # Cela ne devrait pas arriver car on commence à la période lookback_periods
            obs_size = 5 * self.lookback_periods
            return np.zeros(obs_size, dtype=np.float32)
        
        # Prendre les lookback_periods dernières périodes
        start_idx = self.current_step - self.lookback_periods + 1
        end_idx = self.current_step + 1
        historical_data = self.df.iloc[start_idx:end_idx]
        
        # Normaliser les données par rapport au prix de clôture actuel
        current_price = self.df.iloc[self.current_step]['Close']
        
        obs_features = []
        
        # Pour chaque période historique, ajouter les features OHLCV normalisées
        for i in range(len(historical_data)):
            row = historical_data.iloc[i]
            
            if current_price > 0:  # Éviter la division par zéro
                # Prix OHLC normalisés par rapport au prix de clôture actuel
                open_norm = (row['Open'] - current_price) / current_price if row['Open'] > 0 else 0
                high_norm = (row['High'] - current_price) / current_price if row['High'] > 0 else 0
                low_norm = (row['Low'] - current_price) / current_price if row['Low'] > 0 else 0
                close_norm = (row['Close'] - current_price) / current_price if row['Close'] > 0 else 0
                
                # Volume normalisé (log pour réduire l'échelle)
                volume_norm = np.log(row['Volume'] + 1) / 20 if row['Volume'] > 0 else 0
            else:
                open_norm = high_norm = low_norm = close_norm = volume_norm = 0
            
            obs_features.extend([open_norm, high_norm, low_norm, close_norm, volume_norm])
        
        obs = np.array(obs_features, dtype=np.float32)
        
        # Remplacer les NaN et inf par des valeurs sûres
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        
        return obs
    
    def _calculate_position_size(self, price):
        """
        Calcule la taille de position basée sur 10% du capital disponible
        Permet d'ouvrir plusieurs positions simultanées
        """
        if price <= 0:
            return 0
        
        # Utiliser 10% du capital total pour cette position
        position_capital = self.balance * self.max_position_percentage
        position_size = position_capital / price
        
        return max(0, position_size)
    
    def _get_open_positions_count(self):
        """Retourne le nombre de positions ouvertes"""
        return len(self.positions)
    
    def _can_open_new_position(self):
        """Vérifie si on peut ouvrir une nouvelle position"""
        return len(self.positions) < self.max_positions
    
    def _update_compatibility_variables(self):
        """Met à jour les variables de compatibilité pour les métriques"""
        if not self.positions:
            self.position = 0
            self.position_size = 0
            self.entry_price = 0
        else:
            # Calculer la position nette (somme des types de positions)
            net_position = sum(pos['type'] for pos in self.positions)
            self.position = 1 if net_position > 0 else (-1 if net_position < 0 else 0)
            
            # Calculer la taille totale des positions
            self.position_size = sum(pos['size'] for pos in self.positions)
            
            # Calculer le prix d'entrée moyen pondéré
            if self.positions:
                total_value = sum(pos['size'] * pos['entry_price'] for pos in self.positions)
                self.entry_price = total_value / self.position_size if self.position_size > 0 else 0
    
    def _get_market_conditions(self):
        """
        Analyse les conditions de marché basées sur les données historiques
        """
        if self.current_step < self.lookback_periods:
            return {
                'trend': 'neutral',
                'volatility': 'low',
                'volume_trend': 'normal'
            }
        
        # Analyser les 10 dernières périodes pour déterminer la tendance
        recent_data = self.df.iloc[max(0, self.current_step-9):self.current_step+1]
        
        if len(recent_data) < 2:
            return {
                'trend': 'neutral',
                'volatility': 'low',
                'volume_trend': 'normal'
            }
        
        # Tendance des prix
        price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
        if price_change > 0.02:  # +2%
            trend = 'bullish'
        elif price_change < -0.02:  # -2%
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Volatilité (écart-type des rendements)
        returns = recent_data['Close'].pct_change().dropna()
        volatility = 'high' if returns.std() > 0.02 else 'low'
        
        # Tendance du volume
        initial_volume = recent_data['Volume'].iloc[0]
        if initial_volume > 0:
            volume_change = (recent_data['Volume'].iloc[-1] - initial_volume) / initial_volume
            volume_trend = 'increasing' if volume_change > 0.1 else 'decreasing' if volume_change < -0.1 else 'normal'
        else:
            volume_trend = 'normal'
        
        return {
            'trend': trend,
            'volatility': volatility,
            'volume_trend': volume_trend
        }
    
    def step(self, action):
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0, True, True, {}
        
        current_price = self.df.iloc[self.current_step]['Open']
        reward = 0
        info = {}
        
        # Enregistrer l'action et le prix
        self.action_history.append(action)
        self.price_history.append(current_price)
        
        # Vérifier et fermer les positions si nécessaire (stop-loss/take-profit)
        position_closed, close_reward = self._check_and_close_positions()
        reward += close_reward
        
        # Exécuter l'action (positions multiples possibles)
        if action == 1 and self._can_open_new_position():  # Ouvrir position longue
            position_size = self._calculate_position_size(current_price)
            if position_size > 0:
                new_position = {
                    'type': 1,
                    'size': position_size,
                    'entry_price': current_price,
                    'stop_loss': current_price * 0.95,  # Stop-loss à -5%
                    'take_profit': current_price * 1.10,  # Take-profit à +10%
                    'id': self.next_position_id
                }
                self.positions.append(new_position)
                self.next_position_id += 1
                self.total_trades += 1
                reward += 0.01  # Petite récompense pour ouvrir une position
                info['action_taken'] = f'open_long_{new_position["id"]}'
                info['positions_count'] = len(self.positions)
        
        elif action == 2 and self._can_open_new_position():  # Ouvrir position courte
            position_size = self._calculate_position_size(current_price)
            if position_size > 0:
                new_position = {
                    'type': -1,
                    'size': position_size,
                    'entry_price': current_price,
                    'stop_loss': current_price * 1.05,  # Stop-loss à +5%
                    'take_profit': current_price * 0.90,  # Take-profit à -10%
                    'id': self.next_position_id
                }
                self.positions.append(new_position)
                self.next_position_id += 1
                self.total_trades += 1
                reward += 0.01  # Petite récompense pour ouvrir une position
                info['action_taken'] = f'open_short_{new_position["id"]}'
                info['positions_count'] = len(self.positions)
        
        elif action == 0 and self.positions:  # Fermer toutes les positions manuellement
            total_pnl = self._calculate_unrealized_pnl(current_price)
            self.balance += total_pnl
            
            if total_pnl > 0:
                self.winning_trades += len(self.positions)
                reward += total_pnl / self.initial_balance * 10  # Récompense proportionnelle au profit
            else:
                self.losing_trades += len(self.positions)
                reward += total_pnl / self.initial_balance * 5  # Pénalité moins sévère pour les pertes
            
            self.total_profit += total_pnl
            closed_positions = len(self.positions)
            self.positions.clear()  # Fermer toutes les positions
            info['action_taken'] = f'close_all_positions_{closed_positions}'
        
        # Mettre à jour les variables de compatibilité
        self._update_compatibility_variables()
        
        # Calculer la récompense basée sur la performance
        unrealized_pnl = self._calculate_unrealized_pnl(current_price) if self.position != 0 else 0
        total_value = self.balance + unrealized_pnl
        
        # Récompense basée sur le changement de valeur du portefeuille
        if len(self.balance_history) > 0:
            portfolio_change = (total_value - self.balance_history[-1]) / self.initial_balance
            reward += portfolio_change * 100  # Amplifier la récompense
        
        # Mettre à jour l'historique
        self.balance_history.append(total_value)
        
        # Calculer le drawdown
        if total_value > self.peak_balance:
            self.peak_balance = total_value
        current_drawdown = (self.peak_balance - total_value) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Pénalité pour drawdown excessif
        if current_drawdown > 0.1:  # Plus de 10% de drawdown
            reward -= current_drawdown * 10
        
        # Passer au step suivant
        self.current_step += 1
        
        # Vérifier si l'épisode est terminé
        done = self.current_step >= self.max_steps or total_value <= self.initial_balance * 0.5
        truncated = False
        
        # Informations supplémentaires
        info.update({
            'balance': self.balance,
            'position': self.position,
            'total_value': total_value,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'total_profit': self.total_profit,
            'max_drawdown': self.max_drawdown,
            'current_price': current_price,
            'market_conditions': self._get_market_conditions()
        })
        
        return self._get_observation(), reward, done, truncated, info
    
    def _calculate_unrealized_pnl(self, current_price):
        """Calcule le PnL non réalisé de toutes les positions ouvertes"""
        if not self.positions:
            return 0
        
        total_pnl = 0
        for position in self.positions:
            if position['type'] == 1:  # Position longue
                pnl = position['size'] * (current_price - position['entry_price'])
            else:  # Position courte
                pnl = position['size'] * (position['entry_price'] - current_price)
            total_pnl += pnl
        
        return total_pnl
    
    def _check_and_close_positions(self):
        """Vérifie et ferme les positions selon les règles de stop-loss et take-profit"""
        if not self.positions:
            return False, 0
        
        current_price = self.df.iloc[self.current_step]['Close']
        positions_closed = False
        total_reward = 0
        positions_to_remove = []
        
        for i, position in enumerate(self.positions):
            position_closed = False
            
            if position['type'] == 1:  # Position longue
                if current_price <= position['stop_loss']:
                    # Stop-loss déclenché
                    pnl = position['size'] * (current_price - position['entry_price'])
                    self.balance += pnl
                    self.total_profit += pnl
                    self.losing_trades += 1
                    total_reward += pnl / self.initial_balance * 5  # Pénalité pour stop-loss
                    position_closed = True
                elif current_price >= position['take_profit']:
                    # Take-profit déclenché
                    pnl = position['size'] * (current_price - position['entry_price'])
                    self.balance += pnl
                    self.total_profit += pnl
                    self.winning_trades += 1
                    total_reward += pnl / self.initial_balance * 15  # Récompense pour take-profit
                    position_closed = True
            
            else:  # Position courte
                if current_price >= position['stop_loss']:
                    # Stop-loss déclenché
                    pnl = position['size'] * (position['entry_price'] - current_price)
                    self.balance += pnl
                    self.total_profit += pnl
                    self.losing_trades += 1
                    total_reward += pnl / self.initial_balance * 5  # Pénalité pour stop-loss
                    position_closed = True
                elif current_price <= position['take_profit']:
                    # Take-profit déclenché
                    pnl = position['size'] * (position['entry_price'] - current_price)
                    self.balance += pnl
                    self.total_profit += pnl
                    self.winning_trades += 1
                    total_reward += pnl / self.initial_balance * 15  # Récompense pour take-profit
                    position_closed = True
            
            if position_closed:
                positions_to_remove.append(i)
                positions_closed = True
        
        # Supprimer les positions fermées (en ordre inverse pour éviter les problèmes d'index)
        for i in reversed(positions_to_remove):
            self.positions.pop(i)
        
        # Mettre à jour les variables de compatibilité
        self._update_compatibility_variables()
        
        return positions_closed, total_reward
    
    def _update_portfolio_value(self):
        """Met à jour la valeur totale du portefeuille"""
        current_price = self.df.iloc[self.current_step]['Close']
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        return self.balance + unrealized_pnl

def load_model_metadata(model_name):
    """
    Charge les métadonnées d'un modèle sauvegardé
    """
    metadata_path = f"models/{model_name}_metadata.json"
    
    if not os.path.exists(metadata_path):
        print(f"⚠️ Fichier de métadonnées non trouvé: {metadata_path}")
        return None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        print(f"❌ Erreur lors du chargement des métadonnées: {e}")
        return None

def load_existing_model(model_name):
    """
    Charge un modèle existant depuis le dossier models/
    """
    model_path = f"models/{model_name}"
    
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ Le modèle {model_name} n'existe pas dans le dossier models/")
        return None, None
    
    try:
        print(f"📂 Chargement du modèle {model_name}...")
        model = PPO.load(model_path)
        
        # Charger les métadonnées pour récupérer les lookback_periods
        metadata = load_model_metadata(model_name)
        lookback_periods = None
        if metadata:
            lookback_periods = metadata.get('lookback_periods', 30)
            print(f"📋 Métadonnées chargées - Périodes d'analyse: {lookback_periods}")
        else:
            print(f"⚠️ Métadonnées non trouvées, utilisation de la valeur par défaut (30)")
            lookback_periods = 30
        
        print(f"✅ Modèle {model_name} chargé avec succès")
        return model, lookback_periods
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle {model_name}: {e}")
        return None, None

def list_available_models():
    """
    Liste tous les modèles disponibles dans le dossier models/
    """
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.zip'):
            models.append(file.replace('.zip', ''))
    
    return sorted(models)

def train_minute_model_historical(symbol="BTC-USD", period="7d", interval="1m", timesteps=50000, 
                                 model_name=None, continue_training=False, lookback_periods=30, filename=None, use_file=False):
    """
    Entraîne un modèle PPO pour le trading minute avec analyse historique
    
    Args:
        symbol: Symbole à trader (ex: BTC-USD, AAPL)
        period: Période des données (ex: 1d, 5d, 1mo)
        interval: Intervalle des données (ex: 1m, 5m, 1h)
        timesteps: Nombre de timesteps pour l'entraînement
        model_name: Nom du modèle à sauvegarder/charger
        continue_training: Continuer l'entraînement d'un modèle existant
        lookback_periods: Nombre de périodes historiques à analyser
        filename: Nom du fichier CSV à charger (optionnel)
        use_file: Utiliser un fichier local au lieu de télécharger
    """
    print(f"\n🚀 Démarrage de l'entraînement avec analyse historique")
    
    if use_file and filename:
        print(f"📁 Fichier: {filename}")
        # Charger les données depuis le fichier
        print(f"\n📥 Chargement des données depuis {filename}...")
        try:
            df = load_data_from_file(filename)
            if df is None or df.empty:
                print(f"❌ Impossible de charger les données depuis {filename}")
                return None
            
            print(f"✅ Données chargées: {len(df)} périodes")
            print(f"📈 Période: {df.index[0]} à {df.index[-1]}")
            
            # Extraire le symbole depuis le nom du fichier pour les métadonnées
            symbol = filename.split('_')[0].replace('-', '/')
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return None
    else:
        print(f"📊 Symbole: {symbol}")
        print(f"📅 Période: {period}")
        print(f"⏱️ Intervalle: {interval}")
        
        # Télécharger les données
        print(f"\n📥 Téléchargement des données pour {symbol}...")
        try:
            df = get_stock_data(symbol, period=period, interval=interval)
            if df is None or df.empty:
                print(f"❌ Impossible de récupérer les données pour {symbol}")
                return None
            
            print(f"✅ Données récupérées: {len(df)} périodes")
            print(f"📈 Période: {df.index[0]} à {df.index[-1]}")
            
        except Exception as e:
            print(f"❌ Erreur lors du téléchargement: {e}")
            return None
    
    print(f"🔍 Périodes d'analyse: {lookback_periods}")
    print(f"🎯 Timesteps: {timesteps:,}")
    
    # Vérifier qu'on a assez de données pour l'analyse historique
    if len(df) < lookback_periods + 100:  # Au moins 100 périodes d'entraînement
        print(f"⚠️ Pas assez de données pour l'analyse historique ({len(df)} < {lookback_periods + 100})")
        print(f"🔄 Réduction de la période d'analyse à {max(10, len(df) // 4)}")
        lookback_periods = max(10, len(df) // 4)
    
    # Créer l'environnement avec analyse historique
    print(f"\n🏗️ Création de l'environnement de trading avec analyse historique...")
    env = MinuteTradingEnvHistorical(df, lookback_periods=lookback_periods)
    env = DummyVecEnv([lambda: env])
    
    # Charger ou créer le modèle
    if continue_training and model_name:
        print(f"\n🔄 Continuation de l'entraînement du modèle {model_name}...")
        model, loaded_lookback_periods = load_existing_model(model_name)
        if model is None:
            print(f"❌ Impossible de charger le modèle {model_name}")
            return None
        
        # Utiliser les lookback_periods du modèle chargé
        if loaded_lookback_periods is not None:
            lookback_periods = loaded_lookback_periods
            print(f"🔄 Utilisation des périodes d'analyse du modèle: {lookback_periods}")
            
            # Recréer l'environnement avec les bons lookback_periods
            env = MinuteTradingEnvHistorical(df, initial_balance=10000.0, lookback_periods=lookback_periods)
            env = DummyVecEnv([lambda: env])
        
        model.set_env(env)
    else:
        print(f"\n🆕 Création d'un nouveau modèle PPO...")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=128,
            n_epochs=100,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            tensorboard_log="./ppo_trading_minute_historical_tensorboard/"
        )
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs("models", exist_ok=True)
    
    # Générer un nom de modèle si non fourni
    if not model_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"ppo_trading_historical_{symbol.replace('/', '-').replace('-', '_')}_{interval}_{timestamp}"
    
    print(f"\n🎓 Démarrage de l'entraînement...")
    print(f"💾 Le modèle sera sauvegardé sous: {model_name}")
    
    try:
        # Entraîner le modèle
        model.learn(total_timesteps=timesteps, progress_bar=True, reset_num_timesteps=not continue_training)
        
        # Sauvegarder le modèle
        model_path = f"models/{model_name}"
        model.save(model_path)
        print(f"\n✅ Modèle sauvegardé: {model_path}")
        
        # Sauvegarder les métadonnées du modèle
        metadata = {
            'model_name': model_name,
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'lookback_periods': lookback_periods,
            'timesteps': timesteps,
            'training_date': datetime.now().isoformat(),
            'model_type': 'PPO',
            'environment': 'MinuteTradingEnvHistorical',
            'observation_space': 'Box(150,)',
            'action_space': 'Discrete(3)',
            'initial_balance': 10000.0
        }
        
        metadata_path = f"models/{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 Métadonnées sauvegardées: {metadata_path}")
        
        # Tester le modèle sur quelques épisodes
        print(f"\n🧪 Test du modèle entraîné...")
        test_env = MinuteTradingEnvHistorical(df, lookback_periods=lookback_periods)
        obs, _ = test_env.reset()
        
        total_reward = 0
        steps = 0
        max_test_steps = min(1000, len(df) - lookback_periods)
        
        for _ in range(max_test_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        final_balance = info.get('total_value', test_env.balance)
        roi = ((final_balance - test_env.initial_balance) / test_env.initial_balance) * 100
        
        print(f"\n📊 Résultats du test:")
        print(f"   💰 Balance finale: ${final_balance:.2f}")
        print(f"   📈 ROI: {roi:.2f}%")
        print(f"   🎯 Récompense totale: {total_reward:.2f}")
        print(f"   📊 Trades: {info.get('total_trades', 0)}")
        print(f"   ✅ Taux de réussite: {info.get('win_rate', 0)*100:.1f}%")
        print(f"   📉 Drawdown max: {info.get('max_drawdown', 0)*100:.1f}%")
        
        # Sauvegarder les résultats
        results = {
            'model_name': model_name,
            'symbol': symbol,
            'period': period,
            'interval': interval,
            'lookback_periods': lookback_periods,
            'timesteps': timesteps,
            'final_balance': float(final_balance),
            'roi': float(roi),
            'total_reward': float(total_reward),
            'total_trades': int(info.get('total_trades', 0)),
            'winning_trades': int(info.get('winning_trades', 0)),
            'losing_trades': int(info.get('losing_trades', 0)),
            'win_rate': float(info.get('win_rate', 0)),
            'max_drawdown': float(info.get('max_drawdown', 0)),
            'training_date': datetime.now().isoformat()
        }
        
        # Sauvegarder dans un fichier JSON
        results_file = f"results/{model_name}_results.json"
        os.makedirs("results", exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Résultats sauvegardés: {results_file}")
        print(f"\n🎉 Entraînement terminé avec succès!")
        
        return {
            'model': model,
            'model_name': model_name,
            'results': results
        }
        
    except Exception as e:
        print(f"❌ Erreur pendant l'entraînement: {e}")
        return None

def get_user_input_for_training():
    """
    Interface interactive pour configurer l'entraînement
    """
    print("\n🎯 Configuration interactive de l'entraînement")
    print("=" * 50)
    
    # Choix entre données en ligne ou fichier local
    print("\n📊 Source des données:")
    print("  1. Télécharger des données en ligne (API)")
    print("  2. Utiliser un fichier local du dossier data/")
    
    while True:
        data_source = input("\n➤ Choisissez la source (1 ou 2): ").strip()
        if data_source in ['1', '2']:
            break
        print("❌ Choix invalide. Entrez 1 ou 2.")
    
    if data_source == '2':
        # Sélection d'un fichier local
        available_files = get_available_data_files()
        
        if not available_files:
            print("❌ Aucun fichier CSV trouvé dans le dossier data/")
            print("🔄 Basculement vers le téléchargement en ligne...")
            data_source = '1'
        else:
            print("\n📁 Fichiers disponibles dans data/:")
            for i, filename in enumerate(available_files, 1):
                print(f"  {i:2d}. {filename}")
            
            while True:
                try:
                    file_choice = input(f"\n➤ Choisissez un fichier (1-{len(available_files)}): ").strip()
                    if file_choice.isdigit():
                        file_num = int(file_choice)
                        if 1 <= file_num <= len(available_files):
                            selected_file = available_files[file_num - 1]
                            
                            # Demander les paramètres d'entraînement pour les fichiers locaux
                            while True:
                                try:
                                    timesteps_input = input("\n🎯 Nombre de timesteps (défaut: 50000): ").strip()
                                    timesteps = int(timesteps_input) if timesteps_input else 50000
                                    if timesteps > 0:
                                        break
                                    else:
                                        print("❌ Le nombre de timesteps doit être positif")
                                except ValueError:
                                    print("❌ Veuillez entrer un nombre valide")
                            
                            while True:
                                try:
                                    lookback_input = input("\n🔍 Nombre de périodes d'analyse (lookback, défaut: 30): ").strip()
                                    lookback_periods = int(lookback_input) if lookback_input else 30
                                    if lookback_periods > 0:
                                        break
                                    else:
                                        print("❌ Le nombre de périodes d'analyse doit être positif")
                                except ValueError:
                                    print("❌ Veuillez entrer un nombre valide")
                            
                            return selected_file, None, None, timesteps, lookback_periods, True  # Retourner le fichier sélectionné
                        else:
                            print(f"❌ Numéro invalide. Entrez un nombre entre 1 et {len(available_files)}")
                    else:
                        print("❌ Veuillez entrer un nombre")
                except ValueError:
                    print("❌ Veuillez entrer un nombre valide")
    
    # Sélection du symbole pour téléchargement en ligne
    print("\n📊 Sélection du symbole:")
    popular_stocks = get_popular_stocks()
    print("\n🔥 Actions populaires:")
    stock_items = popular_stocks[:10]  # Prendre les 10 premiers éléments de la liste
    for i, symbol in enumerate(stock_items, 1):
        print(f"  {i:2d}. {symbol}")
    
    print("\n💰 Cryptomonnaies populaires:")
    crypto_symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]
    for i, symbol in enumerate(crypto_symbols, 11):
        print(f"  {i:2d}. {symbol}")
    
    while True:
        choice = input("\n➤ Choisissez un numéro ou tapez un symbole personnalisé: ").strip()
        
        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= 10:
                symbol = stock_items[choice_num - 1]  # Accès direct à l'élément de la liste
                break
            elif 11 <= choice_num <= 15:
                symbol = crypto_symbols[choice_num - 11]
                break
            else:
                print("❌ Numéro invalide")
        else:
            symbol = choice.upper()
            break
    
    # Sélection de la période
    periods = ["1d", "2d", "3d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]
    print(f"\n📅 Périodes disponibles: {', '.join(periods)}")
    while True:
        period = input("➤ Période (défaut: 7d): ").strip() or "7d"
        if period in periods:
            break
        print(f"❌ Période invalide. Choisissez parmi: {', '.join(periods)}")
    
    # Sélection de l'intervalle
    intervals = ["1m", "5m", "15m", "30m", "1h"]
    print(f"\n ⏱️ Intervalles disponibles: {', '.join(intervals)}")
    while True:
        interval = input("➤ Intervalle (défaut: 1m): ").strip() or "1m"
        if interval in intervals:
            break
        print(f"❌ Intervalle invalide. Choisissez parmi: {', '.join(intervals)}")
    
    # Nombre de timesteps
    while True:
        try:
            timesteps_input = input("\n🎯 Nombre de timesteps (défaut: 50000): ").strip()
            timesteps = int(timesteps_input) if timesteps_input else 50000
            if timesteps > 0:
                break
            else:
                print("❌ Le nombre de timesteps doit être positif")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")
    
    # Nombre de périodes lookback
    while True:
        try:
            lookback_input = input("\n🔍 Nombre de périodes d'analyse (lookback, défaut: 30): ").strip()
            lookback_periods = int(lookback_input) if lookback_input else 30
            if lookback_periods > 0:
                break
            else:
                print("❌ Le nombre de périodes d'analyse doit être positif")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide")
    
    return symbol, period, interval, timesteps, lookback_periods, False

def select_model_for_training():
    """
    Interface pour sélectionner un modèle existant ou créer un nouveau
    """
    print("\n🤖 Configuration du modèle")
    print("=" * 30)
    
    available_models = list_available_models()
    
    if available_models:
        print("\n📋 Modèles disponibles:")
        for i, model in enumerate(available_models, 1):
            print(f"  {i:2d}. {model}")
        
        print("\n🆕 Options:")
        print("  0. Créer un nouveau modèle")
        print(f"  1-{len(available_models)}. Continuer l'entraînement d'un modèle existant")
        
        while True:
            try:
                choice = input("\n➤ Votre choix: ").strip()
                if choice == "0":
                    return None, False
                elif choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(available_models):
                        return available_models[choice_num - 1], True
                    else:
                        print(f"❌ Choix invalide. Entrez un nombre entre 0 et {len(available_models)}")
                else:
                    print("❌ Veuillez entrer un nombre")
            except ValueError:
                print("❌ Veuillez entrer un nombre valide")
    else:
        print("\n📝 Aucun modèle existant trouvé. Un nouveau modèle sera créé.")
        return None, False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Entraîner un modèle de trading minute avec analyse historique PPO')
    parser.add_argument('--symbol', type=str, default=None, help='Symbole à trader (ex: BTC-USD, AAPL)')
    parser.add_argument('--period', type=str, default=None, help='Période des données (ex: 1d, 5d, 1mo)')
    parser.add_argument('--interval', type=str, default=None, help='Intervalle des données (ex: 1m, 5m, 1h)')
    parser.add_argument('--timesteps', type=int, default=None, help='Nombre de timesteps pour l\'entraînement')
    parser.add_argument('--lookback', type=int, default=30, help='Nombre de périodes historiques à analyser')
    parser.add_argument('--model', type=str, default=None, help='Nom du modèle existant à continuer')
    parser.add_argument('--continue', dest='continue_training', action='store_true', help='Continuer l\'entraînement d\'un modèle existant')
    parser.add_argument('--interactive', action='store_true', help='Mode interactif pour la configuration')

    args = parser.parse_args()
    
    # Mode interactif
    if args.interactive:
        symbol, period, interval, timesteps, lookback_periods, use_file = get_user_input_for_training()
        model_name, continue_training = select_model_for_training()
        
        if use_file:
            filename = symbol  # Dans ce cas, symbol contient le nom du fichier
        else:
            filename = None
    else:
        use_file = False
        filename = None
        # Mode ligne de commande
        symbol = args.symbol or "BTC-USD"
        period = args.period or "7d"
        interval = args.interval or "1m"
        timesteps = args.timesteps or 50000
        lookback_periods = args.lookback
        model_name = args.model
        continue_training = args.continue_training
        
        # Vérification pour la continuation d'entraînement
        if continue_training and not model_name:
            available_models = list_available_models()
            if available_models:
                print("📋 Modèles disponibles:")
                for model in available_models:
                    print(f"  - {model}")
                print("\n❌ Veuillez spécifier un modèle avec --model <nom_du_modèle>")
                exit(1)
            else:
                print("❌ Aucun modèle existant trouvé pour continuer l'entraînement.")
                exit(1)

    print(f"\n🚀 Configuration de l'entraînement avec analyse historique:")
    print(f"   📊 Symbole: {symbol}")
    print(f"   📅 Période: {period}")
    print(f"   ⏱️ Intervalle: {interval}")
    print(f"   🔍 Périodes d'analyse: {lookback_periods}")
    print(f"   🎯 Timesteps: {timesteps:,}")
    if continue_training and model_name:
        print(f"   🔄 Continuation du modèle: {model_name}")
    else:
        print(f"   🆕 Nouveau modèle")
    
    # Lancer l'entraînement
    train_minute_model_historical(
        symbol=symbol,
        period=period,
        interval=interval,
        timesteps=timesteps,
        model_name=model_name,
        continue_training=continue_training,
        lookback_periods=lookback_periods,
        filename=filename,
        use_file=use_file
    )