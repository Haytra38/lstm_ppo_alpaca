"""
Module LSTM Predictor pour l'intégration avec PPO
Ce module fournit des classes pour charger et utiliser des modèles LSTM entraînés
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from typing import Optional, List, Dict, Union


class LSTMPredictor:
    """Classe pour charger et utiliser des modèles LSTM entraînés pour le trading"""
    
    def __init__(self, model_path: str, scaler_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialise le prédicteur LSTM
        
        Args:
            model_path: Chemin vers le fichier modèle .pth
            scaler_path: Chemin vers le fichier scaler .pkl (optionnel)
            device: Device PyTorch ('cpu' ou 'cuda')
        """
        self.device = torch.device(device)
        
        # Chargement du checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Récupération des paramètres
        self.config = checkpoint['config']
        self.sequence_length = checkpoint['sequence_length']
        self.features = checkpoint['features']
        self.input_size = checkpoint['input_size']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.bidirectional = getattr(self.config, 'bidirectional', False)
        
        # Chargement du scaler
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None
        
        # Création et chargement du modèle
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Buffer pour stocker l'historique
        self.history_buffer = []
        self.prediction_history = []
        
        print(f"✅ Modèle LSTM chargé:")
        print(f"  Performance: MAE={checkpoint['performance']['test_mae']:.6f}")
        print(f"  Features: {self.features}")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Bidirectional: {self.bidirectional}")
    
    def _create_model(self):
        """Crée le modèle LSTM basé sur la configuration"""
        return LSTMTradingModel(self.config).to(self.device)
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prépare les données pour la prédiction
        
        Args:
            data: DataFrame avec les données de marché
        
        Returns:
            np.array: Données préparées
        """
        # Calcul des features nécessaires
        df_features = data.copy()
        
        if 'returns' in self.features and 'returns' not in df_features.columns:
            df_features['returns'] = df_features['close'].pct_change()
        
        # RSI
        if 'rsi' in self.features:
            delta = df_features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        if 'macd' in self.features:
            ema_12 = df_features['close'].ewm(span=12).mean()
            ema_26 = df_features['close'].ewm(span=26).mean()
            df_features['macd'] = ema_12 - ema_26
            df_features['macd_signal'] = df_features['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        if 'bb_position' in self.features:
            rolling_mean = df_features['close'].rolling(window=20).mean()
            rolling_std = df_features['close'].rolling(window=20).std()
            df_features['bb_position'] = (df_features['close'] - rolling_mean) / (2 * rolling_std)
        
        # Sélection des features
        feature_data = df_features[self.features].dropna()
        
        # Normalisation
        if self.scaler is not None:
            feature_data = self.scaler.transform(feature_data)
        
        return feature_data
    
    def predict(self, data: pd.DataFrame) -> float:
        """
        Effectue une prédiction sur les données
        
        Args:
            data: DataFrame avec les données de marché (doit contenir au moins sequence_length lignes)
        
        Returns:
            float: Prédiction du prix futur
        """
        if len(data) < self.sequence_length:
            raise ValueError(f"Données insuffisantes: {len(data)} < {self.sequence_length}")
        
        # Préparation des données
        features = self.preprocess_data(data)
        
        if len(features) < self.sequence_length:
            raise ValueError(f"Features insuffisants après pré-traitement: {len(features)} < {self.sequence_length}")
        
        # Préparation de la séquence
        sequence = features[-self.sequence_length:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            prediction = self.model(sequence_tensor)
        
        pred_value = prediction.item()
        self.prediction_history.append(pred_value)
        
        return pred_value
    
    def predict_sequence(self, data_sequence: List[pd.DataFrame]) -> List[Optional[float]]:
        """
        Effectue des prédictions sur une séquence de données
        
        Args:
            data_sequence: Liste de DataFrames avec les données de marché
        
        Returns:
            list: Liste des prédictions
        """
        predictions = []
        
        for i, data in enumerate(data_sequence):
            try:
                pred = self.predict(data)
                predictions.append(pred)
            except Exception as e:
                print(f"Erreur lors de la prédiction {i}: {e}")
                predictions.append(None)
        
        return predictions
    
    def get_prediction_statistics(self) -> Dict[str, Union[float, int]]:
        """
        Retourne des statistiques sur les prédictions effectuées
        
        Returns:
            dict: Statistiques des prédictions
        """
        if not self.prediction_history:
            return {'count': 0}
        
        predictions = np.array(self.prediction_history)
        
        return {
            'count': len(predictions),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'last_prediction': float(predictions[-1]) if len(predictions) > 0 else None
        }
    
    def reset_history(self):
        """Réinitialise l'historique des prédictions"""
        self.prediction_history = []
        self.history_buffer = []


class PPOIntegration:
    """Classe pour intégrer les prédictions LSTM dans PPO"""
    
    def __init__(self, lstm_predictor: LSTMPredictor, confidence_threshold: float = 0.01):
        """
        Initialise l'intégration
        
        Args:
            lstm_predictor: Instance de LSTMPredictor
            confidence_threshold: Seuil de confiance pour les prédictions (défaut: 1%)
        """
        self.predictor = lstm_predictor
        self.confidence_threshold = confidence_threshold
        self.signal_history = []
        self.prediction_errors = []
    
    def get_lstm_signal(self, market_data: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """
        Génère un signal de trading basé sur la prédiction LSTM
        
        Args:
            market_data: DataFrame avec les données de marché
        
        Returns:
            dict: Signal avec direction, force et confiance
        """
        try:
            # Prédiction
            prediction = self.predictor.predict(market_data)
            current_price = market_data['close'].iloc[-1]
            
            # Calcul du signal
            price_change = (prediction - current_price) / current_price
            
            # Direction du signal
            if price_change > self.confidence_threshold:
                direction = 'BUY'
                strength = min(abs(price_change) * 10, 1.0)  # Force normalisée
            elif price_change < -self.confidence_threshold:
                direction = 'SELL'
                strength = min(abs(price_change) * 10, 1.0)
            else:
                direction = 'HOLD'
                strength = 0.0
            
            # Confiance basée sur l'historique des erreurs
            confidence = self._calculate_confidence()
            
            signal = {
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'predicted_price': prediction,
                'current_price': current_price,
                'expected_return': price_change,
                'timestamp': datetime.now()
            }
            
            # Historique
            self.signal_history.append(signal)
            
            return signal
            
        except Exception as e:
            print(f"Erreur lors du calcul du signal: {e}")
            current_price = market_data['close'].iloc[-1] if len(market_data) > 0 else 0
            
            return {
                'direction': 'HOLD',
                'strength': 0.0,
                'confidence': 0.0,
                'predicted_price': current_price,
                'current_price': current_price,
                'expected_return': 0.0,
                'timestamp': datetime.now()
            }
    
    def _calculate_confidence(self) -> float:
        """Calcule la confiance basée sur l'historique des performances"""
        # Pour l'instant, retourne une confiance fixe
        # Dans une implémentation réelle, on utiliserait l'historique des erreurs
        # et la performance passée du modèle
        return 0.75
    
    def get_state_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Génère des features supplémentaires pour l'état PPO
        
        Args:
            market_data: DataFrame avec les données de marché
        
        Returns:
            np.array: Features supplémentaires pour l'état PPO
        """
        signal = self.get_lstm_signal(market_data)
        
        # Features pour PPO (6 dimensions)
        features = np.array([
            1.0 if signal['direction'] == 'BUY' else 0.0,      # Signal BUY
            1.0 if signal['direction'] == 'SELL' else 0.0,     # Signal SELL
            signal['strength'],                                  # Force du signal
            signal['confidence'],                                # Confiance
            np.clip(signal['expected_return'], -0.1, 0.1),     # Retour attendu (clippé)
            np.clip(signal['predicted_price'] / signal['current_price'] - 1.0, -0.2, 0.2)  # Ratio prix prédit/prix actuel
        ])
        
        return features
    
    def get_signal_statistics(self) -> Dict[str, Union[int, float]]:
        """
        Retourne des statistiques sur les signaux générés
        
        Returns:
            dict: Statistiques des signaux
        """
        if not self.signal_history:
            return {'total_signals': 0}
        
        buy_signals = sum(1 for s in self.signal_history if s['direction'] == 'BUY')
        sell_signals = sum(1 for s in self.signal_history if s['direction'] == 'SELL')
        hold_signals = sum(1 for s in self.signal_history if s['direction'] == 'HOLD')
        
        avg_strength = np.mean([s['strength'] for s in self.signal_history])
        avg_confidence = np.mean([s['confidence'] for s in self.signal_history])
        
        return {
            'total_signals': len(self.signal_history),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'avg_strength': float(avg_strength),
            'avg_confidence': float(avg_confidence)
        }
    
    def reset_history(self):
        """Réinitialise l'historique des signaux"""
        self.signal_history = []


class LSTMTradingModel(nn.Module):
    """Modèle LSTM pour la prédiction de prix de trading"""
    
    def __init__(self, config):
        super(LSTMTradingModel, self).__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.bidirectional = getattr(config, 'bidirectional', False)
        
        # Couche LSTM
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Calcul de la taille de sortie LSTM
        lstm_output_size = config.hidden_size * (2 if self.bidirectional else 1)
        
        # Couches denses supplémentaires
        dense_layers = []
        if hasattr(config, 'dense_layers') and config.dense_layers:
            prev_size = lstm_output_size
            for i, dense_size in enumerate(config.dense_layers):
                dense_layers.append(nn.Linear(prev_size, dense_size))
                
                # Activation
                if hasattr(config, 'dense_activations') and i < len(config.dense_activations):
                    if config.dense_activations[i] == 'relu':
                        dense_layers.append(nn.ReLU())
                    elif config.dense_activations[i] == 'tanh':
                        dense_layers.append(nn.Tanh())
                    elif config.dense_activations[i] == 'sigmoid':
                        dense_layers.append(nn.Sigmoid())
                
                # Dropout
                if hasattr(config, 'dropout') and config.dropout > 0:
                    dense_layers.append(nn.Dropout(config.dropout))
                
                prev_size = dense_size
            
            # Couche de sortie
            dense_layers.append(nn.Linear(prev_size, 1))
        else:
            # Sortie directe depuis LSTM
            dense_layers.append(nn.Linear(lstm_output_size, 1))
        
        self.dense_layers = nn.Sequential(*dense_layers)
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Prendre la dernière sortie temporelle
        if self.bidirectional:
            # Concaténer les états cachés des deux directions
            last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last_hidden = hidden[-1]
        
        # Passer à travers les couches denses
        output = self.dense_layers(last_hidden)
        
        return output.squeeze()


# Fonctions utilitaires
def load_lstm_model(model_path: str, scaler_path: Optional[str] = None, device: str = 'cpu') -> LSTMPredictor:
    """
    Charge un modèle LSTM entraîné
    
    Args:
        model_path: Chemin vers le fichier modèle
        scaler_path: Chemin vers le fichier scaler (optionnel)
        device: Device PyTorch
    
    Returns:
        LSTMPredictor: Instance du prédicteur chargé
    """
    return LSTMPredictor(model_path, scaler_path, device)


def create_ppo_integration(lstm_predictor: LSTMPredictor, confidence_threshold: float = 0.01) -> PPOIntegration:
    """
    Crée une intégration PPO-LSTM
    
    Args:
        lstm_predictor: Instance LSTMPredictor
        confidence_threshold: Seuil de confiance
    
    Returns:
        PPOIntegration: Instance d'intégration
    """
    return PPOIntegration(lstm_predictor, confidence_threshold)


# Exemple d'utilisation
if __name__ == "__main__":
    # Exemple de chargement et utilisation
    print("🧪 Test du module LSTM Predictor")
    
    # Chemins vers les fichiers (à adapter selon votre configuration)
    model_path = "trained_models/lstm_model_20240101_120000.pth"
    scaler_path = "trained_models/lstm_scaler_20240101_120000.pkl"
    
    if os.path.exists(model_path):
        # Chargement du modèle
        predictor = load_lstm_model(model_path, scaler_path)
        
        # Création de données de test
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1min')
        test_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'high': np.random.randn(100).cumsum() + 101,
            'low': np.random.randn(100).cumsum() + 99,
            'open': np.random.randn(100).cumsum() + 100
        }, index=dates)
        
        # Test de prédiction
        try:
            prediction = predictor.predict(test_data)
            print(f"✅ Prédiction réussie: {prediction:.4f}")
            
            # Test d'intégration PPO
            ppo_integration = create_ppo_integration(predictor)
            signal = ppo_integration.get_lstm_signal(test_data)
            print(f"✅ Signal PPO: {signal['direction']} (force: {signal['strength']:.2f})")
            
            # Test des features d'état
            state_features = ppo_integration.get_state_features(test_data)
            print(f"✅ Features d'état: {state_features}")
            
        except Exception as e:
            print(f"❌ Erreur lors du test: {e}")
    else:
        print("⚠️  Aucun modèle trouvé. Veuillez d'abord entraîner un modèle avec le notebook.")
    
    print("🎯 Module prêt à être utilisé!")