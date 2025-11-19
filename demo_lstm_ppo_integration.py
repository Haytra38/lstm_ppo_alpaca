#!/usr/bin/env python3
"""
Démonstration finale de l'intégration LSTM + PPO
Montre l'utilisation complète du système avec toutes les améliorations
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Ajout du chemin du projet
sys.path.append('.')

from lstm_config import LSTMConfigurator
from lstm_model import LSTMModel

def demonstrate_lstm_ppo_integration():
    """
    Démonstration complète du système LSTM + PPO amélioré
    """
    print("\U0001f680 Démonstration LSTM + PPO Amélioré")
    print("=" * 60)
    
    # 1. Configuration avancée
    print("\n⚙️ 1. Configuration avancée LSTM")
    config = LSTMConfigurator()
    
    # Afficher la configuration actuelle
    current_config = config.current_config
    model_config = current_config["model_config"]
    training_config = current_config["training_config"]
    
    print(f"   \U0001f4ca Architecture: {len(model_config['layers'])} couches LSTM")
    print(f"   ↔️ Bidirectionnel: {any(l.get('bidirectional', False) for l in model_config['layers'])}")
    print(f"   \U0001f4c8 Batch normalization: {any(l.get('batch_normalization', False) for l in model_config['layers'])}")
    print(f"   \U0001f3af Optimiseur: {model_config['optimizer']}")
    print(f"   \U0001f4c8 Learning rate: {model_config['learning_rate']}")
    print(f"   ⏱️  Sequence length: {model_config['sequence_length']}")
    
    # 2. Créer des données de démonstration
    print("\n\U0001f4ca 2. Création de données de démonstration")
    
    # Données synthétiques réalistes
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='1min')
    np.random.seed(42)
    
    # Générer des données de prix avec tendances
    initial_price = 150.0
    prices = [initial_price]
    
    for i in range(1, len(dates)):
        # Ajouter des tendances et cycles
        trend = np.sin(i / 1000) * 0.001  # Cycle long
        noise = np.random.normal(0, 0.01)
        new_price = prices[-1] * (1 + trend + noise)
        prices.append(new_price)
    
    demo_df = pd.DataFrame({
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)
    
    print(f"   ✅ Données créées: {len(demo_df)} périodes")
    print(f"   \U0001f4c8 Prix moyen: ${demo_df['Close'].mean():.2f}")
    print(f"   \U0001f4c8 Volatilité: {demo_df['Close'].std():.2f}")
    
    # 3. Démontrer la configuration personnalisée
    print("\n\U0001f527 3. Configuration personnalisée")
    
    # Modifier quelques paramètres pour la démo
    model_config["layers"][0]["bidirectional"] = True
    model_config["layers"][0]["batch_normalization"] = True
    model_config["learning_rate"] = 0.001
    training_config["epochs"] = 50
    training_config["batch_size"] = 32
    
    print("   ✅ Paramètres personnalisés appliqués")
    print("   ↔️ LSTM bidirectionnel: activé")
    print("   \U0001f4c8 Batch normalization: activé")
    print("   \U0001f4c8 Learning rate: 0.001")
    print("   ⏱️  Epochs: 50")
    
    # 4. Créer et tester le modèle
    print("\n\U0001f9ea 4. Test du modèle LSTM")
    
    try:
        model = LSTMModel(scaler_type='robust')
        
        # Obtenir la configuration formatée
        lstm_config = config.get_config_for_lstm_model()
        lstm_config['data_path'] = 'demo_data.csv'  # Pour la démo
        
        print("   ✅ Modèle LSTM créé")
        print(f"   \U0001f4ca Type de scaler: robust")
        print(f"   \U0001f4c8 Nombre de paramètres: configurable")
        
        # Test de prédiction rapide (sans entraînement complet)
        print("\n   \U0001f52e Test de prédiction (mock)")
        
        # Simuler une prédiction
        test_sequence = demo_df.tail(60)  # 60 dernières périodes
        mock_prediction = {
            "future": {
                "Close": [demo_df['Close'].iloc[-1] * (1 + i * 0.001) for i in range(1, 6)],
                "Open": [demo_df['Open'].iloc[-1] * (1 + i * 0.001) for i in range(1, 6)],
                "High": [demo_df['High'].iloc[-1] * (1 + i * 0.001) for i in range(1, 6)],
                "Low": [demo_df['Low'].iloc[-1] * (1 + i * 0.001) for i in range(1, 6)],
                "Volume": [demo_df['Volume'].iloc[-1] * (1 + i * 0.01) for i in range(1, 6)]
            }
        }
        
        print("   ✅ Prédiction mock générée")
        print(f"   \U0001f4c8 Prédictions (5 steps): {len(mock_prediction['future']['Close'])} valeurs")
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
    
    # 5. Démontrer l'intégration PPO
    print("\n\U0001f3d7️ 5. Intégration PPO")
    
    try:
        from train_minute_model_lstm import MinuteTradingEnvHistorical
        
        # Créer l'environnement avec LSTM
        env = MinuteTradingEnvHistorical(
            df=demo_df,
            initial_balance=1000.0,
            lookback_periods=30,
            lstm_model_path=None,  # Pas de modèle réel pour la démo
            lstm_scaler_path=None,
            use_lstm_features=True
        )
        
        print("   ✅ Environnement PPO créé")
        print(f"   \U0001f4ca Observation space: {env.observation_space.shape}")
        print(f"   \U0001f3af Action space: {env.action_space}")
        print(f"   \U0001f4c8 Balance initiale: ${env.initial_balance}")
        
        # Test rapide
        obs, info = env.reset()
        print(f"   ✅ Reset réussi: observation shape {obs.shape}")
        
        # Quelques steps
        total_reward = 0
        for step in range(5):
            action = np.random.randint(0, 3)  # Action aléatoire
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        print(f"   ✅ Steps réussis: {step + 1} steps")
        print(f"   \U0001f4b0 Reward total: {total_reward:.4f}")
        
    except Exception as e:
        print(f"   ❌ Erreur PPO: {e}")
    
    # 6. Résumé des améliorations
    print("\n\U0001f4cb 6. Résumé des améliorations")
    print("   ✅ Configuration complète de tous les paramètres LSTM")
    print("   ↔️ Support LSTM bidirectionnel")
    print("   \U0001f4c8 Batch normalization configurable")
    print("   \U0001f3af Multiple optimizers supportés")
    print("   ⏱️  Early stopping et learning rate scheduling")
    print("   \U0001f52e Prédictions avec intervalles de confiance")
    print("   \U0001f3d7️  Intégration complète avec PPO")
    print("   \U0001f9ee Système de prédiction modulaire")
    
    print("\n\U0001f389 Démonstration terminée!")
    print("\U0001f680 Le système LSTM+PPO est maintenant entièrement configurable et opérationnel")
    
    # 7. Instructions pour l'utilisation
    print("\n\U0001f4da Instructions d'utilisation:")
    print("\n   Pour entraîner un modèle LSTM:")
    print("   python train_lstm_for_ppo.py")
    print("\n   Pour tester l'intégration:")
    print("   python test_ppo_lstm_mock.py")
    print("\n   Pour lancer l'entraînement PPO:")
    print("   python train_minute_model_lstm.py --lstm-model <chemin_du_modèle>")
    print("\n   Pour utiliser la configuration avancée:")
    print("   - Modifier lstm_config.py")
    print("   - Utiliser LSTMConfigurator() pour personnaliser")
    print("   - Activer bidirectional, batch_norm, etc.")

def main():
    demonstrate_lstm_ppo_integration()

if __name__ == "__main__":
    main()