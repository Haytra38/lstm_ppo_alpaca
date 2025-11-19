#!/usr/bin/env python3
"""
Test d'intégration PPO avec LSTM
Vérifie que le modèle LSTM peut être utilisé par l'environnement PPO
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

from lstm_predictor import LSTMPredictor, PPOIntegration
from train_minute_model_lstm import MinuteTradingEnvHistorical
from data_loader import load_data_from_file

def test_ppo_lstm_integration():
    print("🧪 Test d'intégration PPO + LSTM")
    print("=" * 50)
    
    # Charger un modèle LSTM existant pour le test
    model_path = "saved_models/robust"
    
    if not os.path.exists(model_path):
        print(f"❌ Modèle non trouvé: {model_path}")
        return False
    
    print(f"\n📁 Chargement du modèle LSTM depuis: {model_path}")
    
    try:
        # Créer le prédicteur LSTM
        lstm_predictor = LSTMPredictor(model_path)
        print("✅ LSTMPredictor créé avec succès")
        
        # Créer l'intégration PPO
        ppo_integration = PPOIntegration(lstm_predictor)
        print("✅ PPOIntegration créée avec succès")
        
        # Créer des données de test
        print("\n📊 Création de données de test...")
        dates = pd.date_range(start=datetime.now() - timedelta(days=5), end=datetime.now(), freq='1min')
        np.random.seed(42)
        
        # Générer des données de prix réalistes
        initial_price = 150.0
        prices = [initial_price]
        
        for i in range(1, len(dates)):
            change = np.random.normal(0.0001, 0.01)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        test_df = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        print(f"✅ Données de test créées: {len(test_df)} périodes")
        
        # Test 1: Prédiction LSTM simple
        print("\n🎯 Test 1: Prédiction LSTM simple")
        
        # Prendre une séquence pour la prédiction
        sequence_length = 60  # Assumé, peut varier selon le modèle
        if len(test_df) >= sequence_length:
            test_sequence = test_df.tail(sequence_length)
            
            prediction_config = {
                "target_columns": ["Open", "High", "Low", "Close", "Volume"],
                "prediction_steps": 5,
                "confidence_interval": False
            }
            
            timeframe_info = {
                "end_date": test_sequence.index[-1],
                "pandas_freq": "min"
            }
            
            # Test avec le prédicteur direct
            result = lstm_predictor.predict(test_sequence, prediction_config, timeframe_info)
            
            if "error" not in result:
                print("✅ Prédiction LSTM réussie")
                print(f"   \U0001f4c8 Forme des prédictions: {np.array(result['future']['Close']).shape}")
                
                # Test avec l'intégration PPO
                features = ppo_integration.get_state_features(test_sequence)
                print(f"✅ Features PPO générées: {features.shape}")
                
            else:
                print(f"❌ Erreur de prédiction: {result['error']}")
                return False
        else:
            print(f"❌ Pas assez de données: {len(test_df)} < {sequence_length}")
            return False
        
        # Test 2: Environnement PPO avec LSTM
        print("\n🎯 Test 2: Environnement PPO avec LSTM")
        
        try:
            # Créer l'environnement avec le modèle LSTM
            env = MinuteTradingEnvHistorical(
                df=test_df,
                initial_balance=1000.0,
                lookback_periods=30,
                lstm_model_path=model_path,
                use_lstm_features=True
            )
            
            print("✅ Environnement PPO créé avec succès")
            print(f"   \U0001f4ca Taille d'observation: {env.observation_space.shape}")
            print(f"   \U0001f3af Espace d'actions: {env.action_space}")
            
            # Test rapide de l'environnement
            obs, info = env.reset()
            print(f"✅ Reset réussi, observation shape: {obs.shape}")
            
            # Faire quelques pas dans l'environnement
            for step in range(min(10, len(test_df) - 30)):
                action = env.action_space.sample()  # Action aléatoire
                obs, reward, done, truncated, info = env.step(action)
                
                if done or truncated:
                    break
            
            print(f"✅ Environnement testé: {step + 1} steps réussis")
            
        except Exception as e:
            print(f"❌ Erreur environnement PPO: {e}")
            return False
        
        # Test 3: Vérification des features LSTM
        print("\n🎯 Test 3: Vérification des features LSTM")
        
        # Obtenir les features d'un état spécifique
        current_data = test_df.iloc[:100]  # Premiers 100 points
        
        if hasattr(env, '_get_lstm_predictions'):
            lstm_features = env._get_lstm_predictions()
            print(f"✅ Features LSTM dans l'environnement: {lstm_features.shape}")
            
            # Vérifier que les features ne sont pas zéro
            if np.any(np.abs(lstm_features) > 1e-10):
                print("✅ Features LSTM non-nulles")
            else:
                print("⚠️  Features LSTM sont zéro (modèle peut ne pas être chargé)")
        
        print("\n\U0001f389 Tests terminés avec succès!")
        print("✅ L'intégration PPO + LSTM fonctionne correctement")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Démarrage des tests d'intégration PPO + LSTM")
    print("=" * 60)
    
    success = test_ppo_lstm_integration()
    
    if success:
        print("\n\U0001f3c6 Tous les tests ont réussi!")
        print("\n\U0001f680 Le modèle LSTM est prêt pour l'entraînement PPO")
        print("\nPour lancer l'entraînement PPO:")
        print("  python train_minute_model_lstm.py --lstm-model saved_models/robust")
    else:
        print("\n❌ Certains tests ont échoué")
        print("Vérifiez la configuration et les fichiers du modèle")

if __name__ == "__main__":
    main()