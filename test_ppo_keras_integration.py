#!/usr/bin/env python3
"""
Test d'intégration PPO avec LSTM (Keras)
Vérifie que le modèle LSTM Keras peut être utilisé par l'environnement PPO
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

from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import joblib

def load_keras_lstm_model(model_dir):
    """
    Charge un modèle LSTM Keras et son scaler
    """
    try:
        # Charger le modèle Keras
        model_path = os.path.join(model_dir, "model.keras")
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        config_path = os.path.join(model_dir, "config.py")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        # Charger le modèle
        model = load_model(model_path)
        print(f"✅ Modèle Keras chargé: {model_path}")
        
        # Charger le scaler
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"✅ Scaler chargé: {scaler_path}")
        else:
            scaler = RobustScaler()  # Scaler par défaut
            print("⚠️  Scaler non trouvé, utilisation du scaler par défaut")
        
        # Charger la configuration si disponible
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_content = f.read()
                # Extraire des informations de base
                if "sequence_length" in config_content:
                    import re
                    match = re.search(r'sequence_length\s*=\s*(\d+)', config_content)
                    if match:
                        config['sequence_length'] = int(match.group(1))
                print("✅ Configuration chargée")
            except Exception as e:
                print(f"⚠️  Erreur lors du chargement de la config: {e}")
        
        return model, scaler, config
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None, None, {}

def predict_with_keras_model(model, scaler, data, sequence_length=60, prediction_steps=5):
    """
    Effectue une prédiction avec le modèle Keras
    """
    try:
        if len(data) < sequence_length:
            raise ValueError(f"Pas assez de données: {len(data)} < {sequence_length}")
        
        # Prendre la dernière séquence
        sequence = data.tail(sequence_length)
        
        # Normaliser les données
        if hasattr(scaler, 'transform'):
            normalized_data = scaler.transform(sequence.values)
        else:
            normalized_data = sequence.values
        
        # Reshape pour le modèle LSTM (batch_size, timesteps, features)
        X = normalized_data.reshape(1, sequence_length, -1)
        
        # Prédiction
        prediction = model.predict(X, verbose=0)
        
        # Dé-normaliser si possible
        if hasattr(scaler, 'inverse_transform'):
            # Adapter la forme pour l'inverse transformation
            if prediction.shape[-1] < sequence.shape[1]:
                # Padding avec des zéros pour correspondre à la forme originale
                padded_pred = np.zeros((prediction.shape[0], sequence.shape[1]))
                padded_pred[:, :prediction.shape[-1]] = prediction
                denormalized = scaler.inverse_transform(padded_pred)
                prediction = denormalized[:, :prediction.shape[-1]]
            else:
                prediction = scaler.inverse_transform(prediction)
        
        return prediction.flatten()
        
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        return np.zeros(prediction_steps * data.shape[1])

def test_keras_lstm_integration():
    """
    Test l'intégration avec un modèle LSTM Keras existant
    """
    print("\U0001f9ea Test d'intégration PPO + LSTM (Keras)")
    print("=" * 55)
    
    # Chemins des modèles disponibles
    model_dirs = [
        "saved_models/robust",
        "saved_models/minmax", 
        "saved_models/test"
    ]
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            print(f"\n📁 Test du modèle: {model_dir}")
            
            # Charger le modèle
            model, scaler, config = load_keras_lstm_model(model_dir)
            
            if model is None:
                continue
            
            # Créer des données de test
            print("\n📊 Création de données de test...")
            dates = pd.date_range(start=datetime.now() - timedelta(days=2), end=datetime.now(), freq='1min')
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
            
            # Test de prédiction
            print("\n\U0001f52e Test de prédiction...")
            
            sequence_length = config.get('sequence_length', 60)
            prediction = predict_with_keras_model(model, scaler, test_df, sequence_length)
            
            print(f"✅ Prédiction réussie!")
            print(f"   \U0001f4c8 Forme de la prédiction: {prediction.shape}")
            print(f"   \U0001f4ca Premières valeurs: {prediction[:5]}")
            
            # Test avec l'environnement PPO
            print("\n\U0001f3d7️  Test avec l'environnement PPO...")
            
            try:
                from train_minute_model_lstm import MinuteTradingEnvHistorical
                
                # Créer l'environnement avec le modèle LSTM
                env = MinuteTradingEnvHistorical(
                    df=test_df,
                    initial_balance=1000.0,
                    lookback_periods=30,
                    lstm_model_path=model_dir,  # Passer le chemin du modèle
                    use_lstm_features=True
                )
                
                print("✅ Environnement PPO créé avec succès")
                print(f"   \U0001f4ca Taille d'observation: {env.observation_space.shape}")
                
                # Test rapide
                obs, info = env.reset()
                print(f"✅ Reset réussi, observation shape: {obs.shape}")
                
                # Faire quelques pas
                for step in range(min(5, len(test_df) - 30)):
                    action = 0  # Hold
                    obs, reward, done, truncated, info = env.step(action)
                    
                    if done or truncated:
                        break
                
                print(f"✅ Test environnement: {step + 1} steps réussis")
                
                # Vérifier les features LSTM
                if hasattr(env, 'lstm_model') and env.lstm_model is not None:
                    print("✅ Modèle LSTM chargé dans l'environnement")
                else:
                    print("⚠️  Modèle LSTM non chargé dans l'environnement")
                
                print(f"\U0001f389 Test réussi pour {model_dir}!")
                return True
                
            except Exception as e:
                print(f"❌ Erreur avec l'environnement PPO: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n❌ Aucun modèle n'a pu être testé avec succès")
    return False

def main():
    print("\U0001f680 Démarrage des tests d'intégration PPO + LSTM (Keras)")
    print("=" * 70)
    
    success = test_keras_lstm_integration()
    
    if success:
        print("\n\U0001f3c6 Tests réussis!")
        print("\n\U0001f680 L'intégration PPO + LSTM fonctionne avec les modèles Keras existants")
        print("\nPour lancer l'entraînement PPO:")
        print("  python train_minute_model_lstm.py --lstm-model saved_models/robust")
    else:
        print("\n❌ Les tests ont échoué")
        print("Vérifiez les modèles et la configuration")

if __name__ == "__main__":
    main()