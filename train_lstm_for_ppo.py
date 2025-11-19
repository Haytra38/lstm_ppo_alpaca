#!/usr/bin/env python3
"""
Script d'entraînement LSTM pour l'intégration PPO
Utilise la configuration complète avec tous les paramètres disponibles
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Ajout du chemin du projet
sys.path.append('.')

from lstm_config import LSTMConfigurator
from lstm_model import LSTMModel
from data_loader import get_stock_data, load_data_from_file

def main():
    print("🚀 Démarrage de l'entraînement LSTM pour l'intégration PPO")
    print("=" * 60)
    
    # Configuration avancée avec tous les paramètres
    config = LSTMConfigurator()
    
    print("\n⚙️ Configuration avancée du modèle LSTM:")
    # Utiliser la configuration existante et la personnaliser
    
    # Configuration personnalisée pour le trading - modifier la configuration existante
    print("\n⚙️ Configuration personnalisée du modèle LSTM:")
    
    # Accéder à la configuration actuelle et la modifier
    current_config = config.current_config
    
    # Modifier la configuration du modèle
    model_config = current_config["model_config"]
    
    # Mettre à jour les couches LSTM
    model_config["layers"] = [
        {
            "units": 128,
            "return_sequences": True,
            "dropout": 0.2,
            "sequence_length": 60,
            "bidirectional": True,
            "batch_normalization": True,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "kernel_regularizer": None,
            "recurrent_regularizer": None,
            "dropout_type": "standard",
            "use_bias": True,
            "unit_forget_bias": True
        },
        {
            "units": 64,
            "return_sequences": True,
            "dropout": 0.2,
            "sequence_length": 60,
            "bidirectional": True,
            "batch_normalization": True,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "kernel_regularizer": None,
            "recurrent_regularizer": None,
            "dropout_type": "standard",
            "use_bias": True,
            "unit_forget_bias": True
        },
        {
            "units": 32,
            "return_sequences": False,
            "dropout": 0.2,
            "sequence_length": 60,
            "bidirectional": True,
            "batch_normalization": True,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "kernel_regularizer": None,
            "recurrent_regularizer": None,
            "dropout_type": "standard",
            "use_bias": True,
            "unit_forget_bias": True
        }
    ]
    
    # Mettre à jour les autres paramètres
    model_config["sequence_length"] = 60
    model_config["learning_rate"] = 0.001
    model_config["optimizer"] = "adam"
    model_config["mixed_precision"] = True
    model_config["gradient_clipping"]["enabled"] = True
    model_config["gradient_clipping"]["max_norm"] = 1.0
    
    # Modifier la configuration d'entraînement
    training_config = current_config["training_config"]
    training_config["epochs"] = 100
    training_config["batch_size"] = 32
    training_config["target_columns"] = ["Open", "High", "Low", "Close", "Volume"]
    training_config["validation_split"] = 0.2
    training_config["learning_rate"] = 0.001
    training_config["early_stopping"]["patience"] = 15
    
    print(f"\n📊 Configuration appliquée:")
    print(f"   📏 Sequence length: {model_config['sequence_length']}")
    print(f"   🔮 Target columns: {training_config['target_columns']}")
    print(f"   🧠 LSTM layers: {len([l for l in model_config['layers'] if l.get('units', 0) > 0])}")
    print(f"   ↔️ Bidirectional: {any(l.get('bidirectional', False) for l in model_config['layers'])}")
    print(f"   📊 Batch normalization: {any(l.get('batch_normalization', False) for l in model_config['layers'])}")
    print(f"   \U0001f4ca Epochs: {training_config['epochs']}")
    print(f"   \U0001f4c8 Batch size: {training_config['batch_size']}")
    
    # Chargement des données
    print("\n📥 Chargement des données...")
    
    # Essayer de charger des données depuis un fichier ou en télécharger
    try:
        # Chercher des fichiers de données disponibles
        data_files = [f for f in os.listdir('.') if f.endswith('.csv') and 'stock' in f.lower()]
        
        if data_files:
            # Charger le premier fichier disponible
            df = load_data_from_file(data_files[0])
            print(f"✅ Données chargées depuis {data_files[0]}")
        else:
            # Télécharger des données exemple
            print("📥 Téléchargement de données exemple AAPL...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)  # 1 an de données
            
            df = get_stock_data(
                symbol="AAPL",
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                timeframe="1Min"
            )
            
            if df is not None and not df.empty:
                # Sauvegarder pour usage futur
                filename = f"stock_data_AAPL_{end_date.strftime('%Y%m%d')}.csv"
                df.to_csv(filename)
                print(f"✅ Données sauvegardées dans {filename}")
            else:
                raise Exception("Impossible de télécharger les données")
                
    except Exception as e:
        print(f"❌ Erreur lors du chargement des données: {e}")
        # Créer des données synthétiques pour la démonstration
        print("🔄 Création de données synthétiques pour la démonstration...")
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='1min')
        np.random.seed(42)
        
        # Générer des données de prix réalistes
        initial_price = 150.0
        prices = [initial_price]
        
        for i in range(1, len(dates)):
            # Random walk avec tendance légèrement positive
            change = np.random.normal(0.001, 0.02)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        df = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
        
        print(f"✅ Données synthétiques créées: {len(df)} périodes")
    
    print(f"📊 Données chargées: {len(df)} périodes")
    print(f"   📅 Période: {df.index[0]} à {df.index[-1]}")
    print(f"   \U0001f4c8 Colonnes: {list(df.columns)}")
    
    # Visualisation rapide des données
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.8)
    plt.title('Prix de clôture')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['Volume'], label='Volume', color='orange', alpha=0.8)
    plt.title('Volume')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_overview.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Création et entraînement du modèle
    print("\n🧠 Création du modèle LSTM...")
    
    # Obtenir la configuration formatée pour LSTMModel
    lstm_config = config.get_config_for_lstm_model()
    
    # Créer le modèle avec le scaler configuré
    scaler_config = lstm_config.get('scaler_config', {'scaler_type': 'robust', 'scaler_config': {}})
    model = LSTMModel(scaler_config=scaler_config)
    
    print("\n\U0001f3af Début de l'entraînement...")
    start_time = datetime.now()
    
    # Entraînement avec monitoring
    training_result = model.train(df, lstm_config, verbose=1)
    
    training_time = datetime.now() - start_time
    print(f"\n✅ Entraînement terminé en {training_time}")
    
    # Évaluation du modèle
    print("\n\U0001f4ca Évaluation du modèle...")
    evaluation_result = model.evaluate(df, lstm_config)
    
    if evaluation_result:
        print(f"\n\U0001f4c8 Métriques d'évaluation:")
        for metric, value in evaluation_result.items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.6f}")
            else:
                print(f"   {metric}: {value}")
    
    # Test de prédiction
    print("\n\U0001f52e Test de prédiction...")
    test_sequence = df.tail(lstm_config['model_config']['sequence_length'])
    
    prediction_config = {
        "target_columns": lstm_config['training_config']['target_columns'],
        "prediction_steps": 5,  # 5 steps ahead
        "confidence_interval": True
    }
    
    timeframe_info = {
        "end_date": test_sequence.index[-1],
        "pandas_freq": "min"
    }
    
    prediction_result = model.predict(test_sequence, prediction_config, timeframe_info)
    
    if "error" not in prediction_result:
        print("\n✅ Prédiction réussie!")
        print(f"   \U0001f4c8 Future predictions shape: {np.array(prediction_result['future']['Close']).shape}")
        
        # Visualisation des prédictions
        plt.figure(figsize=(12, 8))
        
        # Prix historique
        historical_data = test_sequence['Close'].values
        future_predictions = prediction_result['future']['Close']
        
        # Créer les indices temporelles
        hist_indices = range(len(historical_data))
        future_indices = range(len(historical_data), len(historical_data) + len(future_predictions))
        
        plt.subplot(2, 1, 1)
        plt.plot(hist_indices, historical_data, 'b-', label='Historical', linewidth=2)
        plt.plot(future_indices, future_predictions, 'r--', label='Predictions', linewidth=2)
        
        if 'confidence_lower' in prediction_result and 'confidence_upper' in prediction_result:
            plt.fill_between(future_indices, 
                           prediction_result['confidence_lower']['Close'],
                           prediction_result['confidence_upper']['Close'],
                           alpha=0.3, color='red', label='Confidence Interval')
        
        plt.title('Prédictions LSTM')
        plt.xlabel('Time Steps')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Erreurs de prédiction (si disponibles)
        if 'errors' in prediction_result:
            plt.subplot(2, 1, 2)
            errors = prediction_result['errors']
            if isinstance(errors, dict):
                for col, error_values in errors.items():
                    if isinstance(error_values, (list, np.ndarray)):
                        plt.plot(error_values, label=f'{col} Error')
            plt.title('Erreurs de Prédiction')
            plt.xlabel('Time Steps')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('lstm_predictions.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print(f"❌ Erreur de prédiction: {prediction_result['error']}")
    
    # Sauvegarde du modèle
    print("\n\U0001f4be Sauvegarde du modèle...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"lstm_model_ppo_ready_{timestamp}"
    
    save_result = model.save_model(model_name)
    
    if save_result:
        print(f"✅ Modèle sauvegardé: {model_name}")
        print(f"   \U0001f4c1 Dossier: {save_result['model_path']}")
        print(f"   \U0001f9ee Scaler: {save_result['scaler_path']}")
        print(f"   \U0001f4c8 Config: {save_result['config_path']}")
        
        # Informations pour l'intégration PPO
        sequence_length = lstm_config['model_config']['sequence_length']
        target_columns = lstm_config['training_config']['target_columns']
        prediction_steps = 5  # 5 steps ahead comme configuré
        
        print(f"\n\U0001f680 Informations pour l'intégration PPO:")
        print(f"   \U0001f4c1 Chemin du modèle: {save_result['model_path']}")
        print(f"   \U0001f9ee Chemin du scaler: {save_result['scaler_path']}")
        print(f"   \U0001f4ca Features LSTM: {prediction_steps * len(target_columns)}")
        print(f"   \U0001f4c8 Sequence length: {sequence_length}")
        
        # Créer un fichier d'info pour PPO
        info_file = f"{model_name}_ppo_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Modèle LSTM pour intégration PPO\n")
            f.write(f"Créé le: {datetime.now()}\n")
            f.write(f"Modèle: {save_result['model_path']}\n")
            f.write(f"Scaler: {save_result['scaler_path']}\n")
            f.write(f"Sequence length: {sequence_length}\n")
            f.write(f"Prediction steps: {prediction_steps}\n")
            f.write(f"Target columns: {target_columns}\n")
            f.write(f"LSTM features size: {prediction_steps * len(target_columns)}\n")
            
            # Vérifier les paramètres bidirectionnels et batch normalization
            bidirectional = any(l.get('bidirectional', False) for l in lstm_config['model_config']['layers'])
            batch_norm = any(l.get('batch_normalization', False) for l in lstm_config['model_config']['layers'])
            
            f.write(f"Bidirectional: {bidirectional}\n")
            f.write(f"Batch normalization: {batch_norm}\n")
        
        print(f"\n✅ Fichier d'info créé: {info_file}")
        
    else:
        print("❌ Erreur lors de la sauvegarde du modèle")
    
    print(f"\n\U0001f389 Processus terminé avec succès!")
    print(f"   \U0001f4c8 Modèle entraîné et sauvegardé")
    print(f"   \U0001f4ca Visualisations créées (data_overview.png, lstm_predictions.png)")
    print(f"   \U0001f680 Prêt pour l'intégration PPO")

if __name__ == "__main__":
    main()