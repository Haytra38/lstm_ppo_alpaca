#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test pour démontrer l'affichage amélioré des paramètres du modèle LSTM
"""

import os
import sys
from lstm_model import LSTMModel

def test_lstm_model_display():
    """
    Teste l'affichage des paramètres d'un modèle LSTM
    """
    print("🧠 Test d'affichage des paramètres du modèle LSTM")
    print("=" * 55)
    
    try:
        # Créer une instance temporaire pour accéder à la méthode
        lstm_temp = LSTMModel()
        available_models = lstm_temp.get_available_models()
        
        if not available_models:
            print("❌ Aucun modèle LSTM trouvé dans le dossier saved_models")
            print("💡 Veuillez d'abord entraîner un modèle LSTM")
            return
        
        print(f"\n📋 {len(available_models)} modèle(s) LSTM disponible(s):")
        for i, model_name in enumerate(available_models, 1):
            print(f"  {i}. 🧠 {model_name}")
        
        # Prendre le premier modèle disponible pour la démonstration
        selected_model = available_models[0]
        print(f"\n🔍 Analyse du modèle: {selected_model}")
        
        # Charger et afficher les métadonnées du modèle
        lstm_model = LSTMModel()
        lstm_model.load_model(selected_model)
        
        print(f"\n✅ Modèle LSTM chargé: {selected_model}")
        print("\n📊 Paramètres du modèle LSTM:")
        print("=" * 40)
        print(f"   🔢 Longueur de séquence: {lstm_model.sequence_length} périodes")
        print(f"   🎯 Nombre de prédictions: {lstm_model.nombre_de_predictions} pas en avant")
        print(f"   📈 Nombre de colonnes d'entrée: {lstm_model.nombre_de_colonnes}")
        
        # Affichage des colonnes attendues
        if lstm_model.nombre_de_colonnes == 4:
            print(f"   📋 Colonnes attendues: Open, High, Low, Close")
        elif lstm_model.nombre_de_colonnes == 5:
            print(f"   📋 Colonnes attendues: Open, High, Low, Close, Volume")
        else:
            print(f"   📋 Colonnes attendues: {lstm_model.nombre_de_colonnes} colonnes")
        
        # Configuration détaillée
        if hasattr(lstm_model, 'config') and lstm_model.config:
            print(f"\n⚙️ Configuration d'entraînement:")
            if 'training_date' in lstm_model.config:
                print(f"   📅 Date d'entraînement: {lstm_model.config['training_date']}")
            if 'target_column' in lstm_model.config:
                print(f"   🎯 Colonne cible: {lstm_model.config['target_column']}")
            if 'scaler_type' in lstm_model.config:
                print(f"   📏 Type de normalisation: {lstm_model.config['scaler_type']}")
            if 'epochs' in lstm_model.config:
                print(f"   🔄 Époques d'entraînement: {lstm_model.config['epochs']}")
            if 'batch_size' in lstm_model.config:
                print(f"   📦 Taille de batch: {lstm_model.config['batch_size']}")
        else:
            print(f"\n⚠️ Configuration d'entraînement non disponible")
        
        # Conditions d'utilisation
        print(f"\n⚠️ Conditions d'utilisation:")
        print(f"   • Le modèle nécessite au moins {lstm_model.sequence_length} périodes de données historiques")
        print(f"   • Les données doivent contenir {lstm_model.nombre_de_colonnes} colonnes (OHLC{'V' if lstm_model.nombre_de_colonnes == 5 else ''})")
        print(f"   • Les prédictions sont générées pour {lstm_model.nombre_de_predictions} période(s) future(s)")
        print(f"   • Le modèle utilise la normalisation des données pour améliorer les performances")
        
        # Type de scaler utilisé
        scaler_name = type(lstm_model.scaler).__name__
        if scaler_name == 'RobustScaler':
            print(f"   • Normalisation robuste (résistante aux valeurs aberrantes)")
        elif scaler_name == 'MinMaxScaler':
            print(f"   • Normalisation Min-Max (valeurs entre 0 et 1)")
        else:
            print(f"   • Type de normalisation: {scaler_name}")
        
        print(f"\n✅ Test terminé avec succès!")
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lstm_model_display()