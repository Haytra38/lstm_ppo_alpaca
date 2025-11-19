# Configurateur LSTM - Guide d'utilisation

## 📋 Description

Le script `lstm_config.py` est un outil interactif pour configurer facilement les paramètres de votre modèle LSTM et sélectionner les données d'entraînement. Il offre une interface conviviale pour :

- ⚙️ Définir les hyperparamètres du LSTM
- 📊 Sélectionner les fichiers de données
- 💾 Sauvegarder et charger des configurations
- ✅ Valider les entrées utilisateur
- 📤 Générer du code prêt à l'emploi

## 🚀 Installation et Prérequis

### Dépendances requises
```bash
pip install pandas numpy keras scikit-learn
```

### Structure de projet recommandée
```
LSTM+PPO/
├── data/                    # Dossier contenant vos fichiers CSV
│   └── *.csv
├── configs/                 # Configurations sauvegardées (créé automatiquement)
├── lstm_model.py           # Votre classe LSTMModel
├── lstm_config.py          # Script de configuration
└── README_LSTM_CONFIG.md   # Ce guide
```

## 🎯 Utilisation

### Lancement du configurateur
```bash
python lstm_config.py
```

### Menu principal
Le script propose un menu interactif avec les options suivantes :

1. **📊 Sélectionner un dataset** - Parcourt le dossier `data/` et affiche les fichiers disponibles
2. **🧠 Configurer le modèle LSTM** - Définit l'architecture du réseau
3. **🏋️ Configurer l'entraînement** - Paramètres d'entraînement
4. **📋 Afficher la configuration actuelle** - Résumé des paramètres
5. **💾 Sauvegarder la configuration** - Sauvegarde au format JSON
6. **📂 Charger une configuration** - Charge une configuration existante
7. **📤 Exporter la configuration** - Génère le code Python d'entraînement
8. **❌ Quitter** - Ferme le programme

## 📊 Sélection des données

### Formats supportés
- Fichiers CSV avec colonnes numériques
- Détection automatique des colonnes disponibles
- Affichage de la taille et structure des fichiers

### Exemple d'affichage
```
📊 DATASETS DISPONIBLES
==================================================

1. BTC-USD1Min_2023-12-01_2024-12-31.csv
   📁 Taille: 45.2 MB
   📋 ~525600 lignes (aperçu), 7 colonnes
   🔢 Colonnes numériques: Open, High, Low, Close, Volume
```

## 🧠 Configuration du modèle

### Paramètres configurables

#### Architecture LSTM
- **Nombre de couches** : Profondeur du réseau
- **Unités par couche** : Nombre de neurones LSTM
- **Dropout** : Taux de régularisation (0.0-1.0)
- **Longueur de séquence** : Nombre de pas de temps en entrée

#### Paramètres d'optimisation
- **Taux d'apprentissage** : Learning rate pour l'optimiseur Adam
- **Unités de sortie** : Nombre de prédictions simultanées
- **Nombre de colonnes** : Dimensions des données d'entrée

### Exemple de configuration
```python
{
  "layers": [
    {
      "units": 50,
      "return_sequences": true,
      "dropout": 0.2,
      "sequence_length": 60
    },
    {
      "units": 50,
      "return_sequences": false,
      "dropout": 0.2,
      "sequence_length": 60
    }
  ],
  "dense_units": 1,
  "learning_rate": 0.001,
  "sequence_length": 60,
  "nombre_de_colonnes": 1
}
```

## 🏋️ Configuration de l'entraînement

### Paramètres disponibles
- **Époques** : Nombre d'itérations d'entraînement
- **Taille de batch** : Nombre d'échantillons par batch
- **Colonnes cibles** : Variables à prédire
- **Split de validation** : Proportion des données pour la validation

### Validation automatique
Le script valide automatiquement :
- ✅ Valeurs positives pour les entiers
- ✅ Plages valides pour les pourcentages (0.0-1.0)
- ✅ Existence des colonnes sélectionnées
- ✅ Cohérence des paramètres

## 💾 Gestion des configurations

### Sauvegarde
- Format JSON lisible
- Métadonnées automatiques (date, version)
- Stockage dans le dossier `configs/`

### Chargement
- Liste des configurations disponibles
- Validation de la structure
- Restauration complète des paramètres

### Structure d'une configuration sauvegardée
```json
{
  "model_config": { ... },
  "training_config": { ... },
  "data_config": {
    "selected_file": "BTC-USD1Min_2023-12-01_2024-12-31.csv",
    "file_path": "data/BTC-USD1Min_2023-12-01_2024-12-31.csv",
    "columns_info": {
      "all_columns": ["Date", "Open", "High", "Low", "Close", "Volume"],
      "numeric_columns": ["Open", "High", "Low", "Close", "Volume"]
    }
  },
  "metadata": {
    "created_date": "2024-01-15T10:30:00",
    "saved_date": "2024-01-15T10:35:00",
    "description": "Configuration pour prédiction BTC",
    "version": "1.0"
  }
}
```

## 📤 Export et utilisation

### Génération de code
Le script génère automatiquement :
- Code Python prêt à l'emploi
- Import des modules nécessaires
- Configuration complète du modèle
- Script d'entraînement fonctionnel

### Exemple de code généré
```python
from lstm_model import LSTMModel
import pandas as pd

def main():
    # Charger les données
    data = pd.read_csv('data/BTC-USD1Min_2023-12-01_2024-12-31.csv')
    
    # Créer et configurer le modèle
    lstm_model = LSTMModel()
    model_config = {
        "layers": [...],
        "dense_units": 1,
        "learning_rate": 0.001,
        "sequence_length": 60,
        "nombre_de_colonnes": 1
    }
    lstm_model.create(model_config)
    
    # Configuration d'entraînement
    training_config = {
        "epochs": 100,
        "batch_size": 32,
        "target_columns": ["Close"],
        "validation_split": 0.2
    }
    
    # Entraîner le modèle
    print('🚀 Début de l\'entraînement...')
    history = lstm_model.train(data, training_config)
    print('✅ Entraînement terminé!')
    
    return lstm_model, history

if __name__ == '__main__':
    model, history = main()
```

## 🛠️ Utilisation programmatique

### Utilisation sans interface
```python
from lstm_config import LSTMConfigurator

# Créer le configurateur
config = LSTMConfigurator()

# Sélectionner un dataset
config.select_dataset(1)  # Premier dataset disponible

# Obtenir la configuration pour LSTMModel
lstm_config = config.get_config_for_lstm_model()

# Utiliser avec votre modèle
from lstm_model import LSTMModel
import pandas as pd

lstm_model = LSTMModel()
lstm_model.create(lstm_config['model_config'])

data = pd.read_csv(lstm_config['data_path'])
history = lstm_model.train(data, lstm_config['training_config'])
```

## 🔧 Personnalisation

### Modifier les dossiers par défaut
```python
config = LSTMConfigurator(
    data_folder="mes_donnees",
    config_folder="mes_configs"
)
```

### Ajouter des validations personnalisées
Vous pouvez étendre la classe `LSTMConfigurator` pour ajouter vos propres validations :

```python
class MonConfigurator(LSTMConfigurator):
    def _validate_custom_param(self, value, param_name):
        # Votre logique de validation
        return True
```

## 🐛 Résolution de problèmes

### Erreurs courantes

1. **"Aucun fichier de données trouvé"**
   - Vérifiez que le dossier `data/` existe
   - Assurez-vous que vos fichiers sont au format CSV

2. **"Colonnes invalides"**
   - Vérifiez les noms de colonnes dans votre CSV
   - Utilisez des colonnes numériques pour l'entraînement

3. **"Structure de configuration invalide"**
   - Le fichier de configuration est corrompu
   - Recréez la configuration ou utilisez une sauvegarde

### Logs et débogage
Le script affiche des messages détaillés avec des emojis pour faciliter le suivi :
- ✅ Succès
- ❌ Erreurs
- ⚠️ Avertissements
- 🎯 Actions utilisateur

## 📞 Support

Pour toute question ou problème :
1. Vérifiez ce guide d'utilisation
2. Consultez les messages d'erreur détaillés
3. Vérifiez la compatibilité avec votre classe `LSTMModel`

## 🔄 Mises à jour

Le script est conçu pour être facilement extensible. Vous pouvez :
- Ajouter de nouveaux types de validation
- Supporter d'autres formats de données
- Intégrer de nouveaux paramètres de modèle
- Personnaliser l'interface utilisateur

---

**Bonne configuration ! 🚀**