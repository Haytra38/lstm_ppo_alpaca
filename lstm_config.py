#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Configuration LSTM
============================

Ce script permet de configurer les paramètres du modèle LSTM et de sélectionner
un jeu de données d'entraînement depuis le dossier 'data'.

Fonctionnalités :
- Interface pour définir les hyperparamètres du LSTM
- Sélection des fichiers de données disponibles
- Sauvegarde et chargement de configurations
- Validation des entrées utilisateur
- Affichage clair des paramètres sélectionnés

Auteur: Assistant IA
Date: 2024
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import glob

from lstm_model import LSTMModel
import pandas as pd


def dict_to_python_code(obj, indent=0):
    """
    Convertit un dictionnaire Python en code Python avec les bonnes valeurs booléennes.
    
    Args:
        obj: L'objet à convertir (dict, list, ou valeur primitive)
        indent: Niveau d'indentation actuel
    
    Returns:
        str: Représentation en code Python
    """
    indent_str = '    ' * indent
    
    if isinstance(obj, dict):
        if not obj:
            return '{}'
        
        lines = ['{']
        items = list(obj.items())
        for i, (key, value) in enumerate(items):
            comma = ',' if i < len(items) - 1 else ''
            value_str = dict_to_python_code(value, indent + 1)
            lines.append(f'{indent_str}    "{key}": {value_str}{comma}')
        lines.append(f'{indent_str}}}')
        return '\n'.join(lines)
    
    elif isinstance(obj, list):
        if not obj:
            return '[]'
        
        lines = ['[']
        for i, item in enumerate(obj):
            comma = ',' if i < len(obj) - 1 else ''
            item_str = dict_to_python_code(item, indent + 1)
            lines.append(f'{indent_str}    {item_str}{comma}')
        lines.append(f'{indent_str}]')
        return '\n'.join(lines)
    
    elif isinstance(obj, bool):
        return 'True' if obj else 'False'
    
    elif isinstance(obj, str):
        return f'"{obj}"'
    
    elif isinstance(obj, (int, float)):
        return str(obj)
    
    elif obj is None:
        return 'None'
    
    else:
        return repr(obj)


def save_config_as_python(config, filepath):
    """
    Sauvegarde une configuration au format Python natif.
    
    Args:
        config: Configuration à sauvegarder
        filepath: Chemin du fichier de sauvegarde
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("#!/usr/bin/env python3\n")
            f.write("# -*- coding: utf-8 -*-\n")
            f.write("\"\"\"\n")
            f.write("Configuration LSTM générée automatiquement\n")
            f.write(f"Générée le: {datetime.now().isoformat()}\n")
            f.write("\"\"\"\n\n")
            f.write("# Configuration LSTM\n")
            f.write(f"LSTM_CONFIG = {dict_to_python_code(config)}\n")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde: {e}")
        return False


def load_config_from_python(filepath):
    """
    Charge une configuration depuis un fichier Python.
    
    Args:
        filepath: Chemin du fichier de configuration
    
    Returns:
        dict: Configuration chargée ou None en cas d'erreur
    """
    try:
        # Créer un namespace pour exécuter le fichier
        namespace = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Exécuter le contenu du fichier
        exec(content, namespace)
        
        # Récupérer la configuration
        if 'LSTM_CONFIG' in namespace:
            return namespace['LSTM_CONFIG']
        else:
            print(f"❌ Variable LSTM_CONFIG non trouvée dans {filepath}")
            return None
            
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None

class LSTMConfigurator:
    """
    Classe pour configurer les paramètres du modèle LSTM et gérer les données d'entraînement.
    """
    
    def __init__(self, data_folder: str = "data", config_folder: str = "configs"):
        """
        Initialise le configurateur LSTM.
        
        Args:
            data_folder (str): Dossier contenant les fichiers de données
            config_folder (str): Dossier pour sauvegarder les configurations
        """
        self.data_folder = data_folder
        self.config_folder = config_folder
        self.current_config = self._get_default_config()
        
        # Créer le dossier de configuration s'il n'existe pas
        if not os.path.exists(self.config_folder):
            os.makedirs(self.config_folder)

    def set_parameters(self, params: Dict[str, Any]) -> None:
        mc = self.current_config.get('model_config', {})
        tc = self.current_config.get('training_config', {})
        # Basiques
        if 'sequence_length' in params:
            mc['sequence_length'] = params['sequence_length']
            for layer in mc.get('layers', []):
                layer['sequence_length'] = params['sequence_length']
        if 'learning_rate' in params:
            mc['learning_rate'] = params['learning_rate']
            tc['learning_rate'] = params['learning_rate']
        if 'batch_size' in params:
            tc['batch_size'] = params['batch_size']
        if 'epochs' in params:
            tc['epochs'] = params['epochs']
        if 'prediction_horizon' in params:
            tc['prediction_horizon'] = params['prediction_horizon']
        # Architecture
        if 'hidden_size' in params and mc.get('layers'):
            mc['layers'][0]['units'] = params['hidden_size']
        if 'num_layers' in params:
            num = int(params['num_layers'])
            layers = mc.get('layers', [])
            if num <= len(layers):
                mc['layers'] = layers[:num]
            else:
                base = layers[-1] if layers else {
                    'units': 64, 'return_sequences': True, 'dropout': 0.2,
                    'sequence_length': mc.get('sequence_length', 60),
                    'bidirectional': False, 'batch_normalization': False
                }
                mc['layers'] = layers + [dict(base) for _ in range(num - len(layers))]
        if 'dropout' in params:
            for layer in mc.get('layers', []):
                layer['dropout'] = params['dropout']
        if 'bidirectional' in params:
            for layer in mc.get('layers', []):
                layer['bidirectional'] = params['bidirectional']
        if 'batch_normalization' in params:
            for layer in mc.get('layers', []):
                layer['batch_normalization'] = params['batch_normalization']
        if 'activation_function' in params:
            for layer in mc.get('layers', []):
                layer['activation'] = params['activation_function']
        if 'recurrent_activation' in params:
            for layer in mc.get('layers', []):
                layer['recurrent_activation'] = params['recurrent_activation']
        # Optimiseur et entraînement avancé
        if 'optimizer' in params:
            mc['optimizer'] = params['optimizer']
        if 'weight_decay' in params:
            tc['weight_decay'] = params['weight_decay']
        if 'gradient_clipping' in params:
            mc['gradient_clipping'] = {
                'enabled': True,
                'max_norm': params['gradient_clipping'],
                'norm_type': 2
            }
        if 'use_lr_scheduler' in params:
            tc['use_lr_scheduler'] = params['use_lr_scheduler']
        if 'lr_scheduler_type' in params:
            tc['lr_scheduler_type'] = params['lr_scheduler_type']
        if 'lr_scheduler_patience' in params:
            tc['lr_scheduler_patience'] = params['lr_scheduler_patience']
        if 'lr_scheduler_factor' in params:
            tc['lr_scheduler_factor'] = params['lr_scheduler_factor']
        if 'use_early_stopping' in params:
            tc['use_early_stopping'] = params['use_early_stopping']
        if 'early_stopping_patience' in params:
            tc['early_stopping_patience'] = params['early_stopping_patience']
        if 'early_stopping_min_delta' in params:
            tc['early_stopping_min_delta'] = params['early_stopping_min_delta']
        # Prétraitement
        if 'normalize_data' in params:
            tc.setdefault('data_preprocessing', {}).update({'normalize_data': params['normalize_data']})
        if 'standardization_method' in params:
            tc.setdefault('data_preprocessing', {}).update({'standardization_method': params['standardization_method']})
        if 'feature_engineering' in params:
            tc['feature_engineering'] = params['feature_engineering']
        # Splits et flags
        if 'validation_split' in params:
            tc['validation_split'] = params['validation_split']
        if 'test_split' in params:
            tc['test_split'] = params['test_split']
        if 'shuffle_training' in params:
            tc['shuffle_training'] = params['shuffle_training']
        if 'mixed_precision' in params:
            mc['mixed_precision'] = params['mixed_precision']
        # Logging / sauvegarde
        if 'save_model_frequency' in params:
            tc['save_model_frequency'] = params['save_model_frequency']
        if 'log_training_metrics' in params:
            tc['log_training_metrics'] = params['log_training_metrics']
        if 'verbose_training' in params:
            tc['verbose_training'] = params['verbose_training']
        # Persister
        self.current_config['model_config'] = mc
        self.current_config['training_config'] = tc

    def get_parameters(self) -> Dict[str, Any]:
        return {
            'model_config': self.current_config.get('model_config', {}),
            'training_config': self.current_config.get('training_config', {}),
            'data_config': self.current_config.get('data_config', {}),
        }

    def display_configuration(self) -> None:
        self.display_current_configuration()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Retourne la configuration par défaut optimisée du modèle LSTM pour prédiction 1-minute.
        
        Returns:
            Dict[str, Any]: Configuration par défaut optimisée
        """
        return {
            "model_config": {
                "layers": [
                    {
                        "units": 128,
                        "return_sequences": True,
                        "dropout": 0.2,
                        "sequence_length": 240,  # 4 heures de données 1-minute
                        "bidirectional": True,
                        "batch_normalization": True,
                        "activation": "tanh",  # Activation configurable
                        "recurrent_activation": "sigmoid",  # Activation récurrente
                        "kernel_regularizer": None,  # Régularisation du noyau
                        "recurrent_regularizer": None,  # Régularisation récurrente
                        "dropout_type": "standard",  # Type de dropout: "standard", "recurrent", "both"
                        "use_bias": True,  # Utiliser des biais
                        "unit_forget_bias": True  # Biais pour la porte d'oubli
                    },
                    {
                        "units": 64,
                        "return_sequences": True,
                        "dropout": 0.3,
                        "sequence_length": 240,
                        "bidirectional": False,
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
                        "sequence_length": 240,
                        "bidirectional": False,
                        "batch_normalization": False,
                        "activation": "tanh",
                        "recurrent_activation": "sigmoid",
                        "kernel_regularizer": None,
                        "recurrent_regularizer": None,
                        "dropout_type": "standard",
                        "use_bias": True,
                        "unit_forget_bias": True
                    }
                ],
                "dense_layers": [16, 8],  # Couches denses intermédiaires
                "dense_units": 60,
                "learning_rate": 0.0005,  # Learning rate optimisé
                "sequence_length": 240,  # 4 heures pour données 1-minute
                "nombre_de_colonnes": 4,
                "loss_function": "directional_mse",  # Fonction de perte personnalisée
                "optimizer": "adam",  # Optimiseur configurable
                "optimizer_config": {  # Configuration détaillée de l'optimiseur
                    "beta_1": 0.9,
                    "beta_2": 0.999,
                    "epsilon": 1e-07,
                    "amsgrad": False
                },
                "metrics": ["mae", "mse"],  # Métriques à suivre
                "mixed_precision": True,  # Utiliser mixed precision (float16)
                "gradient_clipping": {  # Clipping des gradients
                    "enabled": True,
                    "max_norm": 1.0,
                    "norm_type": 2
                }
            },
            "training_config": {
                "epochs": 200,  # Plus d'epochs avec early stopping
                "batch_size": 64,  # Batch size optimisé
                "target_columns": ["Close"],
                "validation_split": 0.2,
                "learning_rate": 0.0005,  # Cohérent avec model_config
                "data_preprocessing": {  # Configuration du prétraitement des données
                    "scaler_type": "robust",  # "robust", "minmax", "standard", "robust_conservative", "quantile", "maxabs"
                    "scaler_config": {  # Configuration spécifique au scaler
                        "robust": {
                            "quantile_range": [25.0, 75.0]  # Plage par défaut pour RobustScaler
                        },
                        "robust_conservative": {
                            "quantile_range": [10.0, 90.0]  # Plage plus conservatrice
                        },
                        "minmax": {
                            "feature_range": [0, 1]  # Plage pour MinMaxScaler
                        },
                        "standard": {
                            "with_mean": True,
                            "with_std": True
                        },
                        "quantile": {
                            "n_quantiles": 1000,
                            "output_distribution": "uniform"  # ou "normal"
                        },
                        "maxabs": {
                            "copy": True
                        }
                    },
                    "feature_scaling": {  # Configuration du scaling des features
                        "enabled": True,
                        "columns": None,  # None = toutes les colonnes, ou liste spécifique
                        "fit_on_training_only": True  # Ne fitter le scaler que sur les données d'entraînement
                    },
                    "sequence_scaling": {  # Configuration du scaling des séquences
                        "enabled": True,
                        "method": "per_feature"  # "per_feature", "per_sequence", "global"
                    },
                    "handle_outliers": {  # Gestion des outliers
                        "enabled": False,
                        "method": "clip",  # "clip", "remove", "transform"
                        "threshold": 3.0,  # Seuil en écarts-types
                        "clip_range": [-3.0, 3.0]  # Plage de clipping
                    }
                },
                "early_stopping": {  # Early stopping configurable
                    "enabled": True,
                    "patience": 15,
                    "min_delta": 0.0001,
                    "restore_best_weights": True,
                    "monitor": "val_loss"
                },
                "learning_rate_scheduler": {  # Scheduling du learning rate
                    "enabled": True,
                    "type": "reduce_on_plateau",  # "reduce_on_plateau", "exponential_decay", "cosine_decay"
                    "factor": 0.5,
                    "patience": 10,
                    "min_lr": 1e-06,
                    "cooldown": 5
                },
                "callbacks": {  # Autres callbacks
                    "model_checkpoint": {
                        "enabled": True,
                        "save_best_only": True,
                        "save_weights_only": False,
                        "monitor": "val_loss"
                    },
                    "tensorboard": {
                        "enabled": False,
                        "log_dir": "./logs",
                        "histogram_freq": 1,
                        "write_graph": True
                    },
                    "csv_logger": {
                        "enabled": True,
                        "filename": "training_log.csv",
                        "separator": ",",
                        "append": False
                    }
                },
                "validation_strategy": {  # Stratégie de validation
                    "type": "holdout",  # "holdout", "k_fold", "time_series_split"
                    "k_folds": 5,  # Pour k_fold validation
                    "shuffle": False  # Ne pas mélanger pour des séries temporelles
                },
                "data_augmentation": {  # Augmentation de données
                    "enabled": False,
                    "noise_level": 0.01,
                    "time_warping": False,
                    "scaling": False
                }
            },
            "data_config": {
                "selected_file": None,
                "file_path": None,
                "columns_info": None
            },
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "description": "Configuration optimisée pour prédiction 1-minute avec LSTM bidirectionnel",
                "version": "2.0",
                "optimizations": [
                    "Architecture bidirectionnelle",
                    "BatchNormalization",
                    "Fonction de perte directionnelle",
                    "Hyperparamètres optimisés",
                    "Sequence length étendue (4h)"
                ]
            }
        }
    
    def get_optimized_trading_config(self) -> Dict[str, Any]:
        """
        Retourne une configuration spécialement optimisée pour le trading haute fréquence.
        
        Returns:
            Dict[str, Any]: Configuration optimisée pour trading
        """
        return {
            "model_config": {
                "layers": [
                    {
                        "units": 256,  # Plus de neurones pour capturer la complexité
                        "return_sequences": True,
                        "dropout": 0.15,
                        "sequence_length": 480,  # 8 heures de données 1-minute
                        "bidirectional": True,
                        "batch_normalization": True
                    },
                    {
                        "units": 128,
                        "return_sequences": True,
                        "dropout": 0.25,
                        "sequence_length": 480,
                        "bidirectional": True,
                        "batch_normalization": True
                    },
                    {
                        "units": 64,
                        "return_sequences": False,
                        "dropout": 0.2,
                        "sequence_length": 480,
                        "bidirectional": False,
                        "batch_normalization": True
                    }
                ],
                "dense_layers": [32, 16, 8],  # Plus de couches denses
                "dense_units": 5,  # Prédiction sur 5 pas de temps
                "learning_rate": 0.0003,  # Learning rate plus conservateur
                "sequence_length": 480,  # 8 heures pour données 1-minute
                "nombre_de_colonnes": 4,  # OHLC
                "loss_function": "directional_mse"
            },
            "training_config": {
                "epochs": 300,
                "batch_size": 128,  # Batch size plus grand
                "target_columns": ["Open", "High", "Low", "Close"],
                "validation_split": 0.15,  # Plus de données d'entraînement
                "learning_rate": 0.0003
            },
            "data_config": {
                "selected_file": None,
                "file_path": None,
                "columns_info": None
            },
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "description": "Configuration haute performance pour trading 1-minute multi-colonnes",
                "version": "2.1",
                "target_use_case": "High-frequency trading",
                "optimizations": [
                    "Architecture bidirectionnelle avancée",
                    "Prédiction multi-colonnes OHLC",
                    "Sequence length étendue (8h)",
                    "Fonction de perte directionnelle",
                    "Hyperparamètres haute fréquence",
                    "BatchNormalization optimisée"
                ]
            }
        }
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """
        Liste tous les fichiers de données disponibles dans le dossier data.
        
        Returns:
            List[Dict[str, Any]]: Liste des fichiers avec leurs informations
        """
        datasets = []
        
        if not os.path.exists(self.data_folder):
            print(f"⚠️  Le dossier '{self.data_folder}' n'existe pas.")
            return datasets
        
        # Rechercher les fichiers CSV
        csv_files = glob.glob(os.path.join(self.data_folder, "*.csv"))
        
        for file_path in csv_files:
            try:
                # Lire les premières lignes pour obtenir des informations
                df = pd.read_csv(file_path, nrows=5)
                file_info = {
                    "filename": os.path.basename(file_path),
                    "full_path": file_path,
                    "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                    "columns": list(df.columns),
                    "shape_preview": f"~{len(df)} lignes (aperçu), {len(df.columns)} colonnes",
                    "numeric_columns": list(df.select_dtypes(include=['number']).columns)
                }
                datasets.append(file_info)
            except Exception as e:
                print(f"⚠️  Erreur lors de la lecture de {file_path}: {e}")
        
        return datasets
    
    def display_datasets(self) -> None:
        """
        Affiche la liste des datasets disponibles de manière formatée.
        """
        datasets = self.list_available_datasets()
        
        if not datasets:
            print("❌ Aucun fichier de données trouvé dans le dossier 'data'.")
            return
        
        print("\n📊 DATASETS DISPONIBLES")
        print("=" * 50)
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\n{i}. {dataset['filename']}")
            print(f"   📁 Taille: {dataset['size_mb']} MB")
            print(f"   📋 {dataset['shape_preview']}")
            print(f"   🔢 Colonnes numériques: {', '.join(dataset['numeric_columns'][:5])}")
            if len(dataset['numeric_columns']) > 5:
                print(f"      ... et {len(dataset['numeric_columns']) - 5} autres")
    
    def select_dataset(self, dataset_index: Optional[int] = None) -> bool:
        """
        Sélectionne un dataset pour l'entraînement.
        
        Args:
            dataset_index (Optional[int]): Index du dataset à sélectionner (1-based)
        
        Returns:
            bool: True si la sélection a réussi, False sinon
        """
        datasets = self.list_available_datasets()
        
        if not datasets:
            print("❌ Aucun dataset disponible.")
            return False
        
        if dataset_index is None:
            self.display_datasets()
            try:
                dataset_index = int(input(f"\n🎯 Sélectionnez un dataset (1-{len(datasets)}): "))
            except ValueError:
                print("❌ Veuillez entrer un nombre valide.")
                return False
        
        if 1 <= dataset_index <= len(datasets):
            selected_dataset = datasets[dataset_index - 1]
            
            # Mettre à jour la configuration
            self.current_config["data_config"]["selected_file"] = selected_dataset["filename"]
            self.current_config["data_config"]["file_path"] = selected_dataset["full_path"]
            self.current_config["data_config"]["columns_info"] = {
                "all_columns": selected_dataset["columns"],
                "numeric_columns": selected_dataset["numeric_columns"]
            }
            
            print(f"✅ Dataset sélectionné: {selected_dataset['filename']}")
            return True
        else:
            print(f"❌ Index invalide. Veuillez choisir entre 1 et {len(datasets)}.")
            return False
    
    def configure_model_parameters(self) -> None:
        """
        Interface interactive pour configurer les paramètres du modèle LSTM.
        """
        print("\n🔧 CONFIGURATION DU MODÈLE LSTM")
        print("=" * 40)
        
        try:
            # Configuration avancée ou simple
            print("\n📋 Type de configuration:")
            print("1. 🚀 Configuration simple (paramètres essentiels)")
            print("2. ⚙️  Configuration avancée (tous les paramètres)")
            
            config_type = input("\n➤ Votre choix (1-2, défaut: 1): ").strip() or "1"
            
            if config_type == "2":
                self._configure_advanced_model_parameters()
            else:
                self._configure_simple_model_parameters()
                
        except KeyboardInterrupt:
            print("\n⚠️  Configuration annulée par l'utilisateur.")

    def _configure_simple_model_parameters(self) -> None:
        """
        Configuration simplifiée du modèle LSTM.
        """
        print("\n🚀 CONFIGURATION SIMPLE DU MODÈLE")
        print("-" * 40)
        
        # Configuration des couches LSTM
        print("\n📚 Configuration des couches LSTM:")
        
        num_layers = int(input(f"Nombre de couches LSTM (actuel: {len(self.current_config['model_config']['layers'])}): ") or len(self.current_config['model_config']['layers']))
        
        if not self._validate_positive_int(num_layers, "nombre de couches"):
            return
        
        layers = []
        for i in range(num_layers):
            print(f"\n--- Couche {i+1} ---")
            
            # Unités
            default_units = 50 if i < len(self.current_config['model_config']['layers']) else 50
            units = int(input(f"Nombre d'unités (défaut: {default_units}): ") or default_units)
            
            if not self._validate_positive_int(units, "nombre d'unités"):
                return
            
            # Bidirectionnel
            default_bidirectional = self.current_config['model_config']['layers'][i].get('bidirectional', False) if i < len(self.current_config['model_config']['layers']) else False
            bidirectional_input = input(f"Bidirectionnel? (o/N, défaut: {'o' if default_bidirectional else 'n'}): ").strip().lower()
            bidirectional = bidirectional_input in ['o', 'oui', 'y', 'yes'] if bidirectional_input else default_bidirectional
            
            # Dropout
            default_dropout = 0.2
            dropout = float(input(f"Taux de dropout (0.0-1.0, défaut: {default_dropout}): ") or default_dropout)
            
            if not self._validate_float_range(dropout, 0.0, 1.0, "taux de dropout"):
                return
            
            # Batch normalization
            default_bn = self.current_config['model_config']['layers'][i].get('batch_normalization', False) if i < len(self.current_config['model_config']['layers']) else False
            bn_input = input(f"Batch normalization? (o/N, défaut: {'o' if default_bn else 'n'}): ").strip().lower()
            batch_normalization = bn_input in ['o', 'oui', 'y', 'yes'] if bn_input else default_bn
            
            layers.append({
                "units": units,
                "return_sequences": i < num_layers - 1,
                "dropout": dropout,
                "sequence_length": self.current_config['model_config']['sequence_length'],
                "bidirectional": bidirectional,
                "batch_normalization": batch_normalization,
                "activation": "tanh",
                "recurrent_activation": "sigmoid",
                "kernel_regularizer": None,
                "recurrent_regularizer": None,
                "dropout_type": "standard",
                "use_bias": True,
                "unit_forget_bias": True
            })
        
        # Configuration générale
        print("\n⚙️  Configuration générale:")
        
        # Sequence length
        default_seq_len = self.current_config['model_config']['sequence_length']
        sequence_length = int(input(f"Longueur de séquence (défaut: {default_seq_len}): ") or default_seq_len)
        
        if not self._validate_positive_int(sequence_length, "longueur de séquence"):
            return
        
        # Dense units
        default_dense = self.current_config['model_config']['dense_units']
        dense_units = int(input(f"Nombre de prédictions futures (défaut: {default_dense}): ") or default_dense)
        
        if not self._validate_positive_int(dense_units, "unités de sortie"):
            return
        
        # Learning rate
        default_lr = self.current_config['model_config']['learning_rate']
        learning_rate = float(input(f"Taux d'apprentissage (défaut: {default_lr}): ") or default_lr)
        
        if not self._validate_positive_float(learning_rate, "taux d'apprentissage"):
            return
        
        # Optimiseur
        print("\n📊 Optimiseur disponible:")
        print("1. Adam (recommandé)")
        print("2. RMSprop")
        print("3. SGD")
        optimizer_choice = input("➤ Choix (1-3, défaut: 1): ").strip() or "1"
        
        optimizers = {"1": "adam", "2": "rmsprop", "3": "sgd"}
        optimizer = optimizers.get(optimizer_choice, "adam")
        
        # Mixed precision
        default_mixed = self.current_config['model_config'].get('mixed_precision', True)
        mixed_input = input(f"Mixed precision (GPU accélération)? (O/n, défaut: {'o' if default_mixed else 'n'}): ").strip().lower()
        mixed_precision = mixed_input not in ['n', 'non', 'no'] if mixed_input else default_mixed
        
        # Nombre de colonnes
        self._configure_column_count()
        
        # Mettre à jour la configuration
        self.current_config['model_config'].update({
            "layers": layers,
            "dense_units": dense_units,
            "learning_rate": learning_rate,
            "sequence_length": sequence_length,
            "optimizer": optimizer,
            "mixed_precision": mixed_precision
        })
        
        print("✅ Configuration du modèle mise à jour avec succès!")

    def _configure_advanced_model_parameters(self) -> None:
        """
        Configuration avancée du modèle LSTM avec tous les paramètres.
        """
        print("\n⚙️  CONFIGURATION AVANCÉE DU MODÈLE")
        print("-" * 50)
        
        # Configuration des couches LSTM
        print("\n📚 Configuration des couches LSTM:")
        
        num_layers = int(input(f"Nombre de couches LSTM (actuel: {len(self.current_config['model_config']['layers'])}): ") or len(self.current_config['model_config']['layers']))
        
        if not self._validate_positive_int(num_layers, "nombre de couches"):
            return
        
        layers = []
        for i in range(num_layers):
            print(f"\n--- Couche {i+1} ---")
            
            # Unités
            default_units = 50 if i < len(self.current_config['model_config']['layers']) else 50
            units = int(input(f"Nombre d'unités (défaut: {default_units}): ") or default_units)
            
            if not self._validate_positive_int(units, "nombre d'unités"):
                return
            
            # Return sequences
            default_return_seq = i < num_layers - 1
            return_seq_input = input(f"Return sequences? (O/n, défaut: {'o' if default_return_seq else 'n'}): ").strip().lower()
            return_sequences = return_seq_input not in ['n', 'non', 'no'] if return_seq_input else default_return_seq
            
            # Dropout
            default_dropout = 0.2
            dropout = float(input(f"Taux de dropout (0.0-1.0, défaut: {default_dropout}): ") or default_dropout)
            
            if not self._validate_float_range(dropout, 0.0, 1.0, "taux de dropout"):
                return
            
            # Type de dropout
            print("\n🎯 Types de dropout:")
            print("1. Standard (dropout sur les entrées)")
            print("2. Récurent (dropout sur les états récurrents)")
            print("3. Les deux")
            dropout_type_choice = input("➤ Choix (1-3, défaut: 1): ").strip() or "1"
            dropout_types = {"1": "standard", "2": "recurrent", "3": "both"}
            dropout_type = dropout_types.get(dropout_type_choice, "standard")
            
            # Bidirectionnel
            default_bidirectional = self.current_config['model_config']['layers'][i].get('bidirectional', False) if i < len(self.current_config['model_config']['layers']) else False
            bidirectional_input = input(f"Bidirectionnel? (o/N, défaut: {'o' if default_bidirectional else 'n'}): ").strip().lower()
            bidirectional = bidirectional_input in ['o', 'oui', 'y', 'yes'] if bidirectional_input else default_bidirectional
            
            # Batch normalization
            default_bn = self.current_config['model_config']['layers'][i].get('batch_normalization', False) if i < len(self.current_config['model_config']['layers']) else False
            bn_input = input(f"Batch normalization? (o/N, défaut: {'o' if default_bn else 'n'}): ").strip().lower()
            batch_normalization = bn_input in ['o', 'oui', 'y', 'yes'] if bn_input else default_bn
            
            # Activation
            print("\n⚡ Fonctions d'activation:")
            print("1. tanh (recommandé pour LSTM)")
            print("2. relu")
            print("3. sigmoid")
            print("4. linear")
            activation_choice = input("➤ Activation (1-4, défaut: 1): ").strip() or "1"
            activations = {"1": "tanh", "2": "relu", "3": "sigmoid", "4": "linear"}
            activation = activations.get(activation_choice, "tanh")
            
            # Activation récurrente
            print("\n🔄 Activation récurrente:")
            print("1. sigmoid (standard)")
            print("2. hard_sigmoid")
            print("3. tanh")
            recurrent_activation_choice = input("➤ Activation récurrente (1-3, défaut: 1): ").strip() or "1"
            recurrent_activations = {"1": "sigmoid", "2": "hard_sigmoid", "3": "tanh"}
            recurrent_activation = recurrent_activations.get(recurrent_activation_choice, "sigmoid")
            
            # Biais
            default_bias = True
            bias_input = input(f"Utiliser des biais? (O/n, défaut: {'o' if default_bias else 'n'}): ").strip().lower()
            use_bias = bias_input not in ['n', 'non', 'no'] if bias_input else default_bias
            
            # Biais de porte d'oubli
            default_forget_bias = True
            forget_bias_input = input(f"Biais pour porte d'oubli? (O/n, défaut: {'o' if default_forget_bias else 'n'}): ").strip().lower()
            unit_forget_bias = forget_bias_input not in ['n', 'non', 'no'] if forget_bias_input else default_forget_bias
            
            layers.append({
                "units": units,
                "return_sequences": return_sequences,
                "dropout": dropout,
                "sequence_length": self.current_config['model_config']['sequence_length'],
                "bidirectional": bidirectional,
                "batch_normalization": batch_normalization,
                "activation": activation,
                "recurrent_activation": recurrent_activation,
                "kernel_regularizer": None,
                "recurrent_regularizer": None,
                "dropout_type": dropout_type,
                "use_bias": use_bias,
                "unit_forget_bias": unit_forget_bias
            })
        
        # Configuration générale
        print("\n⚙️  Configuration générale:")
        
        # Sequence length
        default_seq_len = self.current_config['model_config']['sequence_length']
        sequence_length = int(input(f"Longueur de séquence (défaut: {default_seq_len}): ") or default_seq_len)
        
        if not self._validate_positive_int(sequence_length, "longueur de séquence"):
            return
        
        # Dense layers
        print("\n🔗 Configuration des couches denses intermédiaires:")
        current_dense_layers = self.current_config['model_config'].get('dense_layers', [16, 8])
        dense_layers_input = input(f"Couches denses (ex: 16,8, défaut: {','.join(map(str, current_dense_layers))}): ").strip()
        
        if dense_layers_input:
            try:
                dense_layers = [int(x.strip()) for x in dense_layers_input.split(',')]
                if not all(self._validate_positive_int(x, "unités de couche dense") for x in dense_layers):
                    return
            except ValueError:
                print("❌ Format invalide. Utilisation des valeurs par défaut.")
                dense_layers = current_dense_layers
        else:
            dense_layers = current_dense_layers
        
        # Dense units
        default_dense = self.current_config['model_config']['dense_units']
        dense_units = int(input(f"Nombre de prédictions futures (défaut: {default_dense}): ") or default_dense)
        
        if not self._validate_positive_int(dense_units, "unités de sortie"):
            return
        
        # Learning rate
        default_lr = self.current_config['model_config']['learning_rate']
        learning_rate = float(input(f"Taux d'apprentissage (défaut: {default_lr}): ") or default_lr)
        
        if not self._validate_positive_float(learning_rate, "taux d'apprentissage"):
            return
        
        # Optimiseur
        print("\n📊 Optimiseurs disponibles:")
        print("1. Adam (recommandé)")
        print("2. RMSprop")
        print("3. SGD")
        print("4. AdamW")
        print("5. Nadam")
        current_optimizer = self.current_config['model_config'].get('optimizer', 'adam')
        optimizer_map = {"1": "adam", "2": "rmsprop", "3": "sgd", "4": "adamw", "5": "nadam"}
        optimizer_choice = input(f"➤ Optimiseur actuel: {current_optimizer}, nouveau (1-5, laisser vide pour garder): ").strip()
        optimizer = optimizer_map.get(optimizer_choice, current_optimizer) if optimizer_choice else current_optimizer
        
        # Configuration de l'optimiseur
        if optimizer != current_optimizer or not self.current_config['model_config'].get('optimizer_config'):
            print(f"\n⚙️  Configuration de {optimizer}:")
            if optimizer == "adam":
                beta_1 = float(input(f"  Beta 1 (défaut: 0.9): ").strip() or "0.9")
                beta_2 = float(input(f"  Beta 2 (défaut: 0.999): ").strip() or "0.999")
                epsilon = float(input(f"  Epsilon (défaut: 1e-07): ").strip() or "1e-07")
                amsgrad = input(f"  AMSGrad? (o/N, défaut: n): ").strip().lower() in ['o', 'oui', 'y', 'yes']
                optimizer_config = {"beta_1": beta_1, "beta_2": beta_2, "epsilon": epsilon, "amsgrad": amsgrad}
            else:
                optimizer_config = self.current_config['model_config'].get('optimizer_config', {})
        else:
            optimizer_config = self.current_config['model_config'].get('optimizer_config', {})
        
        # Mixed precision
        default_mixed = self.current_config['model_config'].get('mixed_precision', True)
        mixed_input = input(f"Mixed precision (GPU accélération)? (O/n, défaut: {'o' if default_mixed else 'n'}): ").strip().lower()
        mixed_precision = mixed_input not in ['n', 'non', 'no'] if mixed_input else default_mixed
        
        # Gradient clipping
        print("\n✂️  Gradient clipping:")
        default_clipping = self.current_config['model_config'].get('gradient_clipping', {}).get('enabled', True)
        clipping_input = input(f"Activer le clipping? (O/n, défaut: {'o' if default_clipping else 'n'}): ").strip().lower()
        clipping_enabled = clipping_input not in ['n', 'non', 'no'] if clipping_input else default_clipping
        
        if clipping_enabled:
            max_norm = float(input(f"  Max norm (défaut: 1.0): ").strip() or "1.0")
            norm_type = int(input(f"  Norm type (1 ou 2, défaut: 2): ").strip() or "2")
            gradient_clipping = {"enabled": True, "max_norm": max_norm, "norm_type": norm_type}
        else:
            gradient_clipping = {"enabled": False, "max_norm": 1.0, "norm_type": 2}
        
        # Métriques
        print("\n📈 Métriques de suivi:")
        print("1. MAE + MSE (recommandé)")
        print("2. MAE uniquement")
        print("3. MSE uniquement")
        print("4. Personnalisé")
        
        current_metrics = self.current_config['model_config'].get('metrics', ["mae", "mse"])
        metrics_choice = input(f"➤ Métriques (1-4, défaut: 1): ").strip() or "1"
        
        if metrics_choice == "4":
            custom_metrics = input("  Entrez les métriques séparées par des virgules (ex: mae,mse,mape): ").strip()
            metrics = [m.strip() for m in custom_metrics.split(',')] if custom_metrics else current_metrics
        else:
            metrics_map = {"1": ["mae", "mse"], "2": ["mae"], "3": ["mse"]}
            metrics = metrics_map.get(metrics_choice, ["mae", "mse"])
        
        # Nombre de colonnes
        self._configure_column_count()
        
        # Mettre à jour la configuration
        self.current_config['model_config'].update({
            "layers": layers,
            "dense_layers": dense_layers,
            "dense_units": dense_units,
            "learning_rate": learning_rate,
            "sequence_length": sequence_length,
            "optimizer": optimizer,
            "optimizer_config": optimizer_config,
            "metrics": metrics,
            "mixed_precision": mixed_precision,
            "gradient_clipping": gradient_clipping
        })
        
        print("✅ Configuration avancée du modèle mise à jour avec succès!")

    def _configure_column_count(self) -> None:
        """
        Configure le nombre de colonnes d'entrée avec synchronisation intelligente.
        """
        current_cols = self.current_config['model_config']['nombre_de_colonnes']
        target_cols = self.current_config['training_config']['target_columns']
        expected_cols = len(target_cols)
        
        print(f"\n🔢 Configuration du nombre de colonnes d'entrée:")
        print(f"   📊 Colonnes cibles actuelles: {', '.join(target_cols)} ({expected_cols} colonnes)")
        print(f"   🎯 Nombre de colonnes configuré dans le modèle: {current_cols}")
        
        if current_cols != expected_cols:
            print(f"   ⚠️  Désynchronisation détectée! Le modèle est configuré pour {current_cols} colonnes mais {expected_cols} colonnes cibles sont sélectionnées.")
            sync_choice = input(f"   🔄 Synchroniser automatiquement avec les colonnes cibles? (O/n): ").strip().lower()
            if sync_choice in ['', 'o', 'oui', 'y', 'yes']:
                nombre_de_colonnes = expected_cols
                print(f"   ✅ Synchronisation effectuée: {nombre_de_colonnes} colonnes")
            else:
                nombre_de_colonnes = int(input(f"   🎯 Nombre de colonnes d'entrée (défaut: {current_cols}): ") or current_cols)
                if not self._validate_positive_int(nombre_de_colonnes, "nombre de colonnes"):
                    return
        else:
            print(f"   ✅ Configuration synchronisée")
            manual_choice = input(f"   🔧 Modifier manuellement le nombre de colonnes? (o/N): ").strip().lower()
            if manual_choice in ['o', 'oui', 'y', 'yes']:
                nombre_de_colonnes = int(input(f"   🎯 Nouveau nombre de colonnes (défaut: {current_cols}): ") or current_cols)
                if not self._validate_positive_int(nombre_de_colonnes, "nombre de colonnes"):
                    return
            else:
                nombre_de_colonnes = current_cols
        
        self.current_config['model_config']['nombre_de_colonnes'] = nombre_de_colonnes
    
    def configure_training_parameters(self) -> None:
        """
        Interface interactive pour configurer les paramètres d'entraînement.
        """
        print("\n🏋️  CONFIGURATION DE L'ENTRAÎNEMENT")
        print("=" * 40)
        
        try:
            # Époques
            default_epochs = self.current_config['training_config']['epochs']
            epochs = int(input(f"Nombre d'époques (défaut: {default_epochs}): ") or default_epochs)
            
            if not self._validate_positive_int(epochs, "nombre d'époques"):
                return
            
            # Taille de batch
            default_batch = self.current_config['training_config']['batch_size']
            batch_size = int(input(f"Taille du batch (défaut: {default_batch}): ") or default_batch)
            
            if not self._validate_positive_int(batch_size, "taille du batch"):
                return
            
            # Sélection intelligente des colonnes cibles
            target_columns = self._select_target_columns_interactive()
            if target_columns is None:
                return  # L'utilisateur a annulé ou il y a eu une erreur
            
            # Split de validation
            default_split = self.current_config['training_config']['validation_split']
            validation_split = float(input(f"Proportion de validation (0.0-1.0, défaut: {default_split}): ") or default_split)
            
            if not self._validate_float_range(validation_split, 0.0, 1.0, "proportion de validation"):
                return
            
            # Mettre à jour la configuration
            self.current_config['training_config'].update({
                "epochs": epochs,
                "batch_size": batch_size,
                "target_columns": target_columns,
                "validation_split": validation_split
            })
            
            # Synchroniser automatiquement le nombre de colonnes dans model_config
            nombre_colonnes = len(target_columns)
            self.current_config['model_config']['nombre_de_colonnes'] = nombre_colonnes
            
            print(f"✅ Configuration d'entraînement mise à jour avec succès!")
            print(f"🔄 Nombre de colonnes d'entrée automatiquement mis à jour: {nombre_colonnes}")
            
        except ValueError as e:
            print(f"❌ Erreur de saisie: {e}")
        except KeyboardInterrupt:
            print("\n⚠️  Configuration annulée par l'utilisateur.")
    
    def _select_target_columns_interactive(self) -> Optional[List[str]]:
        """
        Interface interactive intelligente pour sélectionner les colonnes cibles.
        
        Returns:
            Optional[List[str]]: Liste des colonnes sélectionnées ou None si annulé
        """
        if not self.current_config['data_config']['columns_info']:
            print("⚠️  Aucune information sur les colonnes disponible. Veuillez d'abord sélectionner un dataset.")
            return self.current_config['training_config']['target_columns']
        
        available_cols = self.current_config['data_config']['columns_info']['numeric_columns']
        current_targets = self.current_config['training_config']['target_columns']
        
        print("\n🎯 SÉLECTION DES COLONNES CIBLES")
        print("=" * 40)
        print(f"📊 Colonnes numériques disponibles: {', '.join(available_cols)}")
        print(f"🔄 Colonnes actuellement sélectionnées: {', '.join(current_targets)}")
        
        # Options prédéfinies intelligentes
        preset_options = {
            "1": {"name": "Prix de clôture uniquement", "columns": ["Close"]},
            "2": {"name": "Prix OHLC (Open, High, Low, Close)", "columns": ["Open", "High", "Low", "Close"]},
            "3": {"name": "Prix OHLC + Volume", "columns": ["Open", "High", "Low", "Close", "Volume"]},
            "4": {"name": "Toutes les colonnes numériques", "columns": available_cols},
            "5": {"name": "Sélection personnalisée", "columns": []},
            "6": {"name": "Garder la sélection actuelle", "columns": current_targets}
        }
        
        print("\n📋 Options de sélection:")
        for key, option in preset_options.items():
            if key == "2" and not all(col in available_cols for col in option["columns"]):
                continue  # Skip OHLC if not all columns are available
            if key == "3" and not all(col in available_cols for col in option["columns"]):
                continue  # Skip OHLC+Volume if not all columns are available
            
            available_preset_cols = [col for col in option["columns"] if col in available_cols]
            if available_preset_cols or key in ["5", "6"]:
                if key == "4":
                    print(f"{key}. {option['name']} ({len(available_cols)} colonnes)")
                elif key == "5":
                    print(f"{key}. {option['name']}")
                elif key == "6":
                    print(f"{key}. {option['name']} ({len(current_targets)} colonnes)")
                else:
                    print(f"{key}. {option['name']} ({len(available_preset_cols)} colonnes)")
        
        try:
            choice = input("\n🎯 Votre choix (1-6): ").strip()
            
            if choice in preset_options:
                if choice == "5":  # Sélection personnalisée
                    return self._custom_column_selection(available_cols)
                elif choice == "6":  # Garder la sélection actuelle
                    return current_targets
                else:
                    selected_cols = [col for col in preset_options[choice]["columns"] if col in available_cols]
                    if selected_cols:
                        print(f"✅ Colonnes sélectionnées: {', '.join(selected_cols)}")
                        return selected_cols
                    else:
                        print("❌ Aucune colonne valide dans cette option.")
                        return None
            else:
                print("❌ Choix invalide. Veuillez choisir entre 1 et 6.")
                return None
                
        except KeyboardInterrupt:
            print("\n⚠️  Sélection annulée par l'utilisateur.")
            return None
    
    def _custom_column_selection(self, available_cols: List[str]) -> Optional[List[str]]:
        """
        Interface pour sélection personnalisée des colonnes.
        
        Args:
            available_cols (List[str]): Colonnes disponibles
        
        Returns:
            Optional[List[str]]: Colonnes sélectionnées ou None si annulé
        """
        print("\n🔧 SÉLECTION PERSONNALISÉE")
        print("=" * 30)
        
        # Afficher les colonnes avec des numéros
        print("📊 Colonnes disponibles:")
        for i, col in enumerate(available_cols, 1):
            print(f"{i}. {col}")
        
        print("\n💡 Instructions:")
        print("   - Entrez les numéros des colonnes séparés par des virgules (ex: 1,3,4)")
        print("   - Ou entrez les noms des colonnes séparés par des virgules (ex: Open,Close,Volume)")
        print("   - Appuyez sur Entrée pour annuler")
        
        try:
            selection = input("\n🎯 Votre sélection: ").strip()
            
            if not selection:
                return None
            
            # Essayer d'abord comme des numéros
            if selection.replace(',', '').replace(' ', '').isdigit():
                indices = [int(x.strip()) for x in selection.split(',')]
                selected_cols = []
                for idx in indices:
                    if 1 <= idx <= len(available_cols):
                        selected_cols.append(available_cols[idx - 1])
                    else:
                        print(f"❌ Index invalide: {idx}. Doit être entre 1 et {len(available_cols)}.")
                        return None
            else:
                # Traiter comme des noms de colonnes
                selected_cols = [col.strip() for col in selection.split(',')]
                invalid_cols = [col for col in selected_cols if col not in available_cols]
                if invalid_cols:
                    print(f"❌ Colonnes invalides: {', '.join(invalid_cols)}")
                    return None
            
            if selected_cols:
                print(f"✅ Colonnes sélectionnées: {', '.join(selected_cols)}")
                return selected_cols
            else:
                print("❌ Aucune colonne sélectionnée.")
                return None
                
        except (ValueError, KeyboardInterrupt):
            print("\n⚠️  Sélection annulée.")
            return None
    
    def _select_columns_for_model(self, expected_columns: int) -> Optional[List[str]]:
        """
        Interface pour sélectionner exactement le nombre de colonnes requis par le modèle.
        
        Args:
            expected_columns (int): Nombre de colonnes attendu par le modèle
            
        Returns:
            Optional[List[str]]: Liste des colonnes sélectionnées ou None si annulé
        """
        if not self.current_config['data_config']['columns_info']:
            print("⚠️  Aucune information sur les colonnes disponible. Veuillez d'abord sélectionner un dataset.")
            return None
        
        available_cols = self.current_config['data_config']['columns_info']['numeric_columns']
        
        print(f"\n🎯 SÉLECTION DE {expected_columns} COLONNES POUR LE MODÈLE")
        print("=" * 50)
        print(f"📊 Colonnes numériques disponibles: {', '.join(available_cols)}")
        print(f"🔢 Nombre de colonnes requis: {expected_columns}")
        
        # Options prédéfinies basées sur le nombre de colonnes requis
        preset_options = {}
        option_num = 1
        
        # Option 1 colonne
        if expected_columns == 1:
            preset_options[str(option_num)] = {"name": "Prix de clôture uniquement", "columns": ["Close"]}
            option_num += 1
        
        # Option 4 colonnes (OHLC)
        if expected_columns == 4 and all(col in available_cols for col in ["Open", "High", "Low", "Close"]):
            preset_options[str(option_num)] = {"name": "Prix OHLC (Open, High, Low, Close)", "columns": ["Open", "High", "Low", "Close"]}
            option_num += 1
        
        # Option 5 colonnes (OHLC + Volume)
        if expected_columns == 5 and all(col in available_cols for col in ["Open", "High", "Low", "Close", "Volume"]):
            preset_options[str(option_num)] = {"name": "Prix OHLC + Volume", "columns": ["Open", "High", "Low", "Close", "Volume"]}
            option_num += 1
        
        # Option toutes les colonnes si le nombre correspond
        if len(available_cols) == expected_columns:
            preset_options[str(option_num)] = {"name": "Toutes les colonnes numériques", "columns": available_cols}
            option_num += 1
        
        # Option sélection personnalisée
        preset_options[str(option_num)] = {"name": "Sélection personnalisée", "columns": []}
        
        print("\n📋 Options de sélection:")
        for key, option in preset_options.items():
            if option["columns"]:
                print(f"{key}. {option['name']} ({len(option['columns'])} colonnes)")
            else:
                print(f"{key}. {option['name']}")
        
        try:
            choice = input(f"\n🎯 Votre choix (1-{len(preset_options)}): ").strip()
            
            if choice in preset_options:
                if not preset_options[choice]["columns"]:  # Sélection personnalisée
                    return self._custom_column_selection_with_count(available_cols, expected_columns)
                else:
                    selected_cols = preset_options[choice]["columns"]
                    print(f"✅ Colonnes sélectionnées: {', '.join(selected_cols)}")
                    return selected_cols
            else:
                print(f"❌ Choix invalide. Veuillez choisir entre 1 et {len(preset_options)}.")
                return None
                
        except KeyboardInterrupt:
            print("\n⚠️  Sélection annulée par l'utilisateur.")
            return None
    
    def _custom_column_selection_with_count(self, available_cols: List[str], expected_count: int) -> Optional[List[str]]:
        """
        Sélection personnalisée avec validation du nombre de colonnes.
        
        Args:
            available_cols (List[str]): Liste des colonnes disponibles
            expected_count (int): Nombre de colonnes attendu
            
        Returns:
            Optional[List[str]]: Liste des colonnes sélectionnées ou None si annulé
        """
        print(f"\n📋 Colonnes disponibles: {', '.join(available_cols)}")
        print(f"\n🎯 Sélection personnalisée de {expected_count} colonnes:")
        print("   - Entrez les noms des colonnes séparés par des virgules")
        print("   - Exemple: Open,High,Low,Close,Volume")
        print("   - Tapez 'annuler' pour annuler")
        
        try:
            user_input = input(f"\n📝 Sélectionnez exactement {expected_count} colonnes: ").strip()
            
            if user_input.lower() == 'annuler':
                return None
            
            if not user_input:
                print("❌ Aucune colonne saisie.")
                return None
            
            # Parser l'entrée utilisateur
            selected_cols = [col.strip() for col in user_input.split(',')]
            
            # Valider les colonnes
            invalid_cols = [col for col in selected_cols if col not in available_cols]
            if invalid_cols:
                print(f"❌ Colonnes invalides: {', '.join(invalid_cols)}")
                print(f"   Colonnes disponibles: {', '.join(available_cols)}")
                return None
            
            # Supprimer les doublons tout en préservant l'ordre
            unique_cols = []
            for col in selected_cols:
                if col not in unique_cols:
                    unique_cols.append(col)
            
            # Vérifier le nombre de colonnes
            if len(unique_cols) != expected_count:
                print(f"❌ Nombre incorrect de colonnes: {len(unique_cols)} sélectionnées, {expected_count} requises.")
                return None
            
            print(f"✅ Colonnes sélectionnées: {', '.join(unique_cols)}")
            return unique_cols
            
        except KeyboardInterrupt:
            print("\n⚠️  Sélection annulée par l'utilisateur.")
            return None
    
    def _validate_positive_int(self, value: int, param_name: str) -> bool:
        """
        Valide qu'un entier est positif.
        
        Args:
            value (int): Valeur à valider
            param_name (str): Nom du paramètre pour les messages d'erreur
        
        Returns:
            bool: True si valide, False sinon
        """
        if value <= 0:
            print(f"❌ Le {param_name} doit être un entier positif (reçu: {value}).")
            return False
        return True
    
    def _validate_positive_float(self, value: float, param_name: str) -> bool:
        """
        Valide qu'un float est positif.
        
        Args:
            value (float): Valeur à valider
            param_name (str): Nom du paramètre pour les messages d'erreur
        
        Returns:
            bool: True si valide, False sinon
        """
        if value <= 0:
            print(f"❌ Le {param_name} doit être un nombre positif (reçu: {value}).")
            return False
        return True
    
    def _validate_float_range(self, value: float, min_val: float, max_val: float, param_name: str) -> bool:
        """
        Valide qu'un float est dans une plage donnée.
        
        Args:
            value (float): Valeur à valider
            min_val (float): Valeur minimale
            max_val (float): Valeur maximale
            param_name (str): Nom du paramètre pour les messages d'erreur
        
        Returns:
            bool: True si valide, False sinon
        """
        if not (min_val <= value <= max_val):
            print(f"❌ Le {param_name} doit être entre {min_val} et {max_val} (reçu: {value}).")
            return False
        return True
    
    def save_configuration(self, config_name: Optional[str] = None) -> bool:
        """
        Sauvegarde la configuration actuelle.
        
        Args:
            config_name (Optional[str]): Nom de la configuration
        
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        if config_name is None:
            config_name = input("\n💾 Nom de la configuration à sauvegarder: ").strip()
        
        if not config_name:
            print("❌ Le nom de la configuration ne peut pas être vide.")
            return False
        
        # Ajouter l'extension .py si elle n'est pas présente
        if not config_name.endswith('.py'):
            config_name += '.py'
        
        config_path = os.path.join(self.config_folder, config_name)
        
        # Mettre à jour les métadonnées
        self.current_config['metadata'].update({
            "saved_date": datetime.now().isoformat(),
            "config_name": config_name
        })
        
        return save_config_as_python(self.current_config, config_path)
    
    def load_configuration(self, config_name: Optional[str] = None) -> bool:
        """
        Charge une configuration sauvegardée.
        
        Args:
            config_name (Optional[str]): Nom de la configuration à charger
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        if config_name is None:
            # Lister les configurations disponibles
            configs = self.list_saved_configurations()
            if not configs:
                print("❌ Aucune configuration sauvegardée trouvée.")
                return False
            
            print("\n📂 Configurations disponibles:")
            for i, config in enumerate(configs, 1):
                print(f"{i}. {config}")
            
            try:
                choice = int(input(f"\n🎯 Sélectionnez une configuration (1-{len(configs)}): "))
                if 1 <= choice <= len(configs):
                    config_name = configs[choice - 1]
                else:
                    print(f"❌ Choix invalide. Veuillez choisir entre 1 et {len(configs)}.")
                    return False
            except ValueError:
                print("❌ Veuillez entrer un nombre valide.")
                return False
        
        # Ajouter l'extension .py si elle n'est pas présente
        if not config_name.endswith('.py'):
            config_name += '.py'
        
        config_path = os.path.join(self.config_folder, config_name)
        
        if not os.path.exists(config_path):
            print(f"❌ Configuration non trouvée: {config_path}")
            return False
        
        loaded_config = load_config_from_python(config_path)
        
        if loaded_config is None:
            return False
        
        # Valider la structure de la configuration
        if self._validate_config_structure(loaded_config):
            self.current_config = loaded_config
            print(f"✅ Configuration chargée: {config_name}")
            return True
        else:
            print(f"❌ Structure de configuration invalide dans {config_name}")
            return False
    
    def list_saved_configurations(self) -> List[str]:
        """
        Liste toutes les configurations sauvegardées.
        
        Returns:
            List[str]: Liste des noms de configurations
        """
        if not os.path.exists(self.config_folder):
            return []
        
        configs = []
        for file in os.listdir(self.config_folder):
            if file.endswith('.py'):
                configs.append(file)
        
        return sorted(configs)
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """
        Valide la structure d'une configuration.
        
        Args:
            config (Dict[str, Any]): Configuration à valider
        
        Returns:
            bool: True si la structure est valide, False sinon
        """
        required_keys = ['model_config', 'training_config', 'data_config', 'metadata']
        
        for key in required_keys:
            if key not in config:
                print(f"❌ Clé manquante dans la configuration: {key}")
                return False
        
        # Validation plus détaillée si nécessaire
        model_config = config['model_config']
        if 'layers' not in model_config or not isinstance(model_config['layers'], list):
            print("❌ Configuration des couches invalide")
            return False
        
        return True
    
    def display_current_configuration(self) -> None:
        """
        Affiche la configuration actuelle de manière formatée.
        """
        print("\n📋 CONFIGURATION ACTUELLE")
        print("=" * 50)
        
        # Informations sur le dataset
        data_config = self.current_config['data_config']
        print(f"\n📊 Dataset:")
        if data_config['selected_file']:
            print(f"   📁 Fichier: {data_config['selected_file']}")
            if data_config['columns_info']:
                cols = data_config['columns_info']['numeric_columns']
                print(f"   🔢 Colonnes numériques: {', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}")
        else:
            print("   ❌ Aucun dataset sélectionné")
        
        # Configuration du modèle
        model_config = self.current_config['model_config']
        print(f"\n🧠 Modèle LSTM:")
        print(f"   🏗️  Nombre de couches: {len(model_config['layers'])}")
        for i, layer in enumerate(model_config['layers'], 1):
            print(f"   📚 Couche {i}: {layer['units']} unités, dropout={layer['dropout']}")
        print(f"   🎯 Unités de sortie: {model_config['dense_units']}")
        print(f"   📏 Longueur de séquence: {model_config['sequence_length']}")
        print(f"   🎓 Taux d'apprentissage: {model_config['learning_rate']}")
        
        # Configuration d'entraînement
        training_config = self.current_config['training_config']
        print(f"\n🏋️  Entraînement:")
        print(f"   🔄 Époques: {training_config['epochs']}")
        print(f"   📦 Taille de batch: {training_config['batch_size']}")
        print(f"   🎯 Colonnes cibles: {', '.join(training_config['target_columns'])}")
        print(f"   ✅ Split de validation: {training_config['validation_split']}")
        
        # Métadonnées
        metadata = self.current_config['metadata']
        print(f"\n📝 Métadonnées:")
        print(f"   📅 Créé le: {metadata.get('created_date', 'N/A')[:19]}")
        print(f"   💾 Sauvegardé le: {metadata.get('saved_date', 'Non sauvegardé')[:19]}")
        print(f"   📖 Description: {metadata.get('description', 'N/A')}")
    
    def get_config_for_lstm_model(self) -> Dict[str, Any]:
        """
        Retourne la configuration formatée pour la classe LSTMModel.
        
        Returns:
            Dict[str, Any]: Configuration compatible avec LSTMModel
        """
        return {
            "model_config": self.current_config['model_config'],
            "training_config": self.current_config['training_config'],
            "data_path": self.current_config['data_config']['file_path'],
            "scaler_config": self.current_config['training_config'].get('data_preprocessing', {
                'scaler_type': 'robust',
                'scaler_config': {}
            })
        }
    
    def select_optimized_config(self) -> None:
        """
        Permet à l'utilisateur de sélectionner une configuration optimisée prédéfinie.
        """
        print("\n⚡ CONFIGURATIONS OPTIMISÉES")
        print("=" * 40)
        print("Sélectionnez une configuration optimisée:")
        print("\n1. 📈 Configuration par défaut (améliorée)")
        print("   - 3 couches LSTM avec bidirectionnel et batch normalization")
        print("   - Séquence de 240 pas (4h de données 1-minute)")
        print("   - Fonction de perte directionnelle")
        print("   - Optimiseur Adam avec gradient clipping")
        
        print("\n2. 🚀 Configuration trading haute fréquence")
        print("   - Architecture plus complexe avec 3 couches LSTM")
        print("   - Séquence de 480 pas (8h de données 1-minute)")
        print("   - Couches denses intermédiaires")
        print("   - Optimisée pour le trading 1-minute")
        
        print("\n3. 🔧 Configuration personnalisée (actuelle)")
        print("   - Garder la configuration actuelle")
        
        try:
            choice = input("\n🎯 Votre choix (1-3): ").strip()
            
            if choice == '1':
                self.current_config = self._get_default_config()
                print("\n✅ Configuration par défaut améliorée appliquée!")
                print("📊 Cette configuration est optimisée pour les prédictions 1-minute avec:")
                print("   - Architecture bidirectionnelle")
                print("   - Batch normalization")
                print("   - Fonction de perte directionnelle")
                
            elif choice == '2':
                self.current_config = self.get_optimized_trading_config()
                print("\n✅ Configuration trading haute fréquence appliquée!")
                print("📊 Cette configuration est optimisée pour le trading haute performance avec:")
                print("   - Architecture complexe à 3 couches")
                print("   - Séquences longues (8h de données)")
                print("   - Couches denses intermédiaires")
                
            elif choice == '3':
                print("\n✅ Configuration actuelle conservée.")
                
            else:
                print("❌ Choix invalide. Configuration actuelle conservée.")
                
        except Exception as e:
            print(f"❌ Erreur lors de la sélection: {e}")
    
    def run_interactive_setup(self) -> None:
        """
        Lance l'interface interactive complète pour configurer le modèle LSTM.
        """
        print("\n🚀 CONFIGURATEUR LSTM INTERACTIF")
        print("=" * 50)
        print("Bienvenue dans le configurateur de modèle LSTM!")
        print("Ce script vous guidera pour configurer votre modèle d'apprentissage.")
        
        while True:
            print("\n📋 MENU PRINCIPAL")
            print("-" * 20)
            print("1. 📊 Sélectionner un dataset")
            print("2. ⚡ Sélectionner une configuration optimisée")
            print("3. 🧠 Configurer le modèle LSTM (manuel)")
            print("4. 🏋️  Configurer l'entraînement")
            print("5. 📋 Afficher la configuration actuelle")
            print("6. 💾 Sauvegarder la configuration")
            print("7. 📂 Charger une configuration")
            print("8. 🚀 Lancer l'entraînement LSTM (nouveau ou continuation)")
            print("9. 📂 Charger et afficher les informations d'un modèle")
            print("10. 🧪 Tester un modèle existant")
            print("11. 📤 Exporter la configuration pour LSTMModel")
            print("12. ❌ Quitter")
            
            try:
                choice = input("\n🎯 Votre choix (1-12): ").strip()
                
                if choice == '1':
                    self.select_dataset()
                elif choice == '2':
                    self.select_optimized_config()
                elif choice == '3':
                    self.configure_model_parameters()
                elif choice == '4':
                    self.configure_training_parameters()
                elif choice == '5':
                    self.display_current_configuration()
                elif choice == '6':
                    self.save_configuration()
                elif choice == '7':
                    self.load_configuration()
                elif choice == '8':
                    trained_model = self.train_lstm_model()
                    if trained_model:
                        print("\n🎉 Modèle entraîné avec succès! Vous pouvez maintenant l'utiliser pour des prédictions.")
                elif choice == '9':
                    loaded_model = self.load_model_info()
                    if loaded_model:
                        print("\n🎉 Informations du modèle affichées avec succès!")
                elif choice == '10':
                    self.test_lstm_model()
                elif choice == '11':
                    self._export_config_for_training()
                elif choice == '12':
                    print("\n👋 Au revoir! Configuration terminée.")
                    break
                else:
                    print("❌ Choix invalide. Veuillez choisir entre 1 et 12.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Configuration interrompue par l'utilisateur. Au revoir!")
                break
            except Exception as e:
                print(f"❌ Erreur inattendue: {e}")
    
    def train_lstm_model(self) -> Optional[Any]:
        """
        Lance l'entraînement du modèle LSTM avec la configuration actuelle.
        Permet de créer un nouveau modèle ou de continuer l'entraînement d'un modèle existant.
        
        Returns:
            Optional[Any]: Le modèle entraîné ou None en cas d'erreur
        """
        if not self.current_config['data_config']['selected_file']:
            print("❌ Veuillez d'abord sélectionner un dataset.")
            return None
        
        try:
           
            print("\n🚀 ENTRAÎNEMENT LSTM")
            print("=" * 50)
            
            # Choix entre nouveau modèle ou continuation
            print("\n🎯 TYPE D'ENTRAÎNEMENT:")
            print("1. 🆕 Créer et entraîner un nouveau modèle")
            print("2. 🔄 Continuer l'entraînement d'un modèle existant")
            
            while True:
                try:
                    choice = input("\n🎯 Votre choix (1-2): ").strip()
                    if choice in ['1', '2']:
                        break
                    else:
                        print("❌ Veuillez choisir 1 ou 2.")
                except KeyboardInterrupt:
                    print("\n⚠️  Entraînement annulé par l'utilisateur.")
                    return None
            
            lstm_model = LSTMModel()
            
            if choice == '1':
                # Nouveau modèle
                print("\n🆕 CRÉATION D'UN NOUVEAU MODÈLE")
                print("-" * 40)
                
                # Créer et configurer le modèle
                print("🧠 Création du modèle LSTM...")
                model_config = self.current_config['model_config']
                result = lstm_model.create(model_config)
                print(f"✅ {result['message']}")
                
            else:
                # Continuer l'entraînement d'un modèle existant
                print("\n🔄 CONTINUATION D'ENTRAÎNEMENT")
                print("-" * 40)
                
                # Lister les modèles disponibles
                available_models = lstm_model.get_available_models()
                
                if not available_models:
                    print("❌ Aucun modèle sauvegardé trouvé.")
                    return None
                
                print("\n📋 Modèles disponibles:")
                for i, model_name in enumerate(available_models, 1):
                    print(f"{i}. {model_name}")
                
                try:
                    model_choice = int(input(f"\n🎯 Sélectionnez un modèle (1-{len(available_models)}): "))
                    if 1 <= model_choice <= len(available_models):
                        selected_model = available_models[model_choice - 1]
                    else:
                        print(f"❌ Choix invalide. Veuillez choisir entre 1 et {len(available_models)}.")
                        return None
                except ValueError:
                    print("❌ Veuillez entrer un nombre valide.")
                    return None
                
                # Charger le modèle existant
                print(f"\n📥 Chargement du modèle: {selected_model}")
                try:
                    lstm_model.load_model(selected_model)
                    print("✅ Modèle chargé avec succès!")
                except Exception as e:
                    print(f"❌ Erreur lors du chargement du modèle: {e}")
                    return None
            
            # Charger les données
            data_path = self.current_config['data_config']['file_path']
            print(f"\n📊 Chargement des données: {self.current_config['data_config']['selected_file']}")
            data = pd.read_csv(data_path)
            print(f"✅ Données chargées: {data.shape[0]} lignes, {data.shape[1]} colonnes")
            
            # Afficher la configuration d'entraînement
            training_config = self.current_config['training_config']
            print(f"\n🏋️  Configuration d'entraînement:")
            print(f"   📊 Époques: {training_config['epochs']}")
            print(f"   📦 Batch size: {training_config['batch_size']}")
            print(f"   🎯 Colonnes cibles: {', '.join(training_config['target_columns'])}")
            print(f"   ✅ Validation split: {training_config['validation_split']}")
            
            # Demander confirmation
            action_text = "nouveau modèle" if choice == '1' else f"continuation du modèle {selected_model if choice == '2' else ''}"
            confirm = input(f"\n❓ Voulez-vous lancer l'entraînement du {action_text} ? (o/N): ").strip().lower()
            if confirm not in ['o', 'oui', 'y', 'yes']:
                print("⚠️  Entraînement annulé.")
                return None
            
            # Lancer l'entraînement
            print(f"\n🚀 Début de l'entraînement...")
            print("-" * 30)
            
            history = lstm_model.train(data, training_config)
            
            print("\n✅ Entraînement terminé avec succès!")
            
            # Proposer de sauvegarder le modèle
            save_model = input("\n💾 Voulez-vous sauvegarder le modèle ? (O/n): ").strip().lower()
            if save_model not in ['n', 'non', 'no']:
                if choice == '1':
                    # Nouveau modèle
                    model_name = input("📝 Nom du modèle (laisser vide pour nom automatique): ").strip()
                    if not model_name:
                        model_name = f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                else:
                    # Modèle existant - proposer un nouveau nom
                    model_name = input(f"📝 Nom du modèle (défaut: {selected_model}_continued): ").strip()
                    if not model_name:
                        model_name = f"{selected_model}_continued"
                
                save_result = lstm_model.save_model(model_name)
                if save_result:
                    print(f"✅ Modèle sauvegardé: {model_name}")
                else:
                    print("❌ Erreur lors de la sauvegarde du modèle")
            
            return lstm_model
            
        except ImportError:
            print("❌ Erreur: Impossible d'importer LSTMModel. Vérifiez que lstm_model.py est présent.")
            return None
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement: {e}")
            return None
    
    def save_trained_model(self, model: Any, model_name: Optional[str] = None) -> bool:
        """
        Sauvegarde un modèle LSTM entraîné.
        
        Args:
            model: Le modèle LSTM entraîné
            model_name: Nom du modèle (optionnel)
        
        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            if model_name is None:
                model_name = input("📝 Nom du modèle à sauvegarder: ").strip()
            
            if not model_name:
                model_name = f"lstm_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            print(f"\n💾 Sauvegarde du modèle: {model_name}")
            result = model.save_model(model_name)
            
            if result:
                print(f"✅ Modèle sauvegardé avec succès: {model_name}")
                return True
            else:
                print("❌ Erreur lors de la sauvegarde du modèle")
                return False
                
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde: {e}")
            return False
    
    def load_model_info(self) -> Optional[Any]:
        """
        Charge un modèle existant et affiche ses informations détaillées.
        
        Returns:
            Optional[Any]: Le modèle chargé ou None en cas d'erreur
        """
        try:
            from lstm_model import LSTMModel
            import pandas as pd
            
            print("\n📂 CHARGEMENT ET INFORMATIONS DU MODÈLE")
            print("=" * 50)
            
            # Lister les modèles disponibles
            lstm_temp = LSTMModel()
            available_models = lstm_temp.get_available_models()
            
            if not available_models:
                print("❌ Aucun modèle sauvegardé trouvé.")
                return None
            
            print("\n📋 Modèles disponibles:")
            for i, model_name in enumerate(available_models, 1):
                print(f"{i}. {model_name}")
            
            try:
                choice = int(input(f"\n🎯 Sélectionnez un modèle (1-{len(available_models)}): "))
                if 1 <= choice <= len(available_models):
                    selected_model = available_models[choice - 1]
                else:
                    print(f"❌ Choix invalide. Veuillez choisir entre 1 et {len(available_models)}.")
                    return None
            except ValueError:
                print("❌ Veuillez entrer un nombre valide.")
                return None
            
            # Charger le modèle
            print(f"\n📥 Chargement du modèle: {selected_model}")
            lstm_model = LSTMModel()
            
            try:
                lstm_model.load_model(selected_model)
                print("✅ Modèle chargé avec succès!")
            except Exception as e:
                print(f"❌ Erreur lors du chargement du modèle: {e}")
                return None
            
            # Afficher les informations détaillées du modèle
            print("\n📊 INFORMATIONS DU MODÈLE")
            print("=" * 40)
            
            # Informations de base
            print(f"📁 Nom du modèle: {selected_model}")
            
            # Paramètres du modèle
            if hasattr(lstm_model, 'nombre_de_colonnes'):
                print(f"🔢 Nombre de colonnes en input: {lstm_model.nombre_de_colonnes}")
            
            if hasattr(lstm_model, 'sequence_length'):
                print(f"📏 Longueur de séquence: {lstm_model.sequence_length}")
            
            if hasattr(lstm_model, 'nombre_de_predictions'):
                print(f"🎯 Nombre de prédictions: {lstm_model.nombre_de_predictions}")
            
            # Architecture du modèle
            if hasattr(lstm_model, 'model') and lstm_model.model is not None:
                print(f"\n🏗️  Architecture du modèle:")
                try:
                    # Compter les couches
                    total_layers = len(lstm_model.model.layers)
                    lstm_layers = sum(1 for layer in lstm_model.model.layers if 'lstm' in layer.name.lower())
                    dense_layers = sum(1 for layer in lstm_model.model.layers if 'dense' in layer.name.lower())
                    
                    print(f"   📚 Total des couches: {total_layers}")
                    print(f"   🧠 Couches LSTM: {lstm_layers}")
                    print(f"   🔗 Couches Dense: {dense_layers}")
                    
                    # Paramètres totaux
                    total_params = lstm_model.model.count_params()
                    print(f"   ⚖️  Paramètres totaux: {total_params:,}")
                    
                    # Forme d'entrée et de sortie
                    input_shape = lstm_model.model.input_shape
                    output_shape = lstm_model.model.output_shape
                    print(f"   📥 Forme d'entrée: {input_shape}")
                    print(f"   📤 Forme de sortie: {output_shape}")
                    
                except Exception as e:
                    print(f"   ⚠️  Impossible d'analyser l'architecture: {e}")
            
            # Informations sur l'optimiseur
            if hasattr(lstm_model, 'model') and lstm_model.model is not None:
                try:
                    optimizer = lstm_model.model.optimizer
                    if optimizer:
                        print(f"\n⚙️  Optimiseur: {optimizer.__class__.__name__}")
                        if hasattr(optimizer, 'learning_rate'):
                            lr = float(optimizer.learning_rate)
                            print(f"   📈 Taux d'apprentissage: {lr}")
                except Exception as e:
                    print(f"   ⚠️  Informations optimiseur non disponibles: {e}")
            
            # Informations sur le scaler
            if hasattr(lstm_model, 'scaler') and lstm_model.scaler is not None:
                print(f"\n🔧 Normalisation: {lstm_model.scaler.__class__.__name__}")
                if hasattr(lstm_model.scaler, 'feature_range'):
                    print(f"   📊 Plage de normalisation: {lstm_model.scaler.feature_range}")
            
            # Historique d'entraînement (si disponible)
            try:
                history = lstm_model.load_training_history(selected_model)
                if history:
                    print(f"\n📈 HISTORIQUE D'ENTRAÎNEMENT")
                    print("-" * 30)
                    
                    # Dernières métriques
                    if 'loss' in history.history:
                        final_loss = history.history['loss'][-1]
                        print(f"   📉 Perte finale: {final_loss:.6f}")
                    
                    if 'val_loss' in history.history:
                        final_val_loss = history.history['val_loss'][-1]
                        print(f"   📊 Perte validation finale: {final_val_loss:.6f}")
                    
                    # Nombre d'époques
                    epochs_trained = len(history.history.get('loss', []))
                    print(f"   🔄 Époques d'entraînement: {epochs_trained}")
                    
                    # Métriques supplémentaires
                    for metric in history.history.keys():
                        if metric not in ['loss', 'val_loss'] and not metric.startswith('val_'):
                            final_metric = history.history[metric][-1]
                            print(f"   📊 {metric.capitalize()} final: {final_metric:.6f}")
                            
            except Exception as e:
                print(f"\n⚠️  Historique d'entraînement non disponible: {e}")
            
            print("\n✅ Informations du modèle affichées avec succès!")
            return lstm_model
            
        except ImportError:
            print("❌ Erreur: Impossible d'importer LSTMModel. Vérifiez que lstm_model.py est présent.")
            return None
        except Exception as e:
            print(f"❌ Erreur lors du chargement/continuation: {e}")
            return None
    
    def test_lstm_model(self) -> None:
        """
        Teste un modèle LSTM existant en effectuant des prédictions.
        Utilise la logique de chargement de l'option 8 pour s'assurer que le modèle est correctement chargé.
        """
        try:
            from lstm_model import LSTMModel
            import pandas as pd
            
            print("\n🧪 TEST DU MODÈLE LSTM")
            print("=" * 50)
            
            # Utiliser la logique de chargement de l'option 8 (load_model_info)
            print("📂 Chargement et vérification du modèle...")
            loaded_model = self.load_model_info()
            
            if not loaded_model:
                print("❌ Impossible de charger le modèle. Test annulé.")
                return
            
            # Le modèle est maintenant chargé et vérifié
            lstm_model = loaded_model
            
            # Vérifications supplémentaires pour s'assurer que le modèle est prêt pour les tests
            print("\n🔍 VÉRIFICATIONS DU MODÈLE POUR LE TEST")
            print("-" * 40)
            
            # Vérifier que le modèle a tous les attributs nécessaires
            required_attributes = ['model', 'scaler', 'sequence_length', 'nombre_de_predictions', 'nombre_de_colonnes']
            missing_attributes = []
            
            for attr in required_attributes:
                if not hasattr(lstm_model, attr) or getattr(lstm_model, attr) is None:
                    missing_attributes.append(attr)
            
            if missing_attributes:
                print(f"❌ Attributs manquants dans le modèle: {', '.join(missing_attributes)}")
                print("Le modèle n'est pas correctement configuré pour les tests.")
                return
            
            print("✅ Tous les attributs requis sont présents")
            print(f"✅ Modèle Keras: {'Chargé' if lstm_model.model is not None else 'Non chargé'}")
            print(f"✅ Scaler: {'Chargé' if lstm_model.scaler is not None else 'Non chargé'}")
            print(f"✅ Configuration: Séquence={lstm_model.sequence_length}, Colonnes={lstm_model.nombre_de_colonnes}, Prédictions={lstm_model.nombre_de_predictions}")
            
            # Vérifier si un dataset est sélectionné
            if not self.current_config['data_config']['selected_file']:
                print("⚠️  Aucun dataset sélectionné. Veuillez sélectionner un dataset pour effectuer des prédictions.")
                return
            
            # Charger les données
            data_path = self.current_config['data_config']['file_path']
            print(f"📊 Chargement des données: {self.current_config['data_config']['selected_file']}")
            data = pd.read_csv(data_path)
            print(f"✅ Données chargées: {data.shape[0]} lignes, {data.shape[1]} colonnes")
            
            # Configuration de prédiction - vérifier la compatibilité des colonnes
            expected_columns = getattr(lstm_model, 'nombre_de_colonnes', 1)
            
            if hasattr(lstm_model, 'target_columns') and lstm_model.target_columns is not None:
                model_target_columns = lstm_model.target_columns
                print(f"\n🎯 Colonnes cibles du modèle chargé: {', '.join(model_target_columns)}")
                print(f"📊 Nombre de colonnes attendu par le modèle: {expected_columns}")
                
                # Vérifier si le nombre de colonnes correspond
                if len(model_target_columns) == expected_columns:
                    target_columns = model_target_columns
                    print("✅ Configuration des colonnes compatible avec le modèle")
                else:
                    print(f"⚠️  Incompatibilité détectée: modèle attend {expected_columns} colonnes, mais {len(model_target_columns)} configurées")
                    print("\n🔧 Sélection des colonnes pour correspondre au modèle:")
                    target_columns = self._select_columns_for_model(expected_columns)
                    if target_columns is None:
                        print("❌ Sélection des colonnes annulée.")
                        return
            else:
                print(f"\n⚠️  Le modèle ne contient pas d'information sur les colonnes cibles")
                print(f"📊 Nombre de colonnes attendu par le modèle: {expected_columns}")
                print("\n🔧 Sélection des colonnes pour correspondre au modèle:")
                target_columns = self._select_columns_for_model(expected_columns)
                if target_columns is None:
                    print("❌ Sélection des colonnes annulée.")
                    return
            
            # Demander le nombre de pas de prédiction
            default_steps = getattr(lstm_model, 'nombre_de_predictions', 10)
            while True:
                try:
                    prediction_steps = input(f"\n📈 Nombre de pas de temps à prédire (défaut: {default_steps}): ").strip()
                    if not prediction_steps:
                        prediction_steps = default_steps
                    else:
                        prediction_steps = int(prediction_steps)
                    
                    if prediction_steps > 0:
                        break
                    else:
                        print("❌ Le nombre de pas doit être positif.")
                except ValueError:
                    print("❌ Veuillez entrer un nombre valide.")
            
            # Configuration de prédiction
            predict_config = {
                "target_columns": target_columns,
                "prediction_steps": prediction_steps,
                "confidence_interval": False
            }
            
            # Informations de timeframe (optionnel)
            timeframe_info = None
            if 'Date' in data.columns or 'date' in data.columns or 'Time' in data.columns or isinstance(data.index, pd.DatetimeIndex):
                # Essayer de détecter la dernière date et la fréquence
                try:
                    dates_series = None
                    if 'Date' in data.columns:
                        dates_series = pd.to_datetime(data['Date'])
                        last_date = dates_series.iloc[-1]
                    elif 'date' in data.columns:
                        dates_series = pd.to_datetime(data['date'])
                        last_date = dates_series.iloc[-1]
                    elif 'Time' in data.columns:
                        dates_series = pd.to_datetime(data['Time'])
                        last_date = dates_series.iloc[-1]
                    else:
                        dates_series = data.index
                        last_date = data.index[-1]
                    
                    # Détection automatique de la fréquence
                    detected_freq = "H"  # Fréquence par défaut
                    if len(dates_series) >= 2:
                        try:
                            # Calculer la différence moyenne entre les dates
                            time_diffs = dates_series.diff().dropna()
                            avg_diff = time_diffs.mean()
                            
                            # Déterminer la fréquence basée sur la différence moyenne
                            if avg_diff <= pd.Timedelta(minutes=1):
                                detected_freq = "min"  # Minutes
                            elif avg_diff <= pd.Timedelta(hours=1):
                                detected_freq = "h"  # Heures
                            elif avg_diff <= pd.Timedelta(days=1):
                                detected_freq = "D"  # Jours
                            elif avg_diff <= pd.Timedelta(weeks=1):
                                detected_freq = "W"  # Semaines
                            else:
                                detected_freq = "ME"  # Mois (nouvelle notation)
                            
                            print(f"📅 Fréquence détectée: {detected_freq} (différence moyenne: {avg_diff})")
                        except Exception as freq_error:
                            print(f"⚠️  Impossible de détecter la fréquence automatiquement: {freq_error}")
                    
                    timeframe_info = {
                        "end_date": last_date,
                        "pandas_freq": detected_freq
                    }
                except Exception as e:
                    print(f"⚠️  Impossible de détecter les dates: {e}")
            
            # Effectuer la prédiction
            print("\n🚀 Génération des prédictions...")
            print("-" * 30)
            
            results = lstm_model.predict(data, predict_config, timeframe_info)
            
            # Assigner les résultats à self.result pour la sauvegarde
            lstm_model.result = results
            
            if "error" in results:
                print(f"❌ Erreur lors de la prédiction: {results['error']}")
                return
            
            # Afficher les résultats
            print("\n✅ Prédictions générées avec succès!")
            print("\n📊 RÉSULTATS DES PRÉDICTIONS")
            print("=" * 40)
            
            # Afficher les prédictions futures
            future_predictions = results['future']
            future_dates = results.get('future_dates', [])
            
            print("\n🔮 Prédictions futures:")
            for i in range(prediction_steps):
                if i < len(future_dates) and future_dates[i] != f"t+{i+1}":
                    date_str = future_dates[i]
                else:
                    date_str = f"t+{i+1}"
                print(f"   📅 {date_str}:")
                for col in target_columns:
                    if col in future_predictions and i < len(future_predictions[col]):
                        value = future_predictions[col][i]
                        print(f"      {col}: {value:.4f}")
            
            # Afficher quelques valeurs historiques pour comparaison
            if 'historical' in results:
                print("\n📈 Dernières valeurs historiques (pour comparaison):")
                historical = results['historical']
                historical_dates = results.get('historical_dates', [])
                
                # Afficher les 3 dernières valeurs
                for col in target_columns:
                    if col in historical:
                        values = historical[col][-3:]  # 3 dernières valeurs
                        print(f"   {col}: {' → '.join([f'{v:.4f}' for v in values])}")
            
            # Proposer de sauvegarder les résultats
            save_results = input("\n💾 Voulez-vous sauvegarder les prédictions dans un fichier Excel ? (o/N): ").strip().lower()
            if save_results in ['o', 'oui', 'y', 'yes']:
                try:
                    filename = input("📝 Nom du fichier (laisser vide pour nom automatique): ").strip()
                    if not filename:
                        from datetime import datetime
                        filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    elif not filename.endswith('.xlsx'):
                        filename += '.xlsx'
                    
                    # Utiliser la méthode save_predictions_to_excel du modèle
                    lstm_model.save_predictions_to_excel(filename)
                    print(f"✅ Prédictions sauvegardées dans: {filename}")
                except Exception as e:
                    print(f"❌ Erreur lors de la sauvegarde: {e}")
            
            print("\n🎉 Test du modèle terminé avec succès!")
            
        except Exception as e:
            print(f"❌ Erreur lors du test du modèle: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _export_config_for_training(self) -> None:
        """
        Exporte la configuration dans un format prêt pour l'entraînement.
        """
        if not self.current_config['data_config']['selected_file']:
            print("❌ Veuillez d'abord sélectionner un dataset.")
            return
        
        config_for_training = self.get_config_for_lstm_model()
        
        print("\n📤 CONFIGURATION POUR LSTMMODEL")
        print("=" * 40)
        print("\n🐍 Code Python pour utiliser cette configuration:")
        print("-" * 50)
        
        print("```python")
        print("from lstm_model import LSTMModel")
        print("import pandas as pd")
        print("")
        print("# Charger les données")
        print(f"data = pd.read_csv('{config_for_training['data_path']}')")
        print("")
        print("# Créer et configurer le modèle")
        print("lstm_model = LSTMModel()")
        print(f"model_config = {dict_to_python_code(config_for_training['model_config'])}")
        print("lstm_model.create(model_config)")
        print("")
        print("# Configuration d'entraînement")
        print(f"training_config = {dict_to_python_code(config_for_training['training_config'])}")
        print("")
        print("# Entraîner le modèle")
        print("history = lstm_model.train(data, training_config)")
        print("```")
        
        # Sauvegarder aussi dans un fichier
        export_filename = f"lstm_training_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        try:
            with open(export_filename, 'w', encoding='utf-8') as f:
                f.write(f"#!/usr/bin/env python3\n")
                f.write(f"# -*- coding: utf-8 -*-\n")
                f.write(f"\"\"\"\n")
                f.write(f"Script d'entraînement LSTM généré automatiquement\n")
                f.write(f"Généré le: {datetime.now().isoformat()}\n")
                f.write(f"\"\"\"\n\n")
                f.write(f"from lstm_model import LSTMModel\n")
                f.write(f"import pandas as pd\n\n")
                f.write(f"def main():\n")
                f.write(f"    # Charger les données\n")
                f.write(f"    data = pd.read_csv('{config_for_training['data_path']}')\n\n")
                f.write(f"    # Créer et configurer le modèle\n")
                f.write(f"    lstm_model = LSTMModel()\n")
                f.write(f"    model_config = {dict_to_python_code(config_for_training['model_config'], indent=1)}\n")
                f.write(f"    lstm_model.create(model_config)\n\n")
                f.write(f"    # Configuration d'entraînement\n")
                f.write(f"    training_config = {dict_to_python_code(config_for_training['training_config'], indent=1)}\n\n")
                f.write(f"    # Entraîner le modèle\n")
                f.write(f"    print('🚀 Début de l\\'entraînement...')\n")
                f.write(f"    history = lstm_model.train(data, training_config)\n")
                f.write(f"    print('✅ Entraînement terminé!')\n\n")
                f.write(f"    return lstm_model, history\n\n")
                f.write(f"if __name__ == '__main__':\n")
                f.write(f"    model, history = main()\n")
            
            print(f"\n💾 Script d'entraînement sauvegardé: {export_filename}")
            
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde du script: {e}")


def main():
    """
    Fonction principale pour lancer le configurateur LSTM.
    """
    try:
        configurator = LSTMConfigurator()
        configurator.run_interactive_setup()
    except KeyboardInterrupt:
        print("\n\n👋 Programme interrompu. Au revoir!")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")


if __name__ == "__main__":
    main()