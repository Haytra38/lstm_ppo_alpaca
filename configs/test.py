#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration LSTM générée automatiquement
Générée le: 2025-11-10T10:11:12.290949
"""

# Configuration LSTM
LSTM_CONFIG = {
    "model_config": {
        "layers": [
            {
                "units": 128,
                "return_sequences": True,
                "dropout": 0.2,
                "sequence_length": 240,
                "bidirectional": True,
                "batch_normalization": True
            },
            {
                "units": 64,
                "return_sequences": True,
                "dropout": 0.3,
                "sequence_length": 240,
                "bidirectional": False,
                "batch_normalization": True
            },
            {
                "units": 32,
                "return_sequences": False,
                "dropout": 0.2,
                "sequence_length": 240,
                "bidirectional": False,
                "batch_normalization": False
            }
        ],
        "dense_layers": [
            16,
            8
        ],
        "dense_units": 1,
        "learning_rate": 0.0005,
        "sequence_length": 240,
        "nombre_de_colonnes": 4,
        "loss_function": "directional_mse"
    },
    "training_config": {
        "epochs": 200,
        "batch_size": 64,
        "target_columns": [
            "Open",
            "High",
            "Low",
            "Close"
        ],
        "validation_split": 0.2,
        "learning_rate": 0.0005
    },
    "data_config": {
        "selected_file": "BTC-USD1Min_2024-12-01_2025-08-31.csv",
        "file_path": "data/BTC-USD1Min_2024-12-01_2025-08-31.csv",
        "columns_info": {
            "all_columns": [
                "Time",
                "Open",
                "High",
                "Low",
                "Close"
            ],
            "numeric_columns": [
                "Open",
                "High",
                "Low",
                "Close"
            ]
        }
    },
    "metadata": {
        "created_date": "2025-11-10T10:10:53.327313",
        "description": "Configuration optimisée pour prédiction 1-minute avec LSTM bidirectionnel",
        "version": "2.0",
        "optimizations": [
            "Architecture bidirectionnelle",
            "BatchNormalization",
            "Fonction de perte directionnelle",
            "Hyperparamètres optimisés",
            "Sequence length étendue (4h)"
        ],
        "saved_date": "2025-11-10T10:11:12.191658",
        "config_name": "test1.py"
    }
}
