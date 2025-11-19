#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration LSTM générée automatiquement
Générée le: 2025-09-02T19:40:49.257518
"""

# Configuration LSTM
LSTM_CONFIG = {
    "model_config": {
        "layers": [
            {
                "units": 200,
                "return_sequences": True,
                "dropout": 0.2,
                "sequence_length": 60
            },
            {
                "units": 50,
                "return_sequences": False,
                "dropout": 0.2,
                "sequence_length": 60
            }
        ],
        "dense_layers": [
            16,
            8
        ],
        "dense_units": 10,
        "learning_rate": 0.0005,
        "sequence_length": 60,
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
        "selected_file": "BTC-USD1Min_2018-12-01_2025-08-31.csv",
        "file_path": "data\BTC-USD1Min_2018-12-01_2025-08-31.csv",
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
        "created_date": "2025-09-02T19:38:26.227572",
        "description": "Configuration optimisée pour prédiction 1-minute avec LSTM bidirectionnel",
        "version": "2.0",
        "optimizations": [
            "Architecture bidirectionnelle",
            "BatchNormalization",
            "Fonction de perte directionnelle",
            "Hyperparamètres optimisés",
            "Sequence length étendue (4h)"
        ],
        "saved_date": "2025-09-02T19:40:49.257518",
        "config_name": "123.py"
    }
}
