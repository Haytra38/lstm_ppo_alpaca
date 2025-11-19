#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration LSTM générée automatiquement
Générée le: 2025-09-02T14:55:36.137749
"""

# Configuration LSTM
LSTM_CONFIG = {
    "model_config": {
        "layers": [
            {
                "units": 256,
                "return_sequences": True,
                "dropout": 0.15,
                "sequence_length": 480,
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
        "dense_layers": [
            32,
            16,
            8
        ],
        "dense_units": 5,
        "learning_rate": 0.0003,
        "sequence_length": 480,
        "nombre_de_colonnes": 4,
        "loss_function": "directional_mse"
    },
    "training_config": {
        "epochs": 300,
        "batch_size": 128,
        "target_columns": [
            "Open",
            "High",
            "Low",
            "Close"
        ],
        "validation_split": 0.15,
        "learning_rate": 0.0003
    },
    "data_config": {
        "selected_file": "BTC-USD1Min_2025-01-01_2025-01-31.csv",
        "file_path": "data\BTC-USD1Min_2025-01-01_2025-01-31.csv",
        "columns_info": {
            "all_columns": [
                "Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ],
            "numeric_columns": [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        }
    },
    "metadata": {
        "created_date": "2025-09-02T14:55:09.053602",
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
        ],
        "saved_date": "2025-09-02T14:55:36.137749",
        "config_name": "optimised.py"
    }
}
