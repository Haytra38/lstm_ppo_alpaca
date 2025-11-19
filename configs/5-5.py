#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration LSTM générée automatiquement
Générée le: 2025-08-28T12:32:05.944264
"""

# Configuration LSTM
LSTM_CONFIG = {
    "model_config": {
        "layers": [
            {
                "units": 50,
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
        "dense_units": 5,
        "learning_rate": 0.001,
        "sequence_length": 60,
        "nombre_de_colonnes": 5
    },
    "training_config": {
        "epochs": 100,
        "batch_size": 32,
        "target_columns": [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ],
        "validation_split": 0.2
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
        "created_date": "2025-08-28T12:30:16.036866",
        "description": "Configuration par défaut",
        "version": "1.0",
        "saved_date": "2025-08-28T12:32:05.934094",
        "config_name": "5-5.py"
    }
}
