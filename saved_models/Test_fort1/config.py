#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration du modèle LSTM
Générée le: 2025-11-20T07:13:06.753834
"""

# Configuration du modèle
MODEL_CONFIG = {
    "layers": [
        {
            "units": 400,
            "return_sequences": True,
            "dropout": 0.2,
            "sequence_length": 240,
            "bidirectional": False,
            "batch_normalization": True,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "kernel_regularizer": None,
            "recurrent_regularizer": None,
            "dropout_type": "both",
            "use_bias": True,
            "unit_forget_bias": True
        },
        {
            "units": 200,
            "return_sequences": True,
            "dropout": 0.2,
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
            "units": 50,
            "return_sequences": True,
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
        },
        {
            "units": 50,
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
    "dense_layers": [
        16,
        8
    ],
    "dense_units": 60,
    "learning_rate": 0.0005,
    "sequence_length": 240,
    "nombre_de_colonnes": 4,
    "loss_function": "directional_mse",
    "optimizer": "adam",
    "optimizer_config": {
        "beta_1": 0.9,
        "beta_2": 0.999,
        "epsilon": 1e-07,
        "amsgrad": False
    },
    "metrics": [
        "mae",
        "mse"
    ],
    "mixed_precision": False,
    "gradient_clipping": {
        "enabled": False,
        "max_norm": 1.0,
        "norm_type": 2
    },
    "target_columns": [
        "Open",
        "High",
        "Low",
        "Close"
    ]
}
