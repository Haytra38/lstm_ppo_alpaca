#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration LSTM générée automatiquement
Générée le: 2025-11-19T23:08:03.453719
"""

# Configuration LSTM
LSTM_CONFIG = {
    "model_config": {
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
        }
    },
    "training_config": {
        "epochs": 200,
        "batch_size": 400,
        "target_columns": [
            "Open",
            "High",
            "Low",
            "Close"
        ],
        "validation_split": 0.2,
        "learning_rate": 0.0005,
        "data_preprocessing": {
            "scaler_type": "robust",
            "scaler_config": {
                "robust": {
                    "quantile_range": [
                        25.0,
                        75.0
                    ]
                },
                "robust_conservative": {
                    "quantile_range": [
                        10.0,
                        90.0
                    ]
                },
                "minmax": {
                    "feature_range": [
                        0,
                        1
                    ]
                },
                "standard": {
                    "with_mean": True,
                    "with_std": True
                },
                "quantile": {
                    "n_quantiles": 1000,
                    "output_distribution": "uniform"
                },
                "maxabs": {
                    "copy": True
                }
            },
            "feature_scaling": {
                "enabled": True,
                "columns": None,
                "fit_on_training_only": True
            },
            "sequence_scaling": {
                "enabled": True,
                "method": "per_feature"
            },
            "handle_outliers": {
                "enabled": False,
                "method": "clip",
                "threshold": 3.0,
                "clip_range": [
                    -3.0,
                    3.0
                ]
            }
        },
        "early_stopping": {
            "enabled": True,
            "patience": 15,
            "min_delta": 0.0001,
            "restore_best_weights": True,
            "monitor": "val_loss"
        },
        "learning_rate_scheduler": {
            "enabled": True,
            "type": "reduce_on_plateau",
            "factor": 0.5,
            "patience": 10,
            "min_lr": 1e-06,
            "cooldown": 5
        },
        "callbacks": {
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
        "validation_strategy": {
            "type": "holdout",
            "k_folds": 5,
            "shuffle": False
        },
        "data_augmentation": {
            "enabled": False,
            "noise_level": 0.01,
            "time_warping": False,
            "scaling": False
        }
    },
    "data_config": {
        "selected_file": "BTC-USD1Min_2021-01-01_2025-11-18.csv",
        "file_path": "data/BTC-USD1Min_2021-01-01_2025-11-18.csv",
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
        "created_date": "2025-11-19T23:04:29.409083",
        "description": "Configuration optimisée pour prédiction 1-minute avec LSTM bidirectionnel",
        "version": "2.0",
        "optimizations": [
            "Architecture bidirectionnelle",
            "BatchNormalization",
            "Fonction de perte directionnelle",
            "Hyperparamètres optimisés",
            "Sequence length étendue (4h)"
        ],
        "saved_date": "2025-11-19T23:08:03.452788",
        "config_name": "Test_fort1.py"
    }
}
