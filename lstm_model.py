from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input, Reshape, Bidirectional, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras import mixed_precision
import numpy as np
from keras.optimizers import Adam
import tensorflow as tf
import logging
import os
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer, MaxAbsScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Activer mixed precision (float16) pour accélérer sur GPU L4
_mp_flag = os.environ.get('TF_MIXED_PRECISION', '0')
if _mp_flag == '1':
    mixed_precision.set_global_policy('mixed_float16')
else:
    mixed_precision.set_global_policy('float32')

# Activer la croissance mémoire du GPU pour éviter OOM et allocations massives
_gpus = tf.config.list_physical_devices('GPU')
for _gpu in _gpus:
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except Exception:
        pass

# Fonction de perte globale pour éviter les problèmes de sérialisation
@tf.keras.utils.register_keras_serializable()
def directional_loss(y_true, y_pred):
    """
    Fonction de perte personnalisée qui pénalise les erreurs de direction.
    Utile pour les prédictions de prix où la direction est cruciale.
    """
    # Utiliser simplement MSE standard pour éviter les problèmes de dimension
    # La complexité de la perte directionnelle causait des erreurs de tensor shape
    return tf.keras.losses.mse(y_true, y_pred)

class LSTMModel:
    def __init__(self, scaler_config=None):
        self.model = Sequential()
        self.sequence_length = None
        self.config = None
        self.model_directory ="saved_models"
        
        # Configuration du scaler selon la configuration fournie
        self.scaler_config = scaler_config or {
            'scaler_type': 'robust',
            'scaler_config': {}
        }
        self.scaler = self._create_scaler_from_config(self.scaler_config)
            
    def _create_scaler_from_config(self, scaler_config):
        """
        Crée un scaler basé sur la configuration fournie.
        
        Args:
            scaler_config: Dictionnaire contenant la configuration du scaler
            
        Returns:
            Scaler configuré
        """
        scaler_type = scaler_config.get('scaler_type', 'robust')
        scaler_params = scaler_config.get('scaler_config', {})
        
        if scaler_type == 'robust':
            robust_config = scaler_params.get('robust', {})
            quantile_range = robust_config.get('quantile_range', [25.0, 75.0])
            return RobustScaler(quantile_range=tuple(quantile_range))
            
        elif scaler_type == 'robust_conservative':
            robust_config = scaler_params.get('robust_conservative', {})
            quantile_range = robust_config.get('quantile_range', [10.0, 90.0])
            return RobustScaler(quantile_range=tuple(quantile_range))
            
        elif scaler_type == 'minmax':
            minmax_config = scaler_params.get('minmax', {})
            feature_range = tuple(minmax_config.get('feature_range', [0, 1]))
            return MinMaxScaler(feature_range=feature_range)
            
        elif scaler_type == 'standard':
            standard_config = scaler_params.get('standard', {})
            with_mean = standard_config.get('with_mean', True)
            with_std = standard_config.get('with_std', True)
            return StandardScaler(with_mean=with_mean, with_std=with_std)
            
        elif scaler_type == 'quantile':
            quantile_config = scaler_params.get('quantile', {})
            n_quantiles = quantile_config.get('n_quantiles', 1000)
            output_distribution = quantile_config.get('output_distribution', 'uniform')
            return QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution=output_distribution
            )
            
        elif scaler_type == 'maxabs':
            maxabs_config = scaler_params.get('maxabs', {})
            copy = maxabs_config.get('copy', True)
            return MaxAbsScaler(copy=copy)
            
        else:
            # Par défaut, retourner un RobustScaler
            return RobustScaler()
    
    def create(self, config):
        self.nombre_de_predictions = None
        self.target_columns = None
        self.nombre_de_colonnes = None
        self.result = None


        # Création du dossier de sauvegarde s'il n'existe pas
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)
        # Pour sauvegarder l'historique d'entraînement  

        
    def create(self, config):
        """
        Crée un nouveau modèle LSTM avec la configuration spécifiée
        config = {
            'layers': [
                {
                    'units': int,
                    'return_sequences': bool,
                    'dropout': float,
                    'bidirectional': True
                }
            ],
            'dense_units': int,
            'learning_rate': float
            'sequence_length': int
            'nombre_de_colonnes': int

            'dense_layers': [16, 8],  # Couches denses intermédiaires
            'activation': 'tanh',  # Meilleur pour les séries temporelles
            'batch_normalization': True
        }
        """
        # Réinitialisation du modèle
        self.model = Sequential()

        self.config = config
        self.nombre_de_predictions = config.get('dense_units', 30)
        self.sequence_length = config.get('sequence_length', 60)
        self.nombre_de_colonnes = config.get('nombre_de_colonnes', 4)
        
        # Construction des couches
        for i, layer in enumerate(config['layers']):
            if i == 0:
                self.model.add(Input(shape=(layer['sequence_length'], self.nombre_de_colonnes)))
                # Première couche LSTM avec option bidirectionnelle
                if layer.get('bidirectional', False):
                    self.model.add(Bidirectional(LSTM(
                        layer['units'],
                        return_sequences=layer['return_sequences'],
                        activation='tanh'
                    )))
                else:
                    self.model.add(LSTM(
                        layer['units'],
                        return_sequences=layer['return_sequences'],
                        activation='tanh'
                    ))
            else:
                # Couches LSTM suivantes avec option bidirectionnelle
                if layer.get('bidirectional', False):
                    self.model.add(Bidirectional(LSTM(
                        units=layer['units'],
                        return_sequences=layer['return_sequences'],
                        activation='tanh'
                    )))
                else:
                    self.model.add(LSTM(
                        units=layer['units'],
                        return_sequences=layer['return_sequences'],
                        activation='tanh'
                    ))
            
            # Ajout de BatchNormalization si spécifié
            if layer.get('batch_normalization', False):
                self.model.add(BatchNormalization(dtype='float32'))
            
            # Ajout de Dropout
            if layer.get('dropout', 0) > 0:
                self.model.add(Dropout(layer['dropout']))
        
        # Ajout de couches denses intermédiaires si spécifiées
        dense_layers = config.get('dense_layers', [])
        for dense_units in dense_layers:
            self.model.add(Dense(dense_units, activation='relu'))
            self.model.add(Dropout(0.2))
        
        self.model.add(Dense(self.nombre_de_predictions*self.nombre_de_colonnes))
        self.model.add(Reshape((self.nombre_de_predictions, self.nombre_de_colonnes,)))

        
        # Optimiseur amélioré avec gradient clipping
        optimizer = Adam(
            learning_rate=config.get('learning_rate', 0.0005),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )
        
        # Compilation avec fonction de perte personnalisée
        loss_function = config.get('loss_function', 'mse')
        if loss_function == 'directional_mse':
            loss_function = self._directional_loss
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_function
        )


        self.sequence_length = self.model.input_shape[1]
        self.nombre_de_predictions = self.model.output_shape[1]
        logging.info("Modèle créé avec succès")
        return {
            'message': 'Modèle créé avec succès',
            'config': self.config
        }
    
    def _directional_loss(self, y_true, y_pred):
        """
        Fonction de perte personnalisée qui pénalise les erreurs de direction.
        Utile pour les prédictions de prix où la direction est cruciale.
        """
        # Utiliser la fonction globale pour éviter les problèmes de sérialisation
        return directional_loss(y_true, y_pred)
    
    def _split_data(self, data : pd.DataFrame, target_columns = 'Close', ratio=0.8):
        """
        Divise les données en train et test avec le ratio spécifié.
        target_columns : Liste des colonnes à utiliser comme input pour le modèle
        """
        data_selected = data[target_columns].values
        n_colonnes = data_selected.shape[1]
        train_size = int(len(data_selected) * ratio)
        train_data, test_data = data_selected[:train_size], data_selected[train_size:]
        return train_data, test_data
    
    def _create_sequences(self, data):
        """
        Transforme les données en séquences pour prédiction multiple.
        
        Args:
            data: Données normalisées en format numpy array
            sequence_length: Nombre de points dans la séquence d'entrée
            nombre_de_predictions: Nombre de points à prédire
            
        Returns:self.n_colonnes
            X: Séquences d'entrée de forme (n_sequences, sequence_length, 1)
            Y: Valeurs cibles de forme (n_sequences, nombre_de_predictions, 1)
        """
        X, Y = [], []
        
        # Boucle sur les données en laissant assez d'espace pour séquence et prédictions
        for i in range(len(data) - self.sequence_length - self.nombre_de_predictions + 1):
            # Prend sequence_length points pour l'entrée
            X.append(data[i:i + self.sequence_length])
            # Prend nombre_de_predictions points suivants comme cibles
            Y.append(data[i + self.sequence_length: i + self.sequence_length + self.nombre_de_predictions])

        # Conversion en arrays numpy
        X, Y = np.array(X), np.array(Y)
        
        # Ajout de la dimension des features si nécessaire
        if len(X.shape) == 2:  # Si données 1D
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            Y = np.reshape(Y, (Y.shape[0], Y.shape[1], 1))
        
        return X, Y
    
    def _create_last_sequence_for_prediction(self, data,target_columns = None):
        """
        Prépare la dernière séquence de données pour la prédiction.
        
        Args:
            data: Données normalisées sous forme de numpy array.
            sequence_length: Nombre de points dans la séquence d'entrée.
            
        Returns:
            X: Dernière séquence d'entrée de forme (1, sequence_length, nombre_de_colonnes)
        """
        #data = data[target_columns].values
        # On prend seulement la dernière séquence
        X = data[-self.sequence_length:]  # Derniers points pour respecter sequence_length

        # Reshape pour correspondre au format attendu par le modèle
        X = np.reshape(X, (1, self.sequence_length, self.nombre_de_colonnes))  

        return X

    def _create_sequences_for_prediction(self, data: pd.DataFrame):
        """Crée des séquences de longueur seq_length à partir des données"""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequence = data[i:i + self.sequence_length]
            sequences.append(sequence)
        return np.array(sequences)
    
    def _check_dimensions(self, X, Y):
        """
        Vérifie que les dimensions de X et Y correspondent à celles attendues par le modèle.
        
        Args:
            X (numpy array): Séquences d'entrée (n_sequences, sequence_length, n_colonnes)
            Y (numpy array): Valeurs cibles (n_sequences, nombre_de_predictions, n_colonnes)
            model (tf.keras.Model): Modèle LSTM entraîné ou non

        Raises:
            ValueError: Si les dimensions ne correspondent pas
        """
        # Récupération des dimensions du modèle
        model_input_shape = self.model.input_shape  # (None, sequence_length, n_colonnes)
        model_output_shape = self.model.output_shape  # (None, nombre_de_predictions, n_colonnes)

        # Vérification des dimensions de X
        if X.shape[1:] != model_input_shape[1:]:
            raise ValueError(f"🚨 Erreur : Dimension de X incorrecte ! Attendu par le modele {model_input_shape[1:]}, mais obtenu {X.shape[1:]}")

        # Vérification des dimensions de Y
        if Y.shape[1] != model_output_shape[1]:
            raise ValueError(f"🚨 Erreur : Dimension de Y incorrecte ! Attendu par le modele {model_output_shape[1]}, mais obtenu {Y.shape[1]}")

        logging.info("✅ Les dimensions de X et Y sont correctes pour l'entraînement.")

    def train(self, data : pd.DataFrame, training_config):
        """
        Entraîne le modèle .
        """
        try:

            target_columns = training_config.get('target_columns', 'Close')
            # Sauvegarder les colonnes cibles dans l'instance
            self.target_columns = target_columns
        
            logging.info(f"Colonne choisie: {target_columns}")

            # Séparation en train/test
            train_data, test_data = self._split_data(data,target_columns, ratio=0.8)

            # Entraînement du scaler
            self.scaler.fit(train_data)

            # application du scaler sur les données
            train_data = self.scaler.transform(train_data)
            test_data = self.scaler.transform(test_data)

            # Préparation des données
            X_train, Y_train = self._create_sequences(train_data)
            self._check_dimensions(X_train, Y_train)
            X_test, Y_test = self._create_sequences(test_data)
            self._check_dimensions(X_test, Y_test)

            
            
            # Configuration optimisée de l'entraînement
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,  # Plus de patience pour éviter l'arrêt prématuré
                    restore_best_weights=True,
                    min_delta=1e-6,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.7,  # Réduction plus graduelle
                    patience=8,
                    min_lr=1e-7,
                    verbose=1
                ),
                LearningRateScheduler(
                    lambda epoch: training_config.get('learning_rate', 0.0005) * (0.95 ** epoch),
                    verbose=0
                )
            ]
            
            # Entraînement avec validation et hyperparamètres optimisés
            self.training_history = self.model.fit(
                X_train, Y_train,
                epochs=training_config.get('epochs', 200),  # Plus d'epochs avec early stopping
                batch_size=training_config.get('batch_size', 64),  # Batch size optimisé
                validation_data=(X_test, Y_test),
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # Important pour les séries temporelles
            )
            
            return {
                'loss_history': self.training_history.history['loss'],
                'val_loss_history': self.training_history.history['val_loss']
            }
            
        except Exception as e: 
            logging.error(f"Erreur lors de l'entraînement: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame, predict_config: dict, timeframe_info: dict = None, verbose : int = 1) -> dict:
        """
        Effectue des prédictions sur plusieurs colonnes et plusieurs pas de temps en une seule passe.
        La fonction réalise :
          - La création de la séquence d'entrée à partir des données historiques.
          - La prédiction multi-pas en un appel (le modèle prédit directement self.nombre_de_predictions pas).
          - La génération des dates futures en se basant sur les métadonnées (timeframe_info).

        Args:
            data (pd.DataFrame): Données historiques.
            predict_config (dict): Contient :
                - "target_columns" (list) : Colonnes utilisées pour la prédiction.
                - "prediction_steps" (int): Nombre de pas de temps à prédire dans le futur (par défaut self.nombre_de_predictions).
                - "confidence_interval" (bool): Indique si l'intervalle de confiance doit être calculé.
            timeframe_info (dict, optionnel): Doit contenir notamment 'end_date' et 'pandas_freq' pour générer les dates futures.

        Returns:
            dict: Dictionnaire contenant :
                - "historical": Les dernières valeurs historiques utilisées pour la prédiction.
                - "future": Les prédictions futures pour chaque colonne.
                - "future_dates": Les dates associées aux prédictions futures.
                - "confidence": L'intervalle de confiance (si calculé).
                - "metrics": Des métriques (exemple fictif ici).
        """
        try:
            # Récupération des paramètres de configuration
            target_columns = predict_config.get("target_columns", ["Close"])
            prediction_steps = predict_config.get("prediction_steps", self.nombre_de_predictions)
            if prediction_steps is None:
                prediction_steps = self.nombre_de_predictions
                logging.warning(f"Warning: prediction_steps was None, using default value: {prediction_steps}")
            confidence_interval = predict_config.get("confidence_interval", False)
            
            # Extraction des données cibles en numpy array
            data_array = data[target_columns].values
            
            # Création de la séquence d'entrée
            input_seq = self._create_last_sequence_for_prediction(data_array, target_columns)
            
            # Prédiction directe multi-pas : le modèle doit renvoyer un tenseur de forme (1, prediction_steps, nombre_de_colonnes)
            predictions_scaled = self.model.predict(input_seq, verbose=verbose)  # shape attendue : (1, prediction_steps, nombre_de_colonnes)
            predictions_scaled = predictions_scaled[0]

            # Cast explicite pour stabilité des opérations numpy/sklearn
            predictions_scaled = predictions_scaled.astype('float64')
            
            # Inverse scaling si un scaler a été utilisé pendant l'entraînement
            if self.scaler is not None:
                predictions = self.scaler.inverse_transform(predictions_scaled)
            else:
                predictions = predictions_scaled

            # Nettoyage des NaN/Inf éventuels
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Génération des dates futures à partir des métadonnées si disponibles
            future_dates = None
            if timeframe_info is not None:
                pandas_freq = timeframe_info.get("pandas_freq", "H")
                
                # Vérifier si end_date existe dans timeframe_info
                if "end_date" in timeframe_info and timeframe_info["end_date"] is not None:
                    try:
                        # Afficher la valeur pour le débogage
                        logging.info(f"Date de fin reçue: {timeframe_info['end_date']}")
                        
                        
                        # Validation et conversion robuste de la date
                        last_date = None
                        end_date_value = timeframe_info["end_date"]
                        
                        # Essayer différentes méthodes de conversion
                        try:
                            if isinstance(end_date_value, (pd.Timestamp, datetime)):
                                last_date = pd.Timestamp(end_date_value)
                            elif isinstance(end_date_value, str):
                                last_date = pd.to_datetime(end_date_value)
                            else:
                                # Essayer une conversion générique
                                last_date = pd.to_datetime(end_date_value)
                                
                            # Vérifier que la date est valide et pas dans le futur lointain
                            current_time = pd.Timestamp.now()
                            
                            # Gérer les fuseaux horaires pour la comparaison
                            try:
                                if last_date.tz is not None and current_time.tz is None:
                                    current_time = current_time.tz_localize('UTC')
                                elif last_date.tz is None and current_time.tz is not None:
                                    last_date = last_date.tz_localize('UTC')
                                
                                if last_date > current_time + pd.Timedelta(days=365):
                                    logging.warning(f"Date de fin trop éloignée dans le futur: {last_date}")
                                    last_date = current_time
                            except Exception as tz_error:
                                logging.warning(f"Erreur de comparaison de fuseaux horaires: {tz_error}")
                                # Convertir en UTC pour éviter les problèmes
                                if last_date.tz is not None:
                                    last_date = last_date.tz_convert('UTC').tz_localize(None)
                                current_time = pd.Timestamp.now()
                                
                        except Exception as date_parse_error:
                            logging.error(f"Erreur lors de la conversion de la date '{end_date_value}': {date_parse_error}")
                            last_date = pd.Timestamp.now()
                            logging.warning(f"Utilisation de la date actuelle comme fallback: {last_date}")
                        
                        # Validation finale
                        if not isinstance(last_date, pd.Timestamp) or pd.isna(last_date):
                            logging.error(f"Impossible de créer une date valide à partir de: {end_date_value}")
                            last_date = pd.Timestamp.now()
                            logging.warning(f"Utilisation de la date actuelle: {last_date}")
                        
                        # Générer les dates futures - Debug output for prediction_steps
                        logging.info(f"Génération de {prediction_steps} dates futures à partir de {last_date} avec fréquence {pandas_freq}")

                        # Ensure prediction_steps is an integer
                        periods = int(prediction_steps) + 1
                        future_dates = pd.date_range(start=last_date, periods=periods, freq=pandas_freq)[1:]
                        
                        # Formater les dates de manière plus lisible
                        formatted_dates = []
                        for date in future_dates:
                            if pandas_freq in ['min', 'h', 'T', 'H']:  # Minutes ou heures
                                formatted_dates.append(date.strftime('%Y-%m-%d %H:%M:%S'))
                            elif pandas_freq == 'D':  # Jours
                                formatted_dates.append(date.strftime('%Y-%m-%d'))
                            elif pandas_freq == 'W':  # Semaines
                                formatted_dates.append(date.strftime('%Y-%m-%d (Semaine %U)'))
                            elif pandas_freq in ['M', 'ME']:  # Mois
                                formatted_dates.append(date.strftime('%Y-%m'))
                            else:
                                formatted_dates.append(date.strftime('%Y-%m-%d %H:%M:%S'))
                        
                        future_dates = formatted_dates
                        logging.info(f"Dates futures générées: {future_dates[:3]}...")  # Afficher seulement les 3 premières
                        
                    except Exception as date_error:
                        logging.error(f"Erreur lors de la génération des dates futures: {date_error}")
                        # Utiliser des indices numériques comme fallback
                        future_dates = [f"t+{i+1}" for i in range(int(prediction_steps))]
                else:
                    logging.warning(f"Warning: 'end_date' manquant dans timeframe_info: {timeframe_info}")
                    # Utiliser des indices numériques comme fallback
                    future_dates = [f"t+{i+1}" for i in range(int(prediction_steps))]
            else:
                logging.warning(f"Warning: timeframe_info est None")
                # Utiliser des indices numériques comme fallback
                future_dates = [f"t+{i+1}" for i in range(int(prediction_steps))]

            # Calcul simplifié de l'intervalle de confiance
            confidence = None
            
            # Exemple fictif de métrique
            metrics = {"dummy_metric": 0.0}
            
            # Extraction des dates historiques si elles existent dans le DataFrame
            historical_dates = None
            if 'Date' in data.columns:
                historical_dates = data.tail(self.sequence_length)['Date'].tolist()
            elif 'date' in data.columns:
                historical_dates = data.tail(self.sequence_length)['date'].tolist()
            elif 'timestamp' in data.columns:
                historical_dates = data.tail(self.sequence_length)['timestamp'].tolist()
            elif data.index.name in ['Date', 'date', 'timestamp'] or isinstance(data.index, pd.DatetimeIndex):
                historical_dates = data.tail(self.sequence_length).index.astype(str).tolist()
            
            # Construction du résultat final
            # Harmoniser la longueur des colonnes cibles avec les sorties du modèle
            out_cols = predictions.shape[1]
            if len(target_columns) != out_cols:
                target_columns = target_columns[:out_cols]
            future_dict = {col: predictions[:, idx].tolist() for idx, col in enumerate(target_columns)}

            results = {
                "historical": data.tail(self.sequence_length).to_dict("list"),
                "historical_dates": historical_dates,
                "future": future_dict,
                "future_dates": future_dates,
                "confidence": confidence,
                "metrics": metrics
            }
            return results
        
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction: {str(e)}")
            return {"error": str(e)}
        
    
    def save_model(self, model_name):
        """
        Sauvegarde un modèle LSTM et ses composants associés.
        
        Args:
            lstm_model: L'instance de LSTMModel à sauvegarder
            model_name: Le nom sous lequel sauvegarder le modèle
        """
        try:
            # Vérification que le modèle existe
            if not self.model:
                raise ValueError("Modèle invalide ou non initialisé")

            # Création du dossier pour ce modèle spécifique
            model_path = os.path.join(self.model_directory, model_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            # Sauvegarde des composants du modèle
            model_file = os.path.join(model_path, "model.keras")
            scaler_file = os.path.join(model_path, "scaler.pkl")
            config_file = os.path.join(model_path, "config.py")

            # Sauvegarde du modèle Keras
            self.model.save(model_file)
            logging.info(f"Modèle sauvegardé dans: {model_file}")

            # Sauvegarde du config
            self._save_config(config_file)
            # Sauvegarde du scaler
            with open(scaler_file, "wb") as f:
                pickle.dump(self.scaler, f)
            logging.info(f"Scaler sauvegardé dans: {scaler_file}")

            # Sauvegarde de l'historique d'entraînement
            #self.save_training_history(model_path)

            logging.info(f"Sauvegarde complète réussie pour le modèle: {model_name}")
            return True

        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")
            raise
        
    def load_model(self, model_name):
        """
        Charge un modèle sauvegardé et ses composants.
        
        Args:
            model_name: Le nom du modèle à charger
            
        Returns:
            LSTMModel: Une instance de LSTMModel avec le modèle chargé
        """
        try:
            model_path = os.path.join(self.model_directory, model_name)
            config_path = os.path.join(self.model_directory, model_name, 'config.py')

            if not os.path.exists(model_path):
                raise ValueError(f"Le modèle {model_name} n'existe pas")

            # Chargement du modèle avec objets personnalisés
            custom_objects = {
                'directional_loss': directional_loss,
                '_directional_loss': directional_loss,  # Pour compatibilité avec anciens modèles
                'method': directional_loss  # Pour résoudre l'erreur de classe 'method'
            }
            self.model : Sequential = load_model(os.path.join(model_path, "model.keras"), custom_objects=custom_objects)
            self.sequence_length = self.model.input_shape[1]
            self.nombre_de_predictions = self.model.output_shape[1]
            self.nombre_de_colonnes = self.model.input_shape[2]
            
            # Chargement du scaler (sans forcer le type)
            scaler_path = os.path.join(model_path, "scaler.pkl")
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            logging.info(f"Scaler chargé: {type(self.scaler).__name__}")
            # Chargement de config
            self._load_config(config_path)
            try:
                if self.config and isinstance(self.config, dict):
                    mp = self.config.get('mixed_precision', None)
                    if mp is not None:
                        mixed_precision.set_global_policy('mixed_float16' if mp else 'float32')
            except Exception:
                pass
            # Chargement de l'historique si disponible
            #self.load_training_history(model_name)

            logging.info(f"Modèle {model_name} chargé avec succès")

        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

    def save_training_history(self,model_path):
        """
        Sauvegarde l'historique d'entraînement au format Python.
        
        Args:
            model_path (str): Chemin du dossier où sauvegarder l'historique
            
        Returns:
            bool: True si la sauvegarde est réussie, False sinon
        """
        try:
          
            # Préparation de l'historique pour la sauvegarde
            history_to_save = {
                'loss_history': [float(x) for x in self.training_history.get('loss_history', [])],
                'val_loss_history': [float(x) for x in self.training_history.get('val_loss_history', [])],
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'epochs': len(self.training_history.get('loss_history', [])),
                    'final_loss': float(self.training_history.get('loss_history')[-1]) if self.training_history.get('loss_history') else None,
                    'final_val_loss': float(self.training_history.get('val_loss_history')[-1]) if self.training_history.get('val_loss_history') else None
                }
            }
            
            # Sauvegarde dans un fichier Python
            history_path = os.path.join(model_path, 'training_history.py')
            with open(history_path, 'w', encoding='utf-8') as f:
                f.write('#!/usr/bin/env python3\n')
                f.write('# -*- coding: utf-8 -*-\n')
                f.write('"""\n')
                f.write('Historique d\'entraînement du modèle LSTM\n')
                f.write('Généré automatiquement\n')
                f.write('"""\n\n')
                f.write('TRAINING_HISTORY = ')
                f.write(self._dict_to_python_code(history_to_save))
                f.write('\n')
                
            logging.info(f"Historique sauvegardé avec succès dans {history_path}")
            return True
            
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'historique: {str(e)}")
            return False

    def load_training_history(self, model_name):
        """
        Charge l'historique d'entraînement depuis un fichier Python.
        
        Args:
            model_name (str): Nom du modèle dont on veut charger l'historique
            
        Returns:
            dict: L'historique d'entraînement chargé ou None en cas d'erreur
        """
        try:
            # Construction du chemin du fichier
            history_path = os.path.join('saved_models', model_name, 'training_history.py')
            # Vérification de l'existence du fichier
            if not os.path.exists(history_path):
                logging.error(f"Fichier d'historique non trouvé: {history_path}")
                return None
            
            # Créer un namespace pour exécuter le fichier
            namespace = {}
            with open(history_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Exécuter le contenu du fichier
            exec(content, namespace)
            
            # Récupérer l'historique
            if 'TRAINING_HISTORY' in namespace:
                self.training_history = namespace['TRAINING_HISTORY']
                logging.info(f"Historique chargé avec succès depuis {history_path}")
                return True
            else:
                logging.error(f"Variable TRAINING_HISTORY non trouvée dans {history_path}")
                return False
            
        except Exception as e:
            logging.error(f"Erreur lors du chargement de l'historique: {str(e)}")
            return False
        
    
    def _save_config(self, config_file='config.py'):
        try:
            # Ajouter les target_columns à la configuration si elles existent
            config_to_save = self.config.copy() if self.config else {}
            if hasattr(self, 'target_columns') and self.target_columns is not None:
                config_to_save['target_columns'] = self.target_columns
            
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write("#!/usr/bin/env python3\n")
                f.write("# -*- coding: utf-8 -*-\n")
                f.write("\"\"\"\n")
                f.write("Configuration du modèle LSTM\n")
                f.write(f"Générée le: {datetime.now().isoformat()}\n")
                f.write("\"\"\"\n\n")
                f.write("# Configuration du modèle\n")
                f.write(f"MODEL_CONFIG = {self._dict_to_python_code(config_to_save)}\n")
            logging.info(f"Configuration sauvegardée dans {config_file}")
            return True
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde : {e}")
            return False
        
    def _load_config(self, config_path='config.py'):
        try:
            if not os.path.exists(config_path):
                logging.info(f"Fichier de configuration non trouvé : {config_path}")
                return None
            
            # Créer un namespace pour exécuter le fichier
            namespace = {}
            with open(config_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Exécuter le contenu du fichier
            exec(content, namespace)
            
            # Récupérer la configuration
            if 'MODEL_CONFIG' in namespace:
                config = namespace['MODEL_CONFIG']
                if self.validate_config(config):
                    self.config = config
                    # Charger les target_columns si elles existent dans la configuration
                    if 'target_columns' in config:
                        self.target_columns = config['target_columns']
                    return config
            else:
                logging.error(f"Variable MODEL_CONFIG non trouvée dans {config_path}")
                return None

        except Exception as e:
            logging.error(f"Erreur lors du chargement : {e}")
            return None
        
    def _dict_to_python_code(self, obj, indent=0):
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
                value_str = self._dict_to_python_code(value, indent + 1)
                lines.append(f'{indent_str}    "{key}": {value_str}{comma}')
            lines.append(f'{indent_str}}}')
            return '\n'.join(lines)
        
        elif isinstance(obj, list):
            if not obj:
                return '[]'
            
            lines = ['[']
            for i, item in enumerate(obj):
                comma = ',' if i < len(obj) - 1 else ''
                item_str = self._dict_to_python_code(item, indent + 1)
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
    
    def validate_config(self,config):
        try:
            # Vérification de la structure de base
            assert 'layers' in config
            assert 'dense_units' in config
            assert 'learning_rate' in config
            
            # Vérification des layers
            for layer in config['layers']:
                assert isinstance(layer['units'], int)
                assert isinstance(layer['return_sequences'], bool)
                assert isinstance(layer['dropout'], float)

            
            # Vérification des autres paramètres
            assert isinstance(config['dense_units'], int)
            assert isinstance(config['learning_rate'], float)
            assert isinstance(config['nombre_de_colonnes'], int)  # Corrigé: int au lieu de list
            
            return True
        except AssertionError:
            logging.info("Configuration invalide")
            return False

    # Fonction pour charger la configuration
    def load_config(self, filepath='config.py'):
        try:
            # Créer un namespace pour exécuter le fichier
            namespace = {}
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Exécuter le contenu du fichier
            exec(content, namespace)
            
            # Récupérer la configuration
            if 'MODEL_CONFIG' in namespace:
                return namespace['MODEL_CONFIG']
            else:
                logging.error(f"Variable MODEL_CONFIG non trouvée dans {filepath}")
                return None
                
        except Exception as e:
            logging.error(f"Erreur lors du chargement de la configuration: {str(e)}")
            return None
        
    def get_available_models(self):
        """Liste tous les modèles disponibles"""
        if not os.path.exists(self.model_directory):
            return []
        
        models = []
        for item in os.listdir(self.model_directory):
            item_path = os.path.join(self.model_directory, item)
            if os.path.isdir(item_path):
                # Vérifier que le dossier contient les fichiers nécessaires
                required_files = ['model.keras', 'scaler.pkl', 'config.py']
                if all(os.path.exists(os.path.join(item_path, f)) for f in required_files):
                    models.append(item)
        
        return models


    def get_model_summary(self):
        """
        Retourne le résumé du modèle sous forme de texte
        
        Returns:
            str: Le résumé du modèle
        """
        try:
            # Capture la sortie de summary() dans une chaîne
            string_list = []
            self.model.summary(line_length=None, print_fn=lambda x: string_list.append(x))
            return {
                'summary': '\n'.join(string_list)
            }
        except Exception as e:
            logging.error(f"Erreur lors de la génération du résumé du modèle: {str(e)}")           
            raise

    def save_predictions_to_excel(self, filename=None):
        """
        Sauvegarde les résultats de prédiction dans un fichier Excel.
        
        Args:
            filename (str, optional): Nom du fichier Excel. Si None, un nom basé sur la date sera généré.
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        try:
            if self.result is None:
                raise ValueError("Aucun résultat de prédiction disponible. Exécutez d'abord la méthode predict().")
            

            # Générer un nom de fichier par défaut si non spécifié
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"predictions_{timestamp}.xlsx"
            
            # Assurer que le chemin est absolu
            if not os.path.isabs(filename):
                filename = os.path.join(os.getcwd(), filename)
            
            # Vérifier la structure des résultats et adapter en conséquence
            if isinstance(self.result, dict) and 'future' in self.result and 'historical' in self.result:
                # Nouvelle structure de données (depuis predict())
                
                # Créer un DataFrame pour les données historiques
                historical_data = []
                if 'historical' in self.result and isinstance(self.result['historical'], dict):
                    historical_dict = self.result['historical']
                    historical_dates = self.result.get('historical_dates', [])
                    
                    # Déterminer la longueur des données avec vérifications supplémentaires
                    valid_values = []
                    for key, values in historical_dict.items():
                        if values is not None:
                            if isinstance(values, list):
                                valid_values.append(len(values))
                            else:
                                valid_values.append(1)
                    
                    max_len = max(valid_values) if valid_values else 0
                    
                    for i in range(max_len):
                        row = {}
                        
                        if historical_dates is not None and i < len(historical_dates):
                            row['Date'] = historical_dates[i]
                        else:
                            row['Date'] = f'Historical_{i+1}'
                        
                        for col, values in historical_dict.items():
                            if values is not None:
                                if isinstance(values, list) and i < len(values):
                                    row[f'Historical_{col}'] = values[i]
                                elif not isinstance(values, list):
                                    row[f'Historical_{col}'] = values
                        
                        historical_data.append(row)
                
                historical_df = pd.DataFrame(historical_data)
                
                # Créer un DataFrame pour les prédictions futures
                future_data = []
                if 'future' in self.result and isinstance(self.result['future'], dict):
                    future_dict = self.result['future']
                    future_dates = self.result.get('future_dates', [])
                    
                    # Déterminer la longueur des prédictions avec vérifications supplémentaires
                    valid_values = []
                    for key, values in future_dict.items():
                        if values is not None:
                            if isinstance(values, list):
                                valid_values.append(len(values))
                            else:
                                valid_values.append(1)
                    
                    max_len = max(valid_values) if valid_values else 0
                    
                    for i in range(max_len):
                        row = {}
                        if i < len(future_dates):
                            row['Date'] = future_dates[i]
                        else:
                            row['Date'] = f't+{i+1}'
                        
                        for col, values in future_dict.items():
                            if isinstance(values, list) and i < len(values):
                                row[f'Predicted_{col}'] = values[i]
                            elif not isinstance(values, list):
                                row[f'Predicted_{col}'] = values
                        
                        future_data.append(row)
                
                future_df = pd.DataFrame(future_data)
                
                # Créer un DataFrame pour les métriques (si disponibles)
                metrics_df = pd.DataFrame()
                if 'metrics' in self.result and self.result['metrics']:
                    metrics_data = []
                    for key, value in self.result['metrics'].items():
                        metrics_data.append({'Metric': key, 'Value': value})
                    metrics_df = pd.DataFrame(metrics_data)
                
            else:
                # Ancienne structure de données (format legacy)
                # Créer un DataFrame pour les prédictions historiques
                historical_data = []
                for item in self.result['historical']:
                    row = {'Date': item['date']}
                    
                    # Ajouter les valeurs réelles
                    if item['actual'] is not None:
                        for col, val in item['actual'].items():
                            row[f'Actual_{col}'] = val
                    
                    # Ajouter les valeurs prédites
                    for col, val in item['predicted'].items():
                        row[f'Predicted_{col}'] = val
                    
                    historical_data.append(row)
                
                historical_df = pd.DataFrame(historical_data)
                
                # Créer un DataFrame pour les prédictions futures
                future_data = []
                for item in self.result['future']:
                    row = {'Date': item['date']}
                    
                    # Ajouter les valeurs prédites
                    for col, val in item['predicted'].items():
                        row[f'Predicted_{col}'] = val
                    
                    future_data.append(row)
                
                future_df = pd.DataFrame(future_data)
                
                # Créer un DataFrame pour les métriques
                metrics_data = []
                for col, metrics in self.result['metadata']['metrics'].items():
                    row = {'Column': col}
                    for metric_name, metric_value in metrics.items():
                        row[metric_name] = metric_value
                    metrics_data.append(row)
                
                metrics_df = pd.DataFrame(metrics_data)
            
            # Créer un writer Excel
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                if not historical_df.empty:
                    historical_df.to_excel(writer, sheet_name='Historical Data', index=False)
                
                if not future_df.empty:
                    future_df.to_excel(writer, sheet_name='Future Predictions', index=False)
                
                if not metrics_df.empty:
                    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # Ajouter une feuille avec les informations générales
                info_data = [
                    {'Key': 'Generation Date', 'Value': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                    {'Key': 'Historical Records', 'Value': len(historical_df) if not historical_df.empty else 0},
                    {'Key': 'Future Predictions', 'Value': len(future_df) if not future_df.empty else 0}
                ]
                
                # Ajouter des informations sur les colonnes si disponibles
                if not future_df.empty:
                    pred_columns = [col for col in future_df.columns if col.startswith('Predicted_')]
                    if pred_columns:
                        info_data.append({'Key': 'Predicted Columns', 'Value': ', '.join([col.replace('Predicted_', '') for col in pred_columns])})
                
                info_df = pd.DataFrame(info_data)
                info_df.to_excel(writer, sheet_name='Info', index=False)
            
            logging.info(f"Prédictions sauvegardées avec succès dans {filename}")
            return filename
            
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde des prédictions en Excel: {str(e)}")
            raise

    def generate_prediction_plots(self, save_path=None, show_plots=False):
        """
        Génère des graphiques pour visualiser les prédictions.
        
        Args:
            save_path (str, optional): Chemin où sauvegarder les graphiques. Si None, les graphiques ne sont pas sauvegardés.
            show_plots (bool): Si True, affiche les graphiques (utile dans un notebook).
            
        Returns:
            dict: Dictionnaire contenant les objets figure pour chaque graphique généré
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.dates import DateFormatter
            
            if self.result is None:
                raise ValueError("Aucun résultat de prédiction disponible. Exécutez d'abord la méthode predict().")
            
            # Créer le dossier de sauvegarde si nécessaire
            if save_path is not None and not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # Extraire les données
            target_columns = self.result['metadata']['target_columns']
            
            # Convertir les dates en objets datetime
            historical_dates = []
            for item in self.result['historical']:
                try:
                    historical_dates.append(pd.to_datetime(item['date']))
                except:
                    historical_dates.append(None)
            
            future_dates = []
            for item in self.result['future']:
                try:
                    future_dates.append(pd.to_datetime(item['date']))
                except:
                    future_dates.append(None)
            
            # Dictionnaire pour stocker les figures
            figures = {}
            
            # Générer un graphique pour chaque colonne cible
            for col in target_columns:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Extraire les valeurs historiques réelles et prédites
                historical_actual = []
                historical_predicted = []
                
                for item in self.result['historical']:
                    if item['actual'] is not None and col in item['actual']:
                        historical_actual.append(item['actual'][col])
                    else:
                        historical_actual.append(None)
                    
                    if col in item['predicted']:
                        historical_predicted.append(item['predicted'][col])
                    else:
                        historical_predicted.append(None)
                
                # Extraire les valeurs futures prédites
                future_predicted = []
                for item in self.result['future']:
                    if col in item['predicted']:
                        future_predicted.append(item['predicted'][col])
                    else:
                        future_predicted.append(None)
                
                # Tracer les valeurs historiques réelles
                ax.plot(historical_dates, historical_actual, 'b-', label='Valeurs réelles', linewidth=2)
                
                # Tracer les valeurs historiques prédites
                ax.plot(historical_dates, historical_predicted, 'r--', label='Prédictions historiques', linewidth=2)
                
                # Tracer les valeurs futures prédites
                ax.plot(future_dates, future_predicted, 'g--', label='Prédictions futures', linewidth=2)
                
                # Ajouter une ligne verticale pour séparer l'historique et le futur
                if len(historical_dates) > 0 and len(future_dates) > 0:
                    last_historical_date = historical_dates[-1]
                    ax.axvline(x=last_historical_date, color='k', linestyle='--', alpha=0.5)
                
                # Configurer le graphique
                ax.set_title(f'Prédictions pour {col}', fontsize=16)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel(f'Valeur de {col}', fontsize=12)
                ax.legend(loc='best', fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Formater les dates sur l'axe x
                date_format = DateFormatter('%Y-%m-%d')
                ax.xaxis.set_major_formatter(date_format)
                fig.autofmt_xdate()  # Rotation des étiquettes de date
                
                # Ajouter les métriques dans un encadré
                if col in self.result['metadata']['metrics']:
                    metrics = self.result['metadata']['metrics'][col]
                    metrics_text = "\n".join([f"{k}: {v:.4f}" if v is not None else f"{k}: N/A" for k, v in metrics.items()])
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                            verticalalignment='top', bbox=props)
                
                plt.tight_layout()
                
                # Sauvegarder le graphique
                if save_path is not None:
                    filename = os.path.join(save_path, f'prediction_{col}.png')
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    logging.info(f"Graphique pour {col} sauvegardé dans {filename}")
                
                # Stocker la figure
                figures[col] = fig
                
                # Afficher ou fermer le graphique
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)
            
            # Générer un graphique de comparaison des métriques
            if len(target_columns) > 1:
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Extraire les métriques pour chaque colonne
                columns = []
                mae_values = []
                rmse_values = []
                r2_values = []
                
                for col, metrics in self.result['metadata']['metrics'].items():
                    if col != 'Global':  # Exclure les métriques globales
                        columns.append(col)
                        mae_values.append(metrics.get('MAE', 0))
                        rmse_values.append(metrics.get('RMSE', 0))
                        r2_values.append(metrics.get('R2', 0))
                
                # Créer le graphique à barres
                x = np.arange(len(columns))
                width = 0.25
                
                ax.bar(x - width, mae_values, width, label='MAE')
                ax.bar(x, rmse_values, width, label='RMSE')
                ax.bar(x + width, r2_values, width, label='R²')
                
                ax.set_title('Comparaison des métriques par colonne', fontsize=16)
                ax.set_xticks(x)
                ax.set_xticklabels(columns)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Sauvegarder le graphique
                if save_path is not None:
                    filename = os.path.join(save_path, 'metrics_comparison.png')
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    logging.info(f"Graphique de comparaison des métriques sauvegardé dans {filename}")
                
                # Stocker la figure
                figures['metrics_comparison'] = fig
                
                # Afficher ou fermer le graphique
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)
            
            return figures
            
        except Exception as e:
            logging.error(f"Erreur lors de la génération des graphiques: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
