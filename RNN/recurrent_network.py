from RNN.utilities import dataframe_to_reframe, split_reframed
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense
from tensorflow.keras import regularizers
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import tensorflow as tf

class RecurrentNetwork:
    def __init__(self, rnn_type, units=100, epochs=100, batch_size=16, dropout=0.2, activation='relu', n_days=7):
        """
        Inizializza la rete neurale ricorrente con i parametri specificati.

        Args:
        - rnn_type: Classe della rete ricorrente (LSTM o GRU).
        - units: Numero di unità nel layer ricorrente.
        - epochs: Numero di epoche di addestramento.
        - batch_size: Dimensione del batch per l'addestramento.
        - dropout: Percentuale di dropout per prevenire overfitting.
        - activation: Funzione di attivazione per il layer ricorrente.
        - n_days: Numero di giorni di input da considerare per la previsione.
        """
        self.layer_class = rnn_type  # Classe del layer ricorrente (LSTM o GRU)
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.activation = activation
        self.n_days = n_days
        self.initialized = False  # Flag per verificare se il modello è stato inizializzato

    def upload(self, dataframe, n_test=14):
        """
        Carica e prepara i dati per l'addestramento e il test.

        Args:
        - dataframe: Il DataFrame con i dati.
        - n_test: Numero di campioni utilizzati per il test.
        """
        self.df = dataframe
        self.n_train = int(len(dataframe) - n_test)  # Calcola la dimensione del set di addestramento
        # Pre-processa i dati e li trasforma in un formato supervisionato
        self.scaler, self.n_features, self.reframed = dataframe_to_reframe(self.df, self.n_days)
        # Divide i dati in set di addestramento e test
        self.train_X, self.train_y, self.test_X, self.test_y = split_reframed(self.reframed, self.n_train, self.n_days, self.n_features)

    def get_parameters(self):
        """
        Ritorna i parametri del modello in un dizionario.
        """
        params = {'units': self.units, 'epochs': self.epochs, 'batch_size': self.batch_size, 'dropout': self.dropout}
        return params

    def update_parameters(self, params):
        """
        Aggiorna i parametri del modello.

        Args:
        - params: Dizionario con i parametri aggiornati.
        """
        self.units = params['units']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.dropout = params['dropout']

    def initialize(self, cudnn=False):
        """
        Inizializza il modello con i layer appropriati.

        Args:
        - cudnn: Booleano che indica se usare CuDNN per ottimizzazioni specifiche (utile per GPU).
        """
        self.model = Sequential()
        # Se cudnn è True, utilizza ottimizzazioni specifiche per LSTM su GPU
        if cudnn:
            self.model.add(
                self.layer_class(
                    self.units, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False,
                    use_bias=True, reset_after=True, input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                    activity_regularizer=regularizers.L2(1e-5)
                )
            )
        else:
            # Usa la configurazione di default per la rete ricorrente
            self.model.add(
                self.layer_class(
                    self.units, activation=self.activation, input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                    activity_regularizer=regularizers.L2(1e-5)
                )
            )
        self.model.add(Dropout(self.dropout))  # Aggiunge un layer di dropout per la regolarizzazione
        self.model.add(Dense(1))  # Layer di output con una sola unità
        self.model.compile(loss='mae', optimizer='adam')  # Compila il modello con la perdita 'mae' e ottimizzatore 'adam'
        self.initialized = True  # Setta il flag di inizializzazione a True

    def train(self):
        """
        Allena il modello usando i dati caricati.
        """
        if self.initialized:
            tf.random.set_seed(1337)  # Imposta un seme per la riproducibilità
            self.history = self.model.fit(
                self.train_X, self.train_y, epochs=self.epochs, batch_size=self.batch_size,
                validation_data=(self.test_X, self.test_y), verbose=0, shuffle=False
            )
        else:
            print('Error: the model is not initialized yet. Call the initialize method and then retry.')

    def tuning(self, parameters=-1, cudnn=False):
        """
        Esegue la ricerca dei migliori iperparametri per il modello.

        Args:
        - parameters: Dizionario con i parametri da esplorare per la ricerca. Se -1, usa valori di default.
        - cudnn: Booleano che indica se usare CuDNN per ottimizzazioni specifiche (utile per GPU).
        """
        if parameters == -1:
            # Parametri di default per la ricerca
            parameters = {
                "units": [50, 100, 150],
                "epochs": [50, 100, 150],
                "batch_size": [8, 16, 32],
                "dropout": [0.2]
            }

        models, params, rmses = [], [], []  # Liste per salvare i modelli, parametri e metriche RMSE

        # Itera su tutti i possibili valori dei parametri
        for units in parameters['units']:
            for epochs in parameters['epochs']:
                for batch_size in parameters['batch_size']:
                    for dropout in parameters['dropout']:
                        model = Sequential()
                        if cudnn:
                            # Modello con ottimizzazioni CuDNN
                            model.add(
                                self.layer_class(
                                    units, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0,
                                    unroll=False, use_bias=True, input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                    bias_regularizer=regularizers.L2(1e-4),
                                    activity_regularizer=regularizers.L2(1e-5)
                                )
                            )
                        else:
                            # Modello senza ottimizzazioni specifiche
                            model.add(
                                self.layer_class(
                                    units, activation='relu', input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                    bias_regularizer=regularizers.L2(1e-4),
                                    activity_regularizer=regularizers.L2(1e-5)
                                )
                            )
                        model.add(Dropout(dropout))  # Aggiunge il layer di dropout
                        model.add(Dense(1))  # Layer di output
                        model.compile(loss='mae', optimizer='adam')  # Compila il modello
                        # Addestra il modello e calcola l'RMSE
                        history = model.fit(self.train_X, self.train_y, epochs=epochs, batch_size=batch_size,
                                            validation_data=(self.test_X, self.test_y), verbose=0, shuffle=False)
                        yhat = model.predict(self.test_X)
                        test_X_reshaped = self.test_X.reshape((self.test_X.shape[0], self.n_features * self.n_days))
                        inv_yhat = np.concatenate((yhat, test_X_reshaped[:, -self.n_features+1:]), axis=1)
                        inv_yhat = self.scaler.inverse_transform(inv_yhat)[:, 0]
                        inv_y = self.scaler.inverse_transform(
                            np.concatenate((self.test_y.reshape((len(self.test_y), 1)), test_X_reshaped[:, -self.n_features+1:]), axis=1)
                        )[:, 0]
                        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))  # Calcola l'RMSE
                        rmses.append(rmse)  # Salva l'RMSE
                        models.append(model)  # Salva il modello
                        params.append([units, epochs, batch_size, dropout])  # Salva i parametri
                        print(rmse, units, epochs, batch_size, dropout)

        # Seleziona il miglior modello in base all'RMSE minimo
        best_index = rmses.index(min(rmses))
        self.model = models[best_index]
        self.model.compile(loss='mae', optimizer='adam')
        self.units, self.epochs, self.batch_size, self.dropout = params[best_index]
        output = [rmses[best_index], params[best_index]]

    def save(self, path):
        """
        Salva il modello e i suoi iperparametri su disco.

        Args:
        - path: Il percorso in cui salvare il modello.
        """
        df = pd.DataFrame({'units': [self.units], 'epochs': [self.epochs],
                           'batch_size': [self.batch_size], 'dropout': [self.dropout]})
        df.to_excel(f'{path}HP.xlsx')  # Salva gli iperparametri in un file Excel
        model_json = self.model.to_json()  # Serializza il modello in formato JSON
        with open(f'{path}model.json', "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(f"{path}model.h5")  # Salva i pesi del modello
        print("Saved model to disk")

    def load(self, path):
        """
        Carica il modello e i suoi iperparametri da disco.

        Args:
        - path: Il percorso da cui caricare il modello.
        """
        df = pd.read_excel(f'{path}HP.xlsx', index_col=0)  # Carica gli iperparametri
        self.units, self.epochs, self.batch_size, self.dropout = df.iloc[0]
        with open(f'{path}model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)  # Ricostruisce il modello da JSON
        self.model.load_weights(f"{path}model.h5")  # Carica i pesi
        self.model.compile(loss='mae', optimizer='adam')  # Ricompila il modello
        print("Loaded model from disk")

    def forecast(self):
        """
        Esegue la previsione sui dati di test.

        Returns:
        - Una lista contenente l'RMSE, le previsioni e i valori reali.
        """
        tf.random.set_seed(1337)  # Imposta un seme per la riproducibilità
        yhat = self.model.predict(self.test_X)
        test_X_reshaped = self.test_X.reshape((self.test_X.shape[0], self.n_features * self.n_days))
        self.inv_yhat = self.scaler.inverse_transform(np.concatenate((yhat, test_X_reshaped[:, -self.n_features+1:]), axis=1))[:, 0]
        self.inv_y = self.scaler.inverse_transform(
            np.concatenate((self.test_y.reshape((len(self.test_y), 1)), test_X_reshaped[:, -self.n_features+1:]), axis=1)
        )[:, 0]
        self.rmse = sqrt(mean_squared_error(self.inv_y, self.inv_yhat))  # Calcola l'RMSE
        return [self.rmse, self.inv_yhat, self.inv_y, ['rmse', 'prediction', 'actual']]



