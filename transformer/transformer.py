from transformer.utilities import transformer_encoder, build_model
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt

class Transformer():
    def __init__(self, head_size=256, num_heads=4, epochs=100, batch_size=16, dropout=0.2, activation='relu', n_days=7):
        """
        Inizializza il modello Transformer con i parametri specificati.

        Args:
            head_size (int): Dimensione di ciascun head nel MultiHeadAttention.
            num_heads (int): Numero di heads nel MultiHeadAttention.
            epochs (int): Numero di epoche di addestramento.
            batch_size (int): Dimensione del batch per l'addestramento.
            dropout (float): Percentuale di dropout per prevenire l'overfitting.
            activation (str): Funzione di attivazione per i layer densi.
            n_days (int): Numero di giorni di input da considerare per la previsione.
        """
        self.head_size = head_size
        self.num_heads = num_heads
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
            dataframe (pd.DataFrame): Il DataFrame contenente i dati.
            n_test (int): Numero di campioni utilizzati per il test.
        """
        self.df = dataframe
        self.n_train = len(dataframe) - n_test  # Calcola la dimensione del set di addestramento
        
        # Pre-processa i dati e li trasforma in un formato supervisionato
        self.scaler, self.n_features, self.reframed = dataframe_to_reframe(self.df, self.n_days)
        
        # Divide i dati in set di addestramento e test
        self.train_X, self.train_y, self.test_X, self.test_y = split_reframed(self.reframed, self.n_train, self.n_days, self.n_features)

    def get_parameters(self):
        """
        Ritorna i parametri del modello in un dizionario.

        Returns:
            dict: I parametri correnti del modello.
        """
        return {
            'head_size': self.head_size,
            'num_heads': self.num_heads,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'dropout': self.dropout
        }

    def update_parameters(self, params):
        """
        Aggiorna i parametri del modello.

        Args:
            params (dict): Dizionario con i parametri aggiornati.
        """
        self.head_size = params.get('head_size', self.head_size)
        self.num_heads = params.get('num_heads', self.num_heads)
        self.epochs = params.get('epochs', self.epochs)
        self.batch_size = params.get('batch_size', self.batch_size)
        self.dropout = params.get('dropout', self.dropout)

    def initialize(self):
        """
        Inizializza il modello Transformer utilizzando i parametri specificati.
        """
        self.model = build_model(
            self.train_X.shape[1:],
            head_size=self.head_size,
            num_heads=self.num_heads,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=self.dropout,
        )
        # Compila il modello con una funzione di perdita e un ottimizzatore
        self.model.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
        self.initialized = True

    def train(self):
        """
        Allena il modello usando i dati caricati.
        """
        if self.initialized:
            tf.random.set_seed(1337)  # Imposta un seme per la riproducibilità
            self.history = self.model.fit(
                self.train_X, self.train_y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(self.test_X, self.test_y),
                verbose=0,
                shuffle=False
            )
        else:
            print('Error: the model is not initialized yet. Call the initialize method and then retry.')

    def tuning(self, parameters=None):
        """
        Esegue la ricerca degli iperparametri ottimali.

        Args:
            parameters (dict): Dizionario contenente le liste di valori da provare (opzionale).

        Returns:
            list: Il migliore RMSE e i parametri corrispondenti.
        """
        if parameters is None:
            parameters = {
                "head_size": [64, 128, 256],
                "num_heads": [2, 4, 8],
                "epochs": [50, 100, 150],
                "batch_size": [8, 16, 32],
                "dropout": [0.2]
            }

        models = []
        params = []
        rmses = []

        # Itera su tutti i valori degli iperparametri
        for head_size in parameters['head_size']:
            for epochs in parameters['epochs']:
                for batch_size in parameters['batch_size']:
                    for dropout in parameters['dropout']:
                        for num_heads in parameters['num_heads']:
                            # Costruisce e compila il modello
                            model = build_model(
                                self.train_X.shape[1:],
                                head_size=head_size,
                                num_heads=num_heads,
                                ff_dim=4,
                                num_transformer_blocks=4,
                                mlp_units=[128],
                                mlp_dropout=0.4,
                                dropout=dropout,
                            )
                            model.compile(loss='mae', optimizer='adam')

                            # Allena il modello
                            history = model.fit(
                                self.train_X, self.train_y,
                                epochs=epochs, batch_size=batch_size,
                                validation_data=(self.test_X, self.test_y),
                                verbose=0, shuffle=False
                            )

                            # Previsione e calcolo dell'RMSE
                            tf.random.set_seed(1337)
                            yhat = model.predict(self.test_X)

                            # Reshape dei dati per la scalatura inversa
                            test_X_reshaped = self.test_X.reshape((self.test_X.shape[0], self.n_features * self.n_days))
                            inv_yhat = self.scaler.inverse_transform(np.concatenate((yhat, test_X_reshaped[:, -self.n_features+1:]), axis=1))[:, 0]
                            inv_y = self.scaler.inverse_transform(np.concatenate((self.test_y.reshape((len(self.test_y), 1)), test_X_reshaped[:, -self.n_features+1:]), axis=1))[:, 0]
                            
                            # Calcolo dell'RMSE
                            rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
                            rmses.append(rmse)
                            models.append(model)
                            params.append([head_size, num_heads, epochs, batch_size, dropout])
                            
                            print(f"RMSE: {rmse}, Parameters: {head_size}, {num_heads}, {epochs}, {batch_size}, {dropout}")

        # Trova il miglior modello e i parametri
        min_rmse_index = rmses.index(min(rmses))
        self.model = models[min_rmse_index]
        self.model.compile(loss='mae', optimizer='adam')

        # Aggiorna i parametri con quelli ottimali
        self.head_size, self.num_heads, self.epochs, self.batch_size, self.dropout = params[min_rmse_index]
        return [rmses[min_rmse_index], params[min_rmse_index]]

    def save(self, path):
        """
        Salva il modello e i suoi iperparametri su disco.

        Args:
            path (str): Il percorso in cui salvare il modello.
        """
        # Salva gli iperparametri in un file Excel
        df = pd.DataFrame({
            'head_size': [self.head_size],
            'num_heads': [self.num_heads],
            'epochs': [self.epochs],
            'batch_size': [self.batch_size],
            'dropout': [self.dropout]
        })
        df.to_excel(f'{path}HP.xlsx')

        # Serializza e salva il modello
        model_json = self.model.to_json()
        with open(f'{path}model.json', "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(f"{path}model.h5")
        print("Model saved to disk.")

    def load(self, path):
        """
        Carica il modello e i suoi iperparametri da disco.

        Args:
            path (str): Il percorso da cui caricare il modello.
        """
        # Carica gli iperparametri da un file Excel
        df = pd.read_excel(f'{path}HP.xlsx', index_col=0)
        self.head_size = df['head_size'][0]
        self.num_heads = df['num_heads'][0]
        self.epochs = df['epochs'][0]
        self.batch_size = df['batch_size'][0]
        self.dropout = df['dropout'][0]

        # Carica e compila il modello
        with open(f'{path}model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(f"{path}model.h5")
        self.model.compile(loss='mae', optimizer='adam')
        print("Model loaded from disk.")

    def forecast(self):
        """
        Esegue la previsione sui dati di test.

        Returns:
            list: Una lista contenente l'RMSE, le previsioni e i valori reali.
        """
        tf.random.set_seed(1337)  # Imposta un seme per la riproducibilità
        yhat = self.model.predict(self.test_X)

        # Reshape dei dati per la scalatura inversa
        test_X_reshaped = self.test_X.reshape((self.test_X.shape[0], self.n_features * self.n_days))
        self.inv_yhat = self.scaler.inverse_transform(np.concatenate((yhat, test_X_reshaped[:, -self.n_features+1:]), axis=1))[:, 0]
        self.inv_y = self.scaler.inverse_transform(np.concatenate((self.test_y.reshape((len(self.test_y), 1)), test_X_reshaped[:, -self.n_features+1:]), axis=1))[:, 0]
        
        # Calcola l'RMSE
        self.rmse = sqrt(mean_squared_error(self.inv_y, self.inv_yhat))
        return [self.rmse, self.inv_yhat, self.inv_y, ['rmse', 'prediction', 'actual']]


