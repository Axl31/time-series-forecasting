from RNN.utilities import dataframe_to_reframe, split_reframed
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import LSTM, GRU, Dropout, Dense, Input
from tensorflow.keras import regularizers
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import tensorflow as tf

class RecurrentNetwork:
    """
    Base class for creating recurrent neural networks (LSTM or GRU) for time series forecasting.

    Attributes:
        layer_class (class): Recurrent layer class (LSTM or GRU).
        units (int): Number of units in the recurrent layer.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        dropout (float): Dropout rate to prevent overfitting.
        activation (str): Activation function for the recurrent layer.
        n_days (int): Number of past days used as input for prediction.
        initialized (bool): Flag indicating if the model has been initialized.
    """

    def __init__(self, rnn_type, units=100, epochs=100, batch_size=16, dropout=0.2, activation='relu', n_days=7):
        """
        Initializes the recurrent network with the specified parameters.

        Args:
            rnn_type (class): Recurrent layer class (LSTM or GRU).
            units (int): Number of units in the recurrent layer.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            dropout (float): Dropout rate to prevent overfitting.
            activation (str): Activation function for the recurrent layer.
            n_days (int): Number of past days to consider as input for prediction.
        """
        self.layer_class = rnn_type
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.activation = activation
        self.n_days = n_days
        self.initialized = False

    def upload(self, dataframe, n_test=14):
        """
        Loads and prepares data for training and testing.

        Args:
            dataframe (pd.DataFrame): Input data.
            n_test (int): Number of samples to reserve for testing.
        """
        self.df = dataframe
        self.n_train = int(len(dataframe) - n_test)
        self.scaler, self.n_features, self.reframed = dataframe_to_reframe(self.df, self.n_days)
        self.train_X, self.train_y, self.test_X, self.test_y = split_reframed(
            self.reframed, self.n_train, self.n_days, self.n_features
        )

    def get_parameters(self):
        """
        Returns model parameters as a dictionary.

        Returns:
            dict: Dictionary containing 'units', 'epochs', 'batch_size', and 'dropout'.
        """
        return {'units': self.units, 'epochs': self.epochs, 'batch_size': self.batch_size, 'dropout': self.dropout}

    def update_parameters(self, params):
        """
        Updates the model's parameters.

        Args:
            params (dict): Dictionary containing updated parameters.
        """
        self.units = params['units']
        self.epochs = params['epochs']
        self.batch_size = params['batch_size']
        self.dropout = params['dropout']

    def initialize(self, cudnn=False):
        """
        Initializes the model with the recurrent layer and compilation settings.

        Args:
            cudnn (bool): If True, use CuDNN-optimized layers for GPU acceleration.
        """
        self.model = Sequential()
        self.model.add(
            Input(shape=(self.train_X.shape[1], self.train_X.shape[2]))
        )
        if cudnn:
            self.model.add(
                self.layer_class(
                    self.units,
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    recurrent_dropout=0,
                    unroll=False,
                    use_bias=True,
                    reset_after=True,
                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                    activity_regularizer=regularizers.L2(1e-5)
                )
            )
        else:
            self.model.add(
                self.layer_class(
                    self.units,
                    activation=self.activation,
                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                    bias_regularizer=regularizers.L2(1e-4),
                    activity_regularizer=regularizers.L2(1e-5)
                )
            )
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(1))
        self.model.compile(loss='mae', optimizer='adam')
        self.initialized = True

    def train(self):
        """
        Trains the model using the uploaded data.
        """
        if self.initialized:
            tf.random.set_seed(1337)
            self.history = self.model.fit(
                self.train_X,
                self.train_y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(self.test_X, self.test_y),
                verbose=0,
                shuffle=False
            )
        else:
            print('Error: the model is not initialized yet. Call initialize() first.')

    def tuning(self, parameters=-1, cudnn=False):
        """
        Performs hyperparameter tuning to find the best model configuration.

        Args:
            parameters (dict or int): Dictionary with hyperparameter lists to explore. If -1, uses default values.
            cudnn (bool): If True, use CuDNN-optimized layers for GPU acceleration.

        Returns:
            list: [best_rmse, best_parameters]
        """
        if parameters == -1:
            parameters = {
                "units": [50, 100, 150],
                "epochs": [50, 100, 150],
                "batch_size": [8, 16, 32],
                "dropout": [0.2]
            }

        models, params, rmses = [], [], []

        for units in parameters['units']:
            for epochs in parameters['epochs']:
                for batch_size in parameters['batch_size']:
                    for dropout in parameters['dropout']:
                        model = Sequential()
                        if cudnn:
                            model.add(
                                self.layer_class(
                                    units, activation='tanh', recurrent_activation='sigmoid',
                                    recurrent_dropout=0, unroll=False, use_bias=True,
                                    input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                    bias_regularizer=regularizers.L2(1e-4),
                                    activity_regularizer=regularizers.L2(1e-5)
                                )
                            )
                        else:
                            model.add(
                                self.layer_class(
                                    units, activation='relu',
                                    kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                    input_shape=(self.train_X.shape[1], self.train_X.shape[2]),
                                    bias_regularizer=regularizers.L2(1e-4),
                                    activity_regularizer=regularizers.L2(1e-5)
                                )
                            )
                        model.add(Dropout(dropout))
                        model.add(Dense(1))
                        model.compile(loss='mae', optimizer='adam')

                        history = model.fit(
                            self.train_X,
                            self.train_y,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(self.test_X, self.test_y),
                            verbose=0,
                            shuffle=False
                        )

                        yhat = model.predict(self.test_X)
                        test_X_reshaped = self.test_X.reshape((self.test_X.shape[0], self.n_features * self.n_days))
                        inv_yhat = np.concatenate((yhat, test_X_reshaped[:, -self.n_features+1:]), axis=1)
                        inv_yhat = self.scaler.inverse_transform(inv_yhat)[:, 0]

                        inv_y = self.scaler.inverse_transform(
                            np.concatenate((self.test_y.reshape((len(self.test_y), 1)),
                                            test_X_reshaped[:, -self.n_features+1:]), axis=1)
                        )[:, 0]

                        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
                        rmses.append(rmse)
                        models.append(model)
                        params.append([units, epochs, batch_size, dropout])
                        print(rmse, units, epochs, batch_size, dropout)

        best_index = rmses.index(min(rmses))
        self.model = models[best_index]
        self.model.compile(loss='mae', optimizer='adam')
        self.units, self.epochs, self.batch_size, self.dropout = params[best_index]
        return [rmses[best_index], params[best_index]]

    def save(self, path):
        """
        Saves the model and its hyperparameters to disk.

        Args:
            path (str): Directory path to save the model.
        """
        df = pd.DataFrame({
            'units': [self.units],
            'epochs': [self.epochs],
            'batch_size': [self.batch_size],
            'dropout': [self.dropout]
        })
        df.to_excel(f'{path}HP.xlsx')
        model_json = self.model.to_json()
        with open(f'{path}model.json', "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(f"{path}model.h5")
        print("Saved model to disk")

    def load(self, path):
        """
        Loads the model and its hyperparameters from disk.

        Args:
            path (str): Directory path to load the model from.
        """
        df = pd.read_excel(f'{path}HP.xlsx', index_col=0)
        self.units, self.epochs, self.batch_size, self.dropout = df.iloc[0]
        with open(f'{path}model.json', 'r') as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(f"{path}model.h5")
        self.model.compile(loss='mae', optimizer='adam')
        print("Loaded model from disk")

    def forecast(self):
        """
        Makes predictions on the test data and calculates RMSE.

        Returns:
            list: [RMSE, predicted_values, actual_values, ['rmse', 'prediction', 'actual']]
        """
        tf.random.set_seed(1337)
        yhat = self.model.predict(self.test_X)
        test_X_reshaped = self.test_X.reshape((self.test_X.shape[0], self.n_features * self.n_days))
        self.inv_yhat = self.scaler.inverse_transform(
            np.concatenate((yhat, test_X_reshaped[:, -self.n_features+1:]), axis=1)
        )[:, 0]
        self.inv_y = self.scaler.inverse_transform(
            np.concatenate((self.test_y.reshape((len(self.test_y), 1)),
                            test_X_reshaped[:, -self.n_features+1:]), axis=1)
        )[:, 0]
        self.rmse = sqrt(mean_squared_error(self.inv_y, self.inv_yhat))
        return [self.rmse, self.inv_yhat, self.inv_y, ['rmse', 'prediction', 'actual']]


