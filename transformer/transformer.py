from transformer.utilities import transformer_encoder, build_model
from RNN.utilities import dataframe_to_reframe, split_reframed, model_from_json
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt


class Transformer:
    """
    A complete Transformer-based forecasting model with utilities for
    data preparation, training, hyperparameter tuning, saving, loading,
    and prediction.
    """

    def __init__(self, head_size=256, num_heads=4, epochs=100, batch_size=16,
                 dropout=0.2, activation='relu', n_days=7):
        """
        Initialize the Transformer model with the provided parameters.

        Args:
            head_size (int): Size of each attention head.
            num_heads (int): Number of attention heads.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size used for training.
            dropout (float): Dropout rate for regularization.
            activation (str): Activation function for dense layers.
            n_days (int): Number of past time steps used for forecasting.
        """
        self.head_size = head_size
        self.num_heads = num_heads
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.activation = activation
        self.n_days = n_days
        self.initialized = False  # Tracks whether the model is built

    # ----------------------------------------------------------------------
    def upload(self, dataframe, n_test=14):
        """
        Load and prepare the dataset for training/testing.

        Args:
            dataframe (pd.DataFrame): Input dataset.
            n_test (int): Number of samples used for testing.
        """
        self.df = dataframe
        self.n_train = len(dataframe) - n_test

        # Convert dataset into supervised learning format
        self.scaler, self.n_features, self.reframed = dataframe_to_reframe(
            self.df, self.n_days
        )

        # Split data into train and test sets
        self.train_X, self.train_y, self.test_X, self.test_y = split_reframed(
            self.reframed, self.n_train, self.n_days, self.n_features
        )

    # ----------------------------------------------------------------------
    def get_parameters(self):
        """
        Return the current model hyperparameters.

        Returns:
            dict: Dictionary of model parameters.
        """
        return {
            'head_size': self.head_size,
            'num_heads': self.num_heads,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'dropout': self.dropout
        }

    # ----------------------------------------------------------------------
    def update_parameters(self, params):
        """
        Update the model parameters.

        Args:
            params (dict): New parameters to apply.
        """
        self.head_size = params.get('head_size', self.head_size)
        self.num_heads = params.get('num_heads', self.num_heads)
        self.epochs = params.get('epochs', self.epochs)
        self.batch_size = params.get('batch_size', self.batch_size)
        self.dropout = params.get('dropout', self.dropout)

    # ----------------------------------------------------------------------
    def initialize(self):
        """
        Build and compile the Transformer model using the current parameters.
        """
        self.model = build_model(
            input_shape=self.train_X.shape[1:],
            head_size=self.head_size,
            num_heads=self.num_heads,
            ff_dim=4,
            num_transformer_blocks=4,
            mlp_units=[128],
            mlp_dropout=0.4,
            dropout=self.dropout,
        )

        self.model.compile(
            loss='mae',
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4)
        )

        self.initialized = True

    # ----------------------------------------------------------------------
    def train(self):
        """
        Train the Transformer model on the prepared dataset.
        """
        if not self.initialized:
            print("Error: model not initialized. Call initialize() first.")
            return

        tf.random.set_seed(1337)
        self.history = self.model.fit(
            self.train_X, self.train_y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(self.test_X, self.test_y),
            verbose=0,
            shuffle=False
        )

    # ----------------------------------------------------------------------
    def tuning(self, parameters=None):
        """
        Perform hyperparameter tuning using grid search.

        Args:
            parameters (dict, optional): Dictionary containing lists of
                                         hyperparameters to test.

        Returns:
            list: Best RMSE and the corresponding parameters.
        """
        if parameters is None:
            parameters = {
                "head_size": [64, 128, 256],
                "num_heads": [2, 4, 8],
                "epochs": [50, 100, 150],
                "batch_size": [8, 16, 32],
                "dropout": [0.2]
            }

        models, params, rmses = [], [], []

        # Iterate through hyperparameter combinations
        for head_size in parameters['head_size']:
            for epochs in parameters['epochs']:
                for batch_size in parameters['batch_size']:
                    for dropout in parameters['dropout']:
                        for num_heads in parameters['num_heads']:

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

                            history = model.fit(
                                self.train_X, self.train_y,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(self.test_X, self.test_y),
                                verbose=0,
                                shuffle=False
                            )

                            # Forecast
                            tf.random.set_seed(1337)
                            yhat = model.predict(self.test_X)

                            # Inverse scaling
                            test_X_reshaped = self.test_X.reshape(
                                (self.test_X.shape[0],
                                 self.n_features * self.n_days)
                            )

                            inv_yhat = self.scaler.inverse_transform(
                                np.concatenate(
                                    (yhat, test_X_reshaped[:, -self.n_features+1:]),
                                    axis=1
                                )
                            )[:, 0]

                            inv_y = self.scaler.inverse_transform(
                                np.concatenate(
                                    (self.test_y.reshape((len(self.test_y), 1)),
                                     test_X_reshaped[:, -self.n_features+1:]),
                                    axis=1
                                )
                            )[:, 0]

                            rmse = sqrt(mean_squared_error(inv_y, inv_yhat))

                            models.append(model)
                            rmses.append(rmse)
                            params.append(
                                [head_size, num_heads, epochs, batch_size, dropout]
                            )

                            print(f"RMSE: {rmse}, Parameters: "
                                  f"{head_size}, {num_heads}, {epochs}, "
                                  f"{batch_size}, {dropout}")

        # Retrieve the best model
        best_index = rmses.index(min(rmses))
        self.model = models[best_index]
        self.model.compile(loss='mae', optimizer='adam')

        # Update parameters
        self.head_size, self.num_heads, self.epochs, self.batch_size, self.dropout = params[best_index]

        return [rmses[best_index], params[best_index]]

    # ----------------------------------------------------------------------
    def save(self, path):
        """
        Save model architecture, weights, and hyperparameters.

        Args:
            path (str): Directory in which to save the model.
        """
        # Save hyperparameters
        df = pd.DataFrame({
            'head_size': [self.head_size],
            'num_heads': [self.num_heads],
            'epochs': [self.epochs],
            'batch_size': [self.batch_size],
            'dropout': [self.dropout]
        })
        df.to_excel(f'{path}HP.xlsx')

        # Save model architecture and weights
        model_json = self.model.to_json()
        with open(f'{path}model.json', "w") as json_file:
            json_file.write(model_json)

        self.model.save_weights(f"{path}model.h5")
        print("Model saved to disk.")

    # ----------------------------------------------------------------------
    def load(self, path):
        """
        Load model architecture, weights, and hyperparameters.

        Args:
            path (str): Directory from which to load the model.
        """
        df = pd.read_excel(f'{path}HP.xlsx', index_col=0)

        self.head_size = df['head_size'][0]
        self.num_heads = df['num_heads'][0]
        self.epochs = df['epochs'][0]
        self.batch_size = df['batch_size'][0]
        self.dropout = df['dropout'][0]

        # Load model JSON
        with open(f'{path}model.json', 'r') as json_file:
            loaded_model_json = json_file.read()

        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(f"{path}model.h5")
        self.model.compile(loss='mae', optimizer='adam')

        print("Model loaded from disk.")

    # ----------------------------------------------------------------------
    def forecast(self):
        """
        Run forecasting on the test dataset.

        Returns:
            list: RMSE, predictions, actual values, and labels.
        """
        tf.random.set_seed(1337)
        yhat = self.model.predict(self.test_X)

        # Inverse transformation
        test_X_reshaped = self.test_X.reshape(
            (self.test_X.shape[0], self.n_features * self.n_days)
        )

        self.inv_yhat = self.scaler.inverse_transform(
            np.concatenate((yhat, test_X_reshaped[:, -self.n_features+1:]), axis=1)
        )[:, 0]

        self.inv_y = self.scaler.inverse_transform(
            np.concatenate(
                (self.test_y.reshape((len(self.test_y), 1)),
                 test_X_reshaped[:, -self.n_features+1:]),
                axis=1
            )
        )[:, 0]

        self.rmse = sqrt(mean_squared_error(self.inv_y, self.inv_yhat))

        return [self.rmse, self.inv_yhat, self.inv_y, ['rmse', 'prediction', 'actual']]
