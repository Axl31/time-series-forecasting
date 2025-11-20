from RNN.recurrent_network import RecurrentNetwork

from tensorflow.keras.layers import LSTM

class LSTMNetwork(RecurrentNetwork):
    def __init__(self, units=100, epochs=100, batch_size=16, dropout=0.2, activation='relu', n_days=7):
        """
        Initializes a recurrent neural network using an LSTM layer.

        Args:
            units (int): Number of units in the LSTM layer.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            dropout (float): Dropout rate to prevent overfitting.
            activation (str): Activation function for the LSTM layer.
            n_days (int): Number of past days to consider as input for the prediction.

        Notes:
            This class inherits from RecurrentNetwork and sets the recurrent layer
            to be LSTM by default. All training and preprocessing methods are handled
            by the base class.
        """
        super().__init__(LSTM, units, epochs, batch_size, dropout, activation, n_days)
        # Calls the base class constructor and sets the layer class to LSTM
