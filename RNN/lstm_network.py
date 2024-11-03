from recurrent_network import RecurrentNetwork

from tensorflow.keras.layers import LSTM

class LSTMNetwork(RecurrentNetwork):
    def __init__(self, units=100, epochs=100, batch_size=16, dropout=0.2, activation='relu', n_days=7):
        """
        Inizializza la rete neurale ricorrente con layer LSTM.

        Args:
        - units: Numero di unità nel layer LSTM.
        - epochs: Numero di epoche di addestramento.
        - batch_size: Dimensione del batch per l'addestramento.
        - dropout: Percentuale di dropout per prevenire overfitting.
        - activation: Funzione di attivazione per il layer LSTM.
        - n_days: Numero di giorni di input da considerare per la previsione.
        """
        super().__init__(LSTM, units, epochs, batch_size, dropout, activation, n_days)
        # Chiama il costruttore della classe base e imposta la classe del layer come LSTM
