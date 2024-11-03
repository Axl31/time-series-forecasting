from RNN.recurrent_network import RecurrentNetwork

from tensorflow.keras.layers import GRU

class GRUNetwork(RecurrentNetwork):
    def __init__(self, units=100, epochs=100, batch_size=16, dropout=0.2, activation='relu', n_days=7):
        """
        Inizializza la rete neurale ricorrente con layer GRU.

        Args:
        - units: Numero di unità nel layer GRU.
        - epochs: Numero di epoche di addestramento.
        - batch_size: Dimensione del batch per l'addestramento.
        - dropout: Percentuale di dropout per prevenire overfitting.
        - activation: Funzione di attivazione per il layer GRU.
        - n_days: Numero di giorni di input da considerare per la previsione.
        """
        super().__init__(GRU, units, epochs, batch_size, dropout, activation, n_days)
        # Chiama il costruttore della classe base e imposta la classe del layer come GRU
