# Importazione delle librerie necessarie da TensorFlow e Keras
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout, GRU, LayerNormalization, Conv1D, MultiHeadAttention, LSTM
from tensorflow.keras import regularizers
from math import sqrt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Crea un blocco di encoder Transformer.

    Args:
        inputs: Tensore di input per il blocco di encoder.
        head_size: Dimensione di ciascun head nel MultiHeadAttention.
        num_heads: Numero di heads nel MultiHeadAttention.
        ff_dim: Dimensione dell'input per il feed-forward network.
        dropout: Tasso di dropout per la regolarizzazione.

    Returns:
        Un tensore risultante dal blocco di encoder.
    """
    # Normalizzazione Layer per stabilizzare i valori di attivazione
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    # Applicazione della Multi-Head Attention
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)

    # Applicazione del dropout per la regolarizzazione
    x = layers.Dropout(dropout)(x)

    # Residual connection: somma dell'input originale e dell'output dell'attenzione
    res = x + inputs

    # Feed Forward Part: Normalizzazione Layer per stabilizzare l'output
    x = layers.LayerNormalization(epsilon=1e-6)(res)

    # Prima convoluzione per trasformare le rappresentazioni
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)

    # Dropout per regolarizzare il feed-forward network
    x = layers.Dropout(dropout)(x)

    # Seconda convoluzione per riportare la dimensione all'input
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    # Somma del risultato e della connessione residuale
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    """
    Costruisce un modello Keras che utilizza blocchi di encoder Transformer seguiti da una rete MLP.

    Args:
        input_shape: Forma degli input del modello (es. (timesteps, features)).
        head_size: Dimensione di ciascun head nel MultiHeadAttention.
        num_heads: Numero di heads nel MultiHeadAttention.
        ff_dim: Dimensione dell'input per il feed-forward network.
        num_transformer_blocks: Numero di blocchi Transformer da utilizzare.
        mlp_units: Lista delle dimensioni dei layer della MLP.
        dropout: Tasso di dropout per la regolarizzazione nei blocchi Transformer.
        mlp_dropout: Tasso di dropout per la regolarizzazione nella MLP.

    Returns:
        Modello Keras compilato.
    """
    # Definizione dell'input del modello
    inputs = keras.Input(shape=input_shape)

    # Inizializzazione del tensore di input
    x = inputs

    # Costruzione dei blocchi Transformer
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Global Average Pooling per ridurre la dimensionalità
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    # Creazione della parte MLP
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)  # Layer denso con attivazione ReLU
        x = layers.Dropout(mlp_dropout)(x)  # Dropout per regolarizzazione

    # Layer finale per output singolo
    outputs = layers.Dense(1)(x)

    # Creazione del modello con specifica degli input e output
    return keras.Model(inputs, outputs)
