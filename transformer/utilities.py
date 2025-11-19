# ============================================================
# Import required TensorFlow and Keras components
# ============================================================
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import (
    Dense, Dropout, GRU, LayerNormalization, Conv1D,
    MultiHeadAttention, LSTM
)
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers
from math import sqrt
import tensorflow as tf


# ============================================================
# Transformer Encoder Block
# ============================================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Creates a Transformer encoder block.

    Args:
        inputs: Input tensor to the encoder block.
        head_size: Dimensionality of each attention head.
        num_heads: Number of attention heads.
        ff_dim: Hidden layer size in the feed-forward network.
        dropout: Dropout rate for regularization.

    Returns:
        A tensor resulting from applying Transformer encoder operations.
    """

    # Layer Normalization to stabilize activations
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    # Multi-Head Self-Attention
    x = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(x, x)

    # Dropout after attention
    x = layers.Dropout(dropout)(x)

    # Residual connection (skip connection)
    res = x + inputs

    # Second Layer Normalization before the feed-forward part
    x = layers.LayerNormalization(epsilon=1e-6)(res)

    # First feed-forward transformation (1D convolution)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)

    # Dropout for regularization
    x = layers.Dropout(dropout)(x)

    # Second feed-forward transformation to restore original dimensionality
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    # Final residual connection
    return x + res


# ============================================================
# Build Transformer-Based Model
# ============================================================
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
    Builds a Keras model using multiple Transformer encoder blocks
    followed by a Multi-Layer Perceptron (MLP).

    Args:
        input_shape: Shape of the input (e.g., (timesteps, features)).
        head_size: Dimensionality of each attention head.
        num_heads: Number of attention heads.
        ff_dim: Hidden layer size in the feed-forward network.
        num_transformer_blocks: Number of Transformer blocks to stack.
        mlp_units: List defining the number of units for each MLP layer.
        dropout: Dropout rate inside Transformer blocks.
        mlp_dropout: Dropout rate inside the MLP layers.

    Returns:
        A compiled Keras model.
    """

    # Model input definition
    inputs = keras.Input(shape=input_shape)

    # Start from the input tensor
    x = inputs

    # Stack multiple Transformer blocks
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Global Average Pooling to flatten temporal dimension
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    # Build the MLP head
    for units in mlp_units:
        x = layers.Dense(units, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)

    # Final output layer (regression)
    outputs = layers.Dense(1)(x)

    # Create and return the model
    return keras.Model(inputs, outputs)
