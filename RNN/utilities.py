# Import necessary libraries for data manipulation, statistical analysis, and deep learning
import pandas as pd  # For handling tabular data
import numpy as np  # For numerical computations and array operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For advanced visualization
from scipy import stats  # For statistical analysis
from scipy.stats import levene  # For testing homogeneity of variance
import statsmodels.api as sm  # For statistical modeling
from statsmodels.formula.api import ols  # For linear regression
from bioinfokit.analys import stat  # For advanced statistical tools
from tensorflow.keras.models import Sequential, model_from_json  # For building deep learning models
from tensorflow.keras.layers import Dense, Dropout, GRU, LayerNormalization, Conv1D, MultiHeadAttention, LSTM  # Neural network layers
from tensorflow.keras import regularizers  # For model regularization
from math import sqrt  # For basic mathematical calculations
from sklearn.metrics import mean_squared_error  # For model evaluation
import tensorflow as tf  # For deep learning
from sklearn.preprocessing import MinMaxScaler  # For data normalization
from tensorflow import keras  # For neural network implementation
from tensorflow.keras import layers  # For constructing neural network layers

# ------------------------------------------------------
# Utility Functions for Time Series Supervised Learning
# ------------------------------------------------------

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Converts a time series into a supervised learning dataset.
    
    Args:
        data (array-like or pd.DataFrame): Input time series data.
        n_in (int): Number of lag observations as input (X).
        n_out (int): Number of future observations as output (y).
        dropnan (bool): Whether to remove rows with NaN values.
    
    Returns:
        pd.DataFrame: Supervised learning DataFrame with lagged inputs and future outputs.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # Input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # Forecast sequence (t, t+1, ... t+n_out)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # Concatenate all columns
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg

def dataframe_to_reframe(dataframe, n_days):
    """
    Normalizes a DataFrame and converts it into supervised learning format.

    Args:
        dataframe (pd.DataFrame): Input data.
        n_days (int): Number of past days to consider for supervised learning.
    
    Returns:
        tuple:
            scaler (MinMaxScaler): Scaler object for inverse normalization.
            n_features (int): Number of features in the DataFrame.
            reframed (pd.DataFrame): Supervised learning formatted DataFrame.
    """
    # Convert DataFrame to float32 values
    values = dataframe.values.astype('float32')
    # Normalize values to range [0,1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    n_features = len(dataframe.columns)
    reframed = series_to_supervised(scaled, n_days, 1)

    # Drop unnecessary columns (future outputs of other variables)
    len_columns = len(reframed.columns) - 1
    for i in range(0, n_features - 1):
        reframed.drop(reframed.columns[len_columns - i], axis=1, inplace=True)
    
    return scaler, n_features, reframed

def split_reframed(reframed, n_train, n_days, n_features, mode='test-train'):
    """
    Splits a supervised DataFrame into training and test sets.

    Args:
        reframed (pd.DataFrame): Supervised learning formatted DataFrame.
        n_train (int): Number of samples for training.
        n_days (int): Number of past days used as input.
        n_features (int): Number of original features.
        mode (str): Split mode ('test-train' for standard split, other for single-sample test).
    
    Returns:
        tuple: (train_X, train_y, test_X, test_y)
            train_X (np.array): Training inputs shaped [samples, n_days, n_features].
            train_y (np.array): Training targets.
            test_X (np.array): Test inputs shaped [samples, n_days, n_features].
            test_y (np.array): Test targets.
    """
    n_obs = n_features * n_days
    values = reframed.values
    train = values[:n_train, :]
    test = values[n_train:, :] if mode == 'test-train' else values[n_train:n_train+1, :]

    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]

    # Reshape input to [samples, timesteps, features] for RNN
    train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
    
    return train_X, train_y, test_X, test_y
