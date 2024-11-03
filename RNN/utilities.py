# Importazione delle librerie necessarie per il calcolo, la manipolazione dei dati, la costruzione di modelli di deep learning e l'analisi statistica
import pandas as pd  # Per la gestione e manipolazione di dati tabulari
import numpy as np  # Per il calcolo numerico e la manipolazione degli array
import matplotlib.pyplot as plt  # Per la visualizzazione dei dati
import seaborn as sns  # Per la visualizzazione avanzata dei dati
from scipy import stats  # Per le statistiche avanzate
from scipy.stats import levene  # Per il test di omogeneità delle varianze
import statsmodels.api as sm  # Per la modellazione statistica
from statsmodels.formula.api import ols  # Per la regressione lineare
from bioinfokit.analys import stat  # Per strumenti di analisi statistica avanzata
from tensorflow.keras.models import Sequential, model_from_json  # Per la costruzione di modelli di deep learning
from tensorflow.keras.layers import Dense, Dropout, GRU, LayerNormalization, Conv1D, MultiHeadAttention, LSTM  # Strati di rete neurale
from tensorflow.keras import regularizers  # Per aggiungere regolarizzazione ai modelli
from math import sqrt  # Per calcoli matematici di base
from sklearn.metrics import mean_squared_error  # Per la valutazione delle prestazioni del modello
import tensorflow as tf  # Per la costruzione di modelli di deep learning
from sklearn.preprocessing import MinMaxScaler  # Per la normalizzazione dei dati
from tensorflow import keras  # Per l'implementazione delle reti neurali
from tensorflow.keras import layers  # Per la costruzione di strati di rete neurale

# Funzione per trasformare una serie temporale in una forma adatta per modelli di machine learning supervisionati
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Trasforma una serie temporale in un formato supervisionato.
    
    Parametri:
    - data: Serie o array di input.
    - n_in: Numero di lag (ritardi) di input (numero di periodi precedenti da includere).
    - n_out: Numero di periodi futuri da prevedere.
    - dropnan: Booleano che indica se eliminare o meno i valori NaN.

    Ritorna:
    - DataFrame trasformato con dati lag (input passati) e output futuri.
    """
    # Determina il numero di variabili nella serie temporale
    n_vars = 1 if type(data) is list else data.shape[1]
    # Crea un DataFrame a partire dai dati
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # Crea colonne per i lag passati (n_in)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))  # Sposta i dati verso il basso di i posizioni per creare ritardi
        # Nomina le colonne basandosi sulla variabile e sul ritardo
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # Crea colonne per gli output futuri (n_out)
    for i in range(0, n_out):
        cols.append(df.shift(i))  # Sposta i dati verso il basso di i posizioni per gli output futuri
        if i == 0:
            # Nomina le colonne per i tempi presenti
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            # Nomina le colonne per i tempi futuri
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # Unisce tutte le colonne in un unico DataFrame
    agg = pd.concat(cols, axis=1)
    agg.columns = names  # Assegna i nomi alle colonne

    # Rimuove eventuali righe con valori NaN se richiesto
    if dropnan:
        agg.dropna(inplace=True)
    
    return agg

# Funzione per normalizzare un DataFrame e convertirlo in un formato supervisionato
def dataframe_to_reframe(dataframe, n_days):
    """
    Normalizza un DataFrame e lo converte in una forma supervisionata.

    Parametri:
    - dataframe: DataFrame di input.
    - n_days: Numero di giorni di input per la serie supervisionata.

    Ritorna:
    - scaler: Oggetto MinMaxScaler per la normalizzazione inversa.
    - n_features: Numero di caratteristiche del DataFrame.
    - reframed: DataFrame trasformato in forma supervisionata.
    """
    # Estrae i valori dal DataFrame e li converte in float32
    values = dataframe.values
    values = values.astype('float32')
    # Normalizza i dati tra 0 e 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    
    # Determina il numero di caratteristiche (colonne) nel DataFrame
    n_features = len(dataframe.columns)
    # Converte i dati normalizzati in formato supervisionato
    reframed = series_to_supervised(scaled, n_days, 1)
    
    # Elimina le colonne non necessarie (cioè output futuri di altre variabili)
    len_columns = len(reframed.columns) - 1
    for i in range(0, n_features - 1):
        reframed.drop(reframed.columns[len_columns - i], axis=1, inplace=True)
    
    return scaler, n_features, reframed

# Funzione per dividere i dati trasformati in set di addestramento e test
def split_reframed(reframed, n_train, n_days, n_features, mode='test-train'):
    """
    Divide i dati trasformati in set di addestramento e test.

    Parametri:
    - reframed: DataFrame trasformato.
    - n_train: Numero di campioni per il set di addestramento.
    - n_days: Numero di giorni per l'input supervisionato.
    - n_features: Numero di caratteristiche originali.
    - mode: Modalità di divisione ('test-train' per suddivisione standard, altro per singolo campione).

    Ritorna:
    - train_X, train_y: Dati di addestramento (input e target).
    - test_X, test_y: Dati di test (input e target).
    """
    # Calcola il numero di osservazioni (input supervisionati)
    n_obs = n_features * n_days
    # Estrae i valori dal DataFrame trasformato
    values = reframed.values
    # Divide i dati in set di addestramento e test
    train = values[:n_train, :]
    if mode == 'test-train':
        test = values[n_train:, :]
    else:
        test = values[n_train:n_train+1, :]
    
    # Divide i dati in input (X) e target (y) per l'addestramento
    train_X, train_y = train[:, :n_obs], train[:, -1]
    # Divide i dati in input (X) e target (y) per il test
    test_X, test_y = test[:, :n_obs], test[:, -1]
    
    # Rimodella gli input in [campioni, n_days, n_features] per l'addestramento dei modelli di deep learning
    train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
    
    return train_X, train_y, test_X, test_y
