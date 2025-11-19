# NSE Stock Price Forecasting

## Project Overview
This project focuses on forecasting stock prices using historical time series data from the **National Stock Exchange of India Ltd. (NSE)**. The main goal is to leverage recurrent neural networks (RNNs), including **LSTM** and **GRU**, to predict future stock prices and evaluate model performance using **Root Mean Squared Error (RMSE)**.

### Dataset
The dataset is publicly available on Kaggle: [National Stock Exchange Time Series](https://www.kaggle.com/datasets/atulanandjha/national-stock-exchange-time-series)

**Context:**  
The NSE is one of the largest stock exchanges in India, established in 1992 in Mumbai. It was the first demutualized electronic exchange in India and launched screen-based trading in 1994. NSE introduced index futures and online trading, becoming the country’s largest exchange by turnover.

**Contents:**  
The dataset contains historical stock data for NSE-listed companies, including daily open, high, low, close, and volume information.

---

## Project Goals
- Transform raw stock time series into supervised learning datasets.  
- Build and train **RNN models** (LSTM and GRU) to forecast stock prices.  
- Evaluate and compare model performance using **RMSE**.  
- Visualize predictions against actual stock prices.

---

## Methodology

### 1. Data Preprocessing
- Handle missing values and normalize features using **MinMaxScaler**.  
- Convert time series data into a supervised learning format, including lag features (`n_days`) as input and next day prices as targets.  
- Split the data into **training** and **testing sets**.

### 2. Model Architecture
- Implement **RNNs** using TensorFlow/Keras:  
  - **LSTMNetwork**: Long Short-Term Memory network for sequential data.  
  - **GRUNetwork**: Gated Recurrent Unit network for faster training with comparable accuracy.  
- Add **Dropout layers** to reduce overfitting.  
- Compile models using **MAE** as loss function and **Adam optimizer**.

### 3. Training and Hyperparameter Tuning
- Train models on training set and validate on test set.  
- Optionally perform **hyperparameter tuning** for units, epochs, batch size, and dropout.

### 4. Forecasting and Evaluation
- Make predictions on test data.  
- Inverse transform normalized data to obtain actual stock price predictions.  
- Evaluate performance using **RMSE**.  
- Visualize predictions vs actual stock prices.
