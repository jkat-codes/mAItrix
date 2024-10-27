import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import numpy as np
from matplotlib import pyplot as plt
import os
from activations import *
import pickle
import tqdm
import itertools
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# data = yf.download("QQQ", period='1mo', interval='15m')
# data.to_pickle('stock_data.pkl')
data = pd.read_pickle("stock_data.pkl")


volume = data["Volume"].values
data["Price Change"] = data[
    "Adj Close"
].diff()  ## this looks for the CHANGE in price between the last close and the (predicted) next close

data["SMA_5"] = data["Adj Close"].rolling(window=5).mean()
data["SMA_10"] = data["Adj Close"].rolling(window=10).mean()
# This is for RSI (momentum indicator)
window_length = 14
delta = data["Adj Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window_length).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window_length).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

# Stochastic Oscillator (%K and %D)
high_14 = data["High"].rolling(window=14).max()
low_14 = data["Low"].rolling(window=14).min()
data["%K"] = (data["Adj Close"] - low_14) * 100 / (high_14 - low_14)
data["%D"] = data["%K"].rolling(window=3).mean()

# Price Rate of Change (ROC)
data["ROC"] = data["Adj Close"].pct_change(periods=10) * 100

data = data.dropna()
print(data)

features = data[
    ["Adj Close", "Volume", "SMA_5", "SMA_10", "RSI", "ROC", "%K", "%D"]
].values

max_values = features.max(axis=0)
min_values = features.min(axis=0)
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_features = scaler.fit_transform(features)


def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i : (i + time_step)])
        Y.append(
            data[i + time_step, 0] - data[i, 0]
        )  ## index 4 is the change in price between the last and next adj close price
    return np.array(X), np.array(Y)


time_step = 10  # number of previous 15 min periods for prediction
X_full, Y_full = create_dataset(normalized_features, time_step)

X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], X_full.shape[2])

num_features = X_full.shape[2]
layers = [X_full.shape[1], 100, 200, 1]

split_ratio = 0.8
split_index = int(len(X_full) * split_ratio)

X_train = X_full[:split_index]
Y_train = Y_full[:split_index]

X_test = X_full[split_index:]
Y_test = Y_full[split_index:]

print("Training set shape: ", X_train.shape, Y_train.shape)
print("Test set shape: ", X_test.shape, Y_test.shape)


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


# Train the model
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Make predictions
predicted_price_changes = model.predict(X_test)
predicted_price_changes = scaler.inverse_transform(predicted_price_changes)

plt.figure(figsize=(12, 6))
plt.plot(data["Price Change"], label="Actual Price Changes", color="blue")
plt.plot(predicted_price_changes, label="Predicted Price Changes", color="orange")
plt.title("Actual vs Predicted Stock Price Changes")
plt.xlabel("Time Steps")
plt.ylabel("Price Change")
plt.legend()
plt.grid()
plt.show()
