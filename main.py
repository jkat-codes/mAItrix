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
from sklearn.metrics import mean_absolute_error

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


def initialize_params(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.rand(
            layer_dims[l], layer_dims[l - 1]
        ) * np.sqrt(
            2.0 / layer_dims[l - 1]
        )  ## this coefficient is He initialization
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    return parameters


def forward_prop(X, parameters, activation, dropout_rate=0.2, training=True):
    forward_cache = {}
    L = len(parameters) // 2
    m = X.shape[0]  # number of samples

    # Flatten the input for the first layer
    forward_cache["A0"] = X.reshape(
        m, -1
    ).T  # Shape: (number of features, number of samples)

    for l in range(1, L + 1):  # Loop through layers
        # Linear transformation
        Z = parameters["W" + str(l)].dot(forward_cache["A" + str(l - 1)]) + parameters[
            "b" + str(l)
        ].reshape(-1, 1)

        if l < L:  # For hidden layers
            if activation == "relu":
                forward_cache["A" + str(l)] = np.maximum(0, Z)
            elif activation == "leaky_relu":
                forward_cache["A" + str(l)] = np.where(
                    Z > 0, Z, 0.01 * Z
                )  # Example for Leaky ReLU
            elif activation == "tanh":
                forward_cache["A" + str(l)] = np.tanh(Z)

            if training:
                D = np.random.randn(*forward_cache["A" + str(l)].shape) < dropout_rate
                forward_cache["A" + str(l)] = (
                    np.multiply(forward_cache["A" + str(l)], D) / dropout_rate
                )
                forward_cache["D" + str(l)] = D

        else:  # Output layer
            forward_cache["A" + str(l)] = Z

    # Output layer (no activation function for regression)
    forward_cache["A" + str(L)] = Z  # Use Z directly for output layer

    return forward_cache["A" + str(L)], forward_cache


def back_prop(AL, Y, parameters, forward_cache, activation, dropout_rate=0.5):
    grads = {}
    L = len(parameters) // 2  ## we divide by 2 because we have W and b
    m = AL.shape[1]

    grads["dZ" + str(L)] = AL - Y.reshape(AL.shape)
    grads["dW" + str(L)] = (
        1.0 / m * np.dot(grads["dZ" + str(L)], forward_cache["A" + str(L - 1)].T)
    )
    grads["db" + str(L)] = 1.0 / m * np.sum(grads["dZ" + str(L)], axis=1, keepdims=True)
    for l in reversed(range(1, L)):
        if activation == "relu":
            grads["dZ" + str(l)] = np.dot(
                parameters["W" + str(l + 1)].T, grads["dZ" + str(l + 1)]
            ) * (
                forward_cache["A" + str(l)] > 0
            )  ## this is the derivative of relu
        elif activation == "tanh":
            grads["dZ" + str(l)] = np.dot(
                parameters["W" + str(l + 1)].T, grads["dZ" + str(l + 1)]
            ) * deriv_tanh(forward_cache["A" + str(l)])

        grads["dZ" + str(l)] *= forward_cache["D" + str(l)] / dropout_rate

        grads["dW" + str(l)] = (
            1.0 / m * np.dot(grads["dZ" + str(l)], forward_cache["A" + str(l - 1)].T)
        )
        grads["db" + str(l)] = (
            1.0 / m * np.sum(grads["dZ" + str(l)], axis=1, keepdims=True)
        )

    return grads


# update parameters so the network can learn
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2  ## we divide by 2 because we have W and b

    for l in range(L):
        parameters["W" + str(l + 1)] = (
            parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        )
        parameters["b" + str(l + 1)] = (
            parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
        )

    return parameters


def model(
    X,
    Y,
    layer_dims,
    learning_rate=0.005,
    activation="relu",
    num_iterations=5000,
    dropout_rate=0.2,
    training=True,
):
    costs = []
    maes = []

    parameters = initialize_params(layer_dims)

    for i in tqdm.tqdm(range(0, num_iterations)):
        AL, forward_cache = forward_prop(
            X, parameters, activation, dropout_rate=dropout_rate, training=True
        )

        cost = np.mean((AL - Y.reshape(AL.shape)) ** 2)  # MSE
        costs.append(cost)

        mae = mean_absolute_error(Y.flatten(), AL.flatten())
        maes.append(mae)

        grads = back_prop(
            AL, Y, parameters, forward_cache, activation, dropout_rate=dropout_rate
        )
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % (num_iterations / 10) == 0:
            print(f"\n Iteration: {i} \t Cost: {cost:.5f} \t Mae: {mae:.5f}")

    return parameters


def save_params(parameters, filename="trained_params.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((parameters), f)
    print(f"Parameters saved to {filename}")


def load_saved_params(filename="trained_params.pkl"):
    with open(filename, "rb") as f:
        parameters = pickle.load(f)
    print(f"Parameters loaded from {filename}")
    return parameters


def plot_predictions(actual, predicted):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Price Differnce", color="blue", alpha=0.7)
    plt.plot(predicted, label="Predicted Price Difference", color="Red", alpha=0.7)
    plt.title("Actual vs. Predicted Price Differences")
    plt.xlabel("Time")
    plt.ylabel("Price Difference")
    plt.legend()
    plt.grid()
    plt.show()


# predictions function
def predict(X, parameters, activation):
    AL, _ = forward_prop(X, parameters, activation, dropout_rate=0.0, training=False)
    return AL.flatten()


if __name__ == "__main__":
    ## Hyperparameters ##
    learning_rate = 0.01
    num_iterations = 100000
    dropout_rate = 0.4

    layers = [80, 5, 1]

    parameters = model(
        X_train,
        Y_train,
        layers,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        dropout_rate=dropout_rate,
        training=True,
    )

    predicted_changes = predict(X_test, parameters, activation="leaky_relu")
    actual_changes = Y_test.flatten()

    plot_predictions(actual_changes, predicted_changes)
