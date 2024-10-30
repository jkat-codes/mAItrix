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
from datetime import datetime

# data = yf.download("QQQ", period='1mo', interval='15m')
# data.to_pickle('stock_data.pkl')
data = pd.read_pickle("stock_data.pkl")


volume = data["Volume"].values
data["Price Change"] = data[
    "Adj Close"
].diff()  ## this looks for the CHANGE in price between the last close and the (predicted) next close

price_change = data["Price Change"].values

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

# MACD
exp1 = data["Adj Close"].ewm(span=12, adjust=False).mean()
exp2 = data["Adj Close"].ewm(span=26, adjust=False).mean()
data["MACD"] = exp1 - exp2
data["Signal Line"] = data["MACD"].ewm(span=9, adjust=False).mean()

# Volume Indicators
data["Volume_SMA"] = data["Volume"].rolling(window=10).mean()
data["Volume_ratio"] = data["Volume"].iloc[:, 0].div(data["Volume_SMA"])


# Momentum indicator
data["momentum"] = data["Adj Close"] - data["Adj Close"].shift(4)

data = data.dropna()


features = data[
    ["Adj Close", "Volume", "SMA_5", "SMA_10", "RSI", "momentum", "MACD", "Signal Line"]
].values


def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i : (i + time_step)])
        Y.append(
            price_change[i + time_step + 1]
        )  ## index 4 is the change in price between the last and next adj close price
    return np.array(X), np.array(Y)


time_step = 10  # number of previous 15 min periods for prediction
X_full, Y_full = create_dataset(features, time_step)


X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], X_full.shape[2])

num_features = X_full.shape[2] * X_full.shape[1]  ## number of input layer nodes

split_ratio = 0.8
split_index = int(len(X_full) * split_ratio)

X_train = X_full[:split_index]
Y_train = Y_full[:split_index]

X_test = X_full[split_index:]
Y_test = Y_full[split_index:]

X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_test_flattened = X_test.reshape(X_test.shape[0], -1)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train_flattened)
X_test_scaled = scaler.transform(X_test_flattened)

X_train = X_train_scaled.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

Y_train = scaler.fit_transform(Y_train)
Y_test = scaler.transform(Y_test)

print("Training set shape: ", X_train.shape, Y_train.shape)
print("Test set shape: ", X_test.shape, Y_test.shape)


def initialize_params(layer_dims):
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f"W{l}"] = np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        ) * np.sqrt(1.0 / layer_dims[l - 1])
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


def back_prop(
    AL, Y, parameters, forward_cache, activation, dropout_rate=0.5, training=True
):
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

        elif activation == "leaky_relu":
            grads["dZ" + str(l)] = np.dot(
                parameters["W" + str(l + 1)].T, grads["dZ" + str(l + 1)]
            ) * deriv_leaky_relu(forward_cache["A" + str(l)])

        if training:
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

    hyperparameters = {
        "learning_rate": learning_rate,
        "activation": activation,
        "num_iterations": num_iterations,
        "dropout_rate": dropout_rate,
    }

    parameters = initialize_params(layer_dims)

    for i in tqdm.tqdm(range(0, num_iterations)):
        AL, forward_cache = forward_prop(
            X, parameters, activation, dropout_rate=dropout_rate, training=training
        )

        cost = np.mean((AL - Y.reshape(AL.shape)) ** 2)  # MSE
        costs.append(cost)

        mae = mean_absolute_error(Y.flatten(), AL.flatten())
        maes.append(mae)

        grads = back_prop(
            AL,
            Y,
            parameters,
            forward_cache,
            activation,
            dropout_rate=dropout_rate,
            training=training,
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


def plot_predictions(actual, predicted, hypers):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Price Differnce", color="blue", alpha=0.7)
    plt.plot(predicted, label="Predicted Price Difference", color="Red", alpha=0.7)
    plt.title("Actual vs. Predicted Price Differences")
    plt.xlabel("Time")
    plt.ylabel("Price Difference")
    plt.legend()
    plt.figtext(
        0.8,
        0.8,
        f'Learning Rate: {hypers["learning_rate"]}\nActivaton: {hypers["activation"]}\nIterations: {hypers["num_iterations"]}\nDropout Rate: {hypers["dropout_rate"]}',
    )
    plt.grid()
    plt.savefig(f"graphs/{str(datetime.now())}.png", bbox_inches="tight")


# predictions function
def predict(X, parameters, activation):
    AL, _ = forward_prop(X, parameters, activation, dropout_rate=0.0, training=False)
    return scaler.inverse_transform(AL.T).flatten()


def find_best_params(learning_rates, dropout_rates):
    layers = [int(num_features), 32, 1]

    for rate, drate in itertools.product(learning_rates, dropout_rates):
        hyperparameters = {
            "learning_rate": rate,
            "activation": "leaky_relu",
            "num_iterations": 5000,
            "dropout_rate": drate,
        }

        parameters = model(
            X_train,
            Y_train,
            layers,
            learning_rate=rate,
            num_iterations=5000,
            dropout_rate=drate,
            training=True,
            activation="leaky_relu",
        )
        predicted_changes = predict(X_test, parameters, activation="leaky_relu")
        actual_changes = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

        plot_predictions(actual_changes, predicted_changes, hyperparameters)


def train_base():
    ## Hyperparameters ##
    learning_rate = 0.00001
    num_iterations = 5000
    dropout_rate = 0.4
    activation = "leaky_relu"

    hyperparameters = {
        "learning_rate": learning_rate,
        "activation": activation,
        "num_iterations": num_iterations,
        "dropout_rate": dropout_rate,
    }

    layers = [int(num_features), 32, 1]

    parameters = model(
        X_train,
        Y_train,
        layers,
        learning_rate=learning_rate,
        num_iterations=num_iterations,
        dropout_rate=dropout_rate,
        training=True,
        activation=activation,
    )

    predicted_changes = predict(X_test, parameters, activation=activation)
    actual_changes = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

    plot_predictions(actual_changes, predicted_changes, hyperparameters)


if __name__ == "__main__":
    learning_rates = [0.00001, 0.0001, 0.001]
    dropout_rate = [0.6, 0.5, 0.4, 0.3]
    find_best_params(learning_rates, dropout_rate)
