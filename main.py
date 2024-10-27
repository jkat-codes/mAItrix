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

# data = yf.download("QQQ", period='1mo', interval='15m')
# data.to_pickle('stock_data.pkl')
data = pd.read_pickle('stock_data.pkl')

prices = data["Adj Close"].values
volume = data["Volume"].values

data['SMA_5'] = data["Adj Close"].rolling(window=5).mean()
data['SMA_10'] = data["Adj Close"].rolling(window=10).mean()

data = data.dropna()

features = data[['Adj Close', 'Volume', 'SMA_5', 'SMA_10']].values

max_values = features.max(axis=0)
min_values = features.min(axis=0)
normalized_features = (features - min_values) / (max_values - min_values)

def create_dataset(data, time_step=1): 
    X, Y = [], []
    for i in range(len(data) - time_step): 
        X.append(data[i: (i + time_step)])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


time_step = 10 # number of previous 15 min periods for prediction
X_full, Y_full = create_dataset(normalized_features, time_step)

X_full = X_full.reshape(X_full.shape[0], X_full.shape[1], X_full.shape[2])

num_features = X_full.shape[2]
layers = [time_step, 100, 200, 1]

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
        parameters[f'W{l}'] = np.random.rand(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2./layer_dims[l-1]) ## this coefficient is He initialization
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1)) 

    return parameters


def forward_prop(X, parameters, activation): 
    forward_cache = {}
    L = len(parameters) // 2
    m = X.shape[0]  # number of samples
    
    # Flatten the input for the first layer
    forward_cache['A0'] = X.reshape(m, -1).T  # Shape: (number of features, number of samples)

    for l in range(1, L + 1):  # Loop through layers
        # Linear transformation
        Z = parameters['W' + str(l)].dot(forward_cache['A' + str(l - 1)]) + parameters['b' + str(l)].reshape(-1, 1)
        
        if l < L:  # For hidden layers
            if activation == 'relu':
                forward_cache['A' + str(l)] = np.maximum(0, Z)
            elif activation == 'leaky_relu':
                forward_cache['A' + str(l)] = np.where(Z > 0, Z, 0.01 * Z)  # Example for Leaky ReLU
            elif activation == 'tanh':
                forward_cache['A' + str(l)] = np.tanh(Z)
        else:  # Output layer
            forward_cache['A' + str(l)] = Z  

    # Output layer (no activation function for regression)
    forward_cache['A' + str(L)] = Z  # Use Z directly for output layer

    return forward_cache['A' + str(L)], forward_cache

## test forward prop 
def test_forward_prop(): 
    layers = [X_train.shape[0], 100, 200, Y_train.shape[0]]
    params = initialize_params(layers)  
    aL, forward_cache = forward_prop(X_train, params, 'relu')

    for l in range(len(params) // 2 + 1): 
        print("Shape of A" + str(l) + " :", forward_cache["A" + str(l)].shape)
        

def back_prop(AL, Y, parameters, forward_cache, activation): 
    grads = {}
    L = len(parameters) // 2 ## we divide by 2 because we have W and b
    m = AL.shape[1]

    grads["dZ" + str(L)] = AL - Y.reshape(AL.shape) 
    grads["dW" + str(L)] = 1. / m * np.dot(grads["dZ" + str(L)], forward_cache["A" + str(L - 1)].T)
    grads["db" + str(L)] = 1. / m * np.sum(grads["dZ" + str(L)], axis=1, keepdims=True)

    for l in reversed(range(1, L)): 
        if activation == "relu": 
            grads["dZ" + str(l)] = np.dot(parameters["W" + str(l + 1)].T, grads["dZ" + str(l + 1)]) * (forward_cache["A" + str(l)] > 0) ## this is the derivative of relu
        elif activation == "tanh": 
            grads["dZ" + str(l)] = np.dot(parameters["W" + str(l + 1)].T, grads["dZ" + str(l + 1)]) * deriv_tanh(forward_cache["A" + str(l)]) 
        
        grads["dW" + str(l)] = 1. / m * np.dot(grads["dZ" + str(l)], forward_cache["A" + str(l - 1)].T)
        grads["db" + str(l)] = 1. / m * np.sum(grads["dZ" + str(l)], axis=1, keepdims=True)

    return grads

## test back prop and observe shapes
def test_back_prop(): 
    layers = [X_train.shape[0], 100, 200, Y_train.shape[0]]
    params = initialize_params(layers)  
    aL, forward_cache = forward_prop(X_train, params, 'relu')

    for l in range(len(params) // 2 + 1): 
        print("Shape of A" + str(l) + " :", forward_cache["A" + str(l)].shape)

    grads = back_prop(forward_cache["A" + str(3)], Y_train, params, forward_cache, 'relu')

    print('\n')

    for l in reversed(range(1, len(grads) // 3 + 1)): 
        print("Shape of dZ" + str(l) + " :", grads['dZ' + str(l)].shape)
        print("Shape of dW" + str(l) + " :", grads['dW' + str(l)].shape)
        print("Shape of dB" + str(l) + " :", grads['db' + str(l)].shape, "\n")

# update parameters so the network can learn
def update_parameters(parameters, grads, learning_rate): 
    L = len(parameters) // 2 ## we divide by 2 because we have W and b

    for l in range(L): 
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def model(X, Y, layer_dims, learning_rate=0.005, activation='relu', num_iterations=5000): 
    costs = []

    parameters = initialize_params(layer_dims)

    for i in tqdm.tqdm(range(0, num_iterations)): 
        AL, forward_cache = forward_prop(X, parameters, activation)

        cost = np.mean((AL - Y.reshape(AL.shape)) ** 2)  # MSE

        grads = back_prop(AL, Y, parameters, forward_cache, activation)
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % (num_iterations / 10) == 0: 
            print(f"\n Iteration: {i} \t Cost: {cost:.5f}")

    return parameters

# predictions function
def predict(X, parameters, activation): 
    AL, _ = forward_prop(X, parameters, activation)
    return AL

def save_params(parameters, filename="trained_params.pkl"): 
    with open(filename, 'wb') as f: 
        pickle.dump((parameters), f)
    print(f'Parameters saved to {filename}')

def load_saved_params(filename='trained_params.pkl'): 
    with open(filename, 'rb') as f: 
        parameters = pickle.load(f)
    print(f'Parameters loaded from {filename}')
    return parameters

import numpy as np
import pandas as pd

def backtest_strategy(predictions, actual_prices, initial_capital=1000, buy_threshold=0.01, sell_threshold=0.01):
    # Initialize variables
    cash = initial_capital
    shares = 0
    portfolio_value = []
    
    for i in range(len(predictions)):
        # Get predicted and actual prices
        predicted_price = predictions[i]
        actual_price = actual_prices[i]
        
        # Decision making
        if predicted_price > actual_price * (1 + buy_threshold):
            # Buy signal
            shares_to_buy = cash // actual_price
            cash -= shares_to_buy * actual_price
            shares += shares_to_buy
        
        elif predicted_price < actual_price * (1 - sell_threshold):
            # Sell signal
            cash += shares * actual_price
            shares = 0
        
        # Calculate portfolio value
        total_value = cash + shares * actual_price
        portfolio_value.append(total_value)
    
    return portfolio_value


if __name__ == "__main__": 

    ## Hyperparameters ## 
    learning_rate = 0.019
    num_iterations = 100000

    layers = [40, 40, 1]

    parameters = model(X_train, Y_train, layers, learning_rate=learning_rate, num_iterations=num_iterations)

    last_sequence = normalized_features[-time_step:]  # Last input sequence
    last_sequence = last_sequence.reshape(1, time_step, num_features)
    predicted_prices, _ = forward_prop(X_test, parameters, 'leaky_relu')
    predicted_prices = predicted_prices.flatten()  # Flatten the output if it's a 2D array

    # Actual prices
    actual_prices = Y_test  # Y_test contains the actual close prices

    # Calculate evaluation metrics
    mae = np.mean(np.abs(predicted_prices - actual_prices))
    mse = np.mean((predicted_prices - actual_prices) ** 2)
    rmse = np.sqrt(mse)

    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

    # Visualize the results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label='Actual Prices', color='blue', linewidth=2)
    plt.plot(predicted_prices, label='Predicted Prices', color='red', alpha=0.7)
    plt.title('Actual vs Predicted Close Prices')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    portfolio_values = backtest_strategy(predicted_prices, actual_prices)

    # Convert to a DataFrame for analysis
    results_df = pd.DataFrame({
        'Portfolio Value': portfolio_values,
    })

    # Plot the portfolio value over time
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['Portfolio Value'], label='Portfolio Value', color='green')
    plt.title('Backtest Portfolio Value Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid()
    plt.show()