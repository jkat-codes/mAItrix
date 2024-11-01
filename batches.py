import cupy as cp
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime



data = yf.download("QQQ", period="1mo", interval="15m")
data.to_pickle("stock_data.pkl")
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


time_step = 15  # number of previous 15 min periods for prediction
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

scaler = StandardScaler()
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
        # He initialization for weights
        parameters[f"W{l}"] = np.random.randn(
            layer_dims[l], layer_dims[l - 1]
        ) * np.sqrt(1.0 / layer_dims[l - 1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
        # Add batch norm parameters for hidden layers only
        if l < L - 1:  # No batch norm on output layer
            parameters[f"gamma{l}"] = np.ones((layer_dims[l], 1))
            parameters[f"beta{l}"] = np.zeros((layer_dims[l], 1))
    return parameters


def batch_norm_forward(Z, gamma, beta, eps=1e-8):
    mu = np.mean(Z, axis=1, keepdims=True)
    var = np.var(Z, axis=1, keepdims=True) + eps
    Z_norm = (Z - mu) / np.sqrt(var)
    out = gamma * Z_norm + beta
    cache = (Z_norm, mu, var, gamma, beta)
    return out, cache


def forward_prop(X, parameters, activation, dropout_rate=0.2, training=True):
    forward_cache = {}
    # Calculate number of layers from parameters
    L = len([key for key in parameters.keys() if key.startswith("W")])

    # Reshape input to handle 3D data (batch_size, sequence_length, features)
    m = X.shape[0]
    X_flat = X.reshape(m, -1).T  # Flatten the input appropriately
    forward_cache["A0"] = X_flat

    for l in range(1, L + 1):
        # Linear forward
        Z = parameters[f"W{l}"].dot(forward_cache[f"A{l-1}"]) + parameters[f"b{l}"]

        # Apply batch normalization to hidden layers (only if parameters exist)
        if l < L and training and f"gamma{l}" in parameters:
            Z, bn_cache = batch_norm_forward(
                Z, parameters[f"gamma{l}"], parameters[f"beta{l}"]
            )
            forward_cache[f"bn_cache{l}"] = bn_cache

        if l < L:  # Hidden layers
            if activation == "leaky_relu":
                A = np.where(Z > 0, Z, 0.01 * Z)
            elif activation == "elu":
                A = np.where(Z > 0, Z, 0.01 * (np.exp(Z) - 1))
            forward_cache[f"Z{l}"] = Z
            forward_cache[f"A{l}"] = A
            if training:
                D = np.random.rand(*A.shape) > dropout_rate
                A = np.multiply(A, D) / (1 - dropout_rate)  # Inverted dropout
                forward_cache[f"D{l}"] = D
        else:  # Output layer
            forward_cache[f"A{l}"] = Z

    return forward_cache[f"A{L}"], forward_cache


def batch_norm_backward(dout, cache):
    Z_norm, mu, var, gamma, beta = cache
    m = dout.shape[1]

    dZ_norm = dout * gamma
    dvar = np.sum(dZ_norm * (Z_norm - mu) * -0.5 * (var**-1.5), axis=1, keepdims=True)
    dmu = np.sum(dZ_norm * -1 / np.sqrt(var), axis=1, keepdims=True) + dvar * np.mean(
        -2 * (Z_norm - mu), axis=1, keepdims=True
    )

    dZ = dZ_norm / np.sqrt(var) + dvar * 2 * (Z_norm - mu) / m + dmu / m
    dgamma = np.sum(dout * Z_norm, axis=1, keepdims=True)
    dbeta = np.sum(dout, axis=1, keepdims=True)

    return dZ, dgamma, dbeta


def back_prop(AL, Y, parameters, forward_cache, activation, dropout_rate=0.5):
    grads = {}
    L = len([key for key in parameters.keys() if key.startswith("W")])
    m = AL.shape[1]

    # Output layer
    grads[f"dZ{L}"] = AL - Y.reshape(AL.shape)

    for l in reversed(range(1, L + 1)):
        # Compute gradients for weights and biases
        grads[f"dW{l}"] = 1.0 / m * np.dot(grads[f"dZ{l}"], forward_cache[f"A{l-1}"].T)
        grads[f"db{l}"] = 1.0 / m * np.sum(grads[f"dZ{l}"], axis=1, keepdims=True)

        if l > 1:  # Hidden layers
            # Compute dZ for previous layer
            dA = np.dot(parameters[f"W{l}"].T, grads[f"dZ{l}"])

            if activation == "leaky_relu":
                dZ = dA * np.where(forward_cache[f"Z{l-1}"] > 0, 1, 0.01)
            elif activation == "elu":
                dZ = dA * np.where(
                    forward_cache[f"Z{l-1}"] > 0,
                    1,
                    0.01 * np.exp(forward_cache[f"Z{l-1}"]),
                )

            # Apply batch norm gradients
            if f"bn_cache{l-1}" in forward_cache:
                dZ, dgamma, dbeta = batch_norm_backward(
                    dZ, forward_cache[f"bn_cache{l-1}"]
                )
                grads[f"dgamma{l-1}"] = dgamma
                grads[f"dbeta{l-1}"] = dbeta

            # Apply dropout
            if f"D{l-1}" in forward_cache:
                dZ *= forward_cache[f"D{l-1}"] / (1 - dropout_rate)

            grads[f"dZ{l-1}"] = dZ

    return grads


def update_parameters(parameters, grads, learning_rate, iteration, decay_rate=0.01):
    # Implement learning rate decay
    # learning_rate = learning_rate / (1 + decay_rate * iteration)
    learning_rate = learning_rate

    L = len([key for key in parameters.keys() if key.startswith("W")])

    for l in range(L):
        # Update weights and biases
        parameters[f"W{l+1}"] -= learning_rate * grads[f"dW{l+1}"]
        parameters[f"b{l+1}"] -= learning_rate * grads[f"db{l+1}"]

        # Update batch norm parameters
        if f"gamma{l+1}" in parameters:
            parameters[f"gamma{l+1}"] -= learning_rate * grads[f"dgamma{l+1}"]
            parameters[f"beta{l+1}"] -= learning_rate * grads[f"dbeta{l+1}"]

    return parameters, learning_rate


def early_stopping_check(costs, patience=10, min_delta=1e-4):
    if len(costs) < patience:
        return False

    recent_costs = costs[-patience:]
    if all(
        recent_costs[i] <= recent_costs[i + 1] for i in range(len(recent_costs) - 1)
    ):
        return True

    return False


# predictions function
def predict(X, parameters, activation):
    AL, _ = forward_prop(X, parameters, activation, dropout_rate=0.0, training=False)
    return scaler.inverse_transform(AL.T).flatten()


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
    plt.show()


def model(
    X,
    Y,
    layer_dims,
    learning_rate=0.005,
    activation="leaky_relu",
    num_iterations=5000,
    dropout_rate=0.2,
    batch_size=32,
    training=True,
):
    costs = []
    maes = []

    parameters = initialize_params(layer_dims)

    best_cost = float("inf")
    best_parameters = None
    patience = 2

    # Mini-batch training
    for i in tqdm.tqdm(range(num_iterations)):
        # Generate mini-batches
        indices = np.random.permutation(X.shape[0])
        for j in range(0, X.shape[0], batch_size):
            batch_indices = indices[j : j + batch_size]
            X_batch = X[batch_indices]
            Y_batch = Y[batch_indices]

            AL, forward_cache = forward_prop(
                X_batch,
                parameters,
                activation,
                dropout_rate=dropout_rate,
                training=training,
            )

            grads = back_prop(
                AL,
                Y_batch,
                parameters,
                forward_cache,
                activation,
                dropout_rate=dropout_rate,
            )

            parameters, curr_learning_rate = update_parameters(
                parameters, grads, learning_rate, i
            )

        # Calculate cost for full dataset
        AL_full, _ = forward_prop(
            X, parameters, activation, dropout_rate=0, training=False
        )
        cost = np.mean((AL_full - Y.reshape(AL_full.shape)) ** 2)  # MSE
        mae = mean_absolute_error(Y.flatten(), AL_full.flatten())

        costs.append(cost)
        maes.append(mae)

        if cost < best_cost:
            best_cost = cost
            best_parameters = parameters.copy()

        if early_stopping_check(costs):
            print("Early stopping triggered.")
            break

        if i % (num_iterations // 10) == 0:
            print(f"\nIteration: {i}")
            print(f"Cost: {cost:.6f}")
            print(f"MAE: {mae:.6f}")
            print(f"Learning rate: {curr_learning_rate:.6f}")

    return best_parameters


def predict_market_price(
    input_data, model_parameters, scaler_X, scaler_y, activation="leaky_relu"
):
    """
    Predict market price using the trained neural network

    Parameters:
    - input_data: Numpy array of recent market data (same format as training data)
    - model_parameters: Trained neural network parameters
    - scaler_X: Input feature scaler
    - scaler_y: Target price change scaler
    - activation: Activation function used during training

    Returns:
    - Current price
    - Predicted price change
    - Predicted price
    """
    # Prepare input data (reshape and scale)
    input_data_flattened = input_data.reshape(1, -1)
    input_data_scaled = scaler_X.transform(input_data_flattened)
    input_data_reshaped = input_data_scaled.reshape(
        1, input_data.shape[0], input_data.shape[1]
    )

    # Predict price change
    predicted_change_scaled, _ = forward_prop(
        input_data_reshaped,
        model_parameters,
        activation,
        dropout_rate=0.0,
        training=False,
    )

    # Inverse transform the predicted change
    predicted_change = scaler_y.inverse_transform(predicted_change_scaled.T).flatten()[
        0
    ]

    # Get current price (last price in the input data)
    current_price = input_data[-1, 0]  # Assuming first column is price

    # Calculate predicted price
    predicted_price = current_price + predicted_change

    return current_price, predicted_change, predicted_price


import itertools
from datetime import datetime
import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def create_parameter_grid():
    """Define hyperparameter search space"""
    param_grid = {
        "learning_rate": [0.01],
        "dropout_rate": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "activation": ["elu", "leaky_relu"],
        "num_iterations": [1000, 2000],
        "batch_size": [32, 64],
        "layer_dims": [
            [15 * 8, 32, 1],
            [15 * 8, 22, 1], 
            [15 * 8, 12, 1]
        ],
    }
    return param_grid


def evaluate_model(actual, predicted):
    """Calculate various performance metrics"""
    metrics = {
        "mae": mean_absolute_error(actual, predicted),
        "mse": mean_squared_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
        # Directional accuracy (how often the model predicts the correct direction)
        "directional_accuracy": np.mean(np.sign(predicted) == np.sign(actual)),
    }
    return metrics


def save_model_results(params, metrics, predictions, actuals, run_id):
    """Save model results and plots"""
    # Create directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"model_runs_daily/{timestamp}_{run_id}"
    os.makedirs(run_dir, exist_ok=True)

    # Save parameters and metrics
    results = {"parameters": params, "metrics": metrics}
    with open(f"{run_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Save predictions
    np.save(f"{run_dir}/predictions.npy", predictions)
    np.save(f"{run_dir}/actuals.npy", actuals)

    # Generate and save plots
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label="Actual", color="blue", alpha=0.7)
    plt.plot(predictions, label="Predicted", color="red", alpha=0.7)
    plt.title(f"Model Predictions (Run {run_id})")
    plt.xlabel("Time")
    plt.ylabel("Price Difference")
    plt.legend()

    # Add metrics and parameters to plot
    info_text = f"MAE: {metrics['mae']:.4f}\n"
    info_text += f"RMSE: {metrics['rmse']:.4f}\n"
    info_text += f"R²: {metrics['r2']:.4f}\n"
    info_text += f"Dir. Acc: {metrics['directional_accuracy']:.2%}\n"
    info_text += f"LR: {params['learning_rate']}\n"
    info_text += f"DR: {params['dropout_rate']}"

    plt.figtext(
        0.02,
        0.98,
        info_text,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(f"{run_dir}/predictions_plot.png")
    plt.close()

    return run_dir


def run_hyperparameter_search(X_train, Y_train, X_test, Y_test):
    """Run automated hyperparameter search"""
    param_grid = create_parameter_grid()

    # Generate all combinations of parameters
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())
    ]

    best_metrics = {"rmse": float("inf")}
    best_params = None
    all_results = []

    for i, params in enumerate(param_combinations):
        print(f"\nTesting combination {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")

        # Train model
        parameters = model(
            X_train,
            Y_train,
            params["layer_dims"],
            learning_rate=params["learning_rate"],
            num_iterations=params["num_iterations"],
            dropout_rate=params["dropout_rate"],
            batch_size=params["batch_size"],
            activation=params["activation"],
        )

        # Generate predictions
        predicted_changes = predict(X_test, parameters, params["activation"])
        actual_changes = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

        # Evaluate model
        metrics = evaluate_model(actual_changes, predicted_changes)

        # Save results
        run_dir = save_model_results(
            params, metrics, predicted_changes, actual_changes, f"run_{i+1}"
        )

        # Track best model
        if metrics["rmse"] < best_metrics["rmse"]:
            best_metrics = metrics
            best_params = params
            # Save best model parameters
            with open(f"{run_dir}/model_parameters.pkl", "wb") as f:
                pickle.dump(parameters, f)

        result = {"params": params, "metrics": metrics, "run_dir": run_dir}
        all_results.append(result)

        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"Directional Accuracy: {metrics['directional_accuracy']:.2%}")

    # Generate summary report
    generate_summary_report(all_results, best_params, best_metrics)

    return best_params, best_metrics, all_results


def generate_summary_report(all_results, best_params, best_metrics):
    """Generate a summary report of all runs"""
    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"model_runs_daily/summary_report_{report_time}.html"

    html_content = """
    <html>
    <head>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid black; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .best-row { background-color: #90EE90; }
        </style>
    </head>
    <body>
        <h1>Hyperparameter Search Results</h1>
        <h2>Best Model Configuration:</h2>
        <pre>
    """

    html_content += f"Parameters: {json.dumps(best_params, indent=4)}\n"
    html_content += f"Metrics: {json.dumps(best_metrics, indent=4)}\n"
    html_content += """
        </pre>
        <h2>All Results:</h2>
        <table>
            <tr>
                <th>Run</th>
                <th>Learning Rate</th>
                <th>Dropout Rate</th>
                <th>Activation</th>
                <th>Iterations</th>
                <th>Batch Size</th>
                <th>RMSE</th>
                <th>Dir. Accuracy</th>
                <th>Plot Link</th>
            </tr>
    """

    for i, result in enumerate(all_results):
        is_best = result["params"] == best_params
        row_class = "best-row" if is_best else ""
        html_content += f"""
            <tr class="{row_class}">
                <td>{i+1}</td>
                <td>{result['params']['learning_rate']}</td>
                <td>{result['params']['dropout_rate']}</td>
                <td>{result['params']['activation']}</td>
                <td>{result['params']['num_iterations']}</td>
                <td>{result['params']['batch_size']}</td>
                <td>{result['metrics']['rmse']:.4f}</td>
                <td>{result['metrics']['directional_accuracy']:.2%}</td>
                <td><a href="../{result['run_dir']}/predictions_plot.png">View Plot</a></td>
            </tr>
        """

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(report_path, "w") as f:
        f.write(html_content)

    print(f"\nSummary report generated: {report_path}")


# Usage example
if __name__ == "__main__":
    # Create directory for model runs if it doesn't exist
    os.makedirs("model_runs_daily", exist_ok=True)

    best_params, best_metrics, all_results = run_hyperparameter_search(
        X_train, Y_train, X_test, Y_test
    )

    print("\nBest parameters found:")
    print(json.dumps(best_params, indent=4))
    print("\nBest metrics achieved:")
    print(json.dumps(best_metrics, indent=4))

# Example usage with enhanced hyperparameters:
# if __name__ == "__main__":
#     learning_rate = 0.001  # Lower initial learning rate
#     num_iterations = 2000  # More iterations
#     dropout_rate = 0.7
#     activation = "leaky_relu"
#     batch_size = 32

#     input_nums = 15 * 8

#     # Deeper network with more neurons
#     layers = [
#         int(input_nums),
#         32,  # Fourth hidden layer
#         1,  # Output layer
#     ]

#     hyperparameters = {
#         "learning_rate": learning_rate,
#         "activation": activation,
#         "num_iterations": num_iterations,
#         "dropout_rate": dropout_rate,
#         "batch_size": batch_size,
#     }

#     parameters = model(
#         X_train,
#         Y_train,
#         layers,
#         learning_rate=learning_rate,
#         num_iterations=num_iterations,
#         dropout_rate=dropout_rate,
#         batch_size=batch_size,
#         training=True,
#         activation=activation,
#     )

#     predicted_train = predict(X_train, parameters, activation)
#     actual_train = scaler.inverse_transform(Y_train.reshape(-1, 1)).flatten()

#     predicted_changes = predict(X_test, parameters, activation=activation)
#     actual_changes = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()

#     plot_predictions(actual_changes, predicted_changes, hyperparameters)
#     plot_predictions(actual_train, predicted_train, hyperparameters)
