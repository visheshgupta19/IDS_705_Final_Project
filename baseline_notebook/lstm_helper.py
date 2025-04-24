import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lstm_helper(data_train, data_test):

    # data_train = pd.read_csv("data_prepared/training_dataset.csv")

    features = ["Date", "Close", "High", "Low", "Open", "Volume", "Pct_Change"]
    target = "Pct_Change"

    X_train = data_train[features].values
    y_train = data_train[target].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_scaled = np.reshape(
        X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    )

    # Note: the default tanh activation works best compared to the other tested functions: sigmoid, relu
    model = Sequential(
        [
            LSTM(
                units=128,
                input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
                return_sequences=True,
            ),
            Dropout(0.2),
            LSTM(units=64, return_sequences=False),
            Dropout(0.2),
            Dense(32),
            Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)

    # data_test = pd.read_csv("data_prepared/testing_dataset.csv")

    X_test = data_test[features].values
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = np.reshape(
        X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
    )

    y_pred = model.predict(X_test_scaled)

    return y_pred
