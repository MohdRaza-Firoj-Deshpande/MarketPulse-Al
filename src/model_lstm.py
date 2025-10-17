import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

def train_lstm(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Close']].values)
    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam',loss='mse')
    model.fit(X, y, epochs=5, batch_size=32)
    return model, scaler
