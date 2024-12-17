# Solar-Flare-Analysis-Using-Numerical-Data
My project is about analysing future solar flare with the help of machine learning model created by me.
[SOLAR_FLARE_TEAM.docx](https://github.com/user-attachments/files/18163172/SOLAR_FLARE_TEAM.docx)
[hessi.solar.flare.UP_To_2018.csv](https://github.com/user-attachments/files/18163176/hessi.solar.flare.UP_To_2018.csv)

SOURCE CODE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Load and preprocess data
file_path = 'hessi.solar.flare.UP_To_2018.csv'
data = pd.read_csv(file_path)

data = data.dropna()

time_column = 'peak'
duration_column = 'duration.s'
peak_count_column = 'peak.c/s'
total_count_column = 'total.counts'

data[time_column] = pd.to_datetime(data[time_column])

scalers = {
    'duration': MinMaxScaler(feature_range=(0, 1)),
    'peak_count': MinMaxScaler(feature_range=(0, 1)),
    'total_count': MinMaxScaler(feature_range=(0, 1))
}

data['normalized_duration'] = scalers['duration'].fit_transform(data[duration_column].values.reshape(-1, 1))
data['normalized_peak_count'] = scalers['peak_count'].fit_transform(data[peak_count_column].values.reshape(-1, 1))
data['normalized_total_count'] = scalers['total_count'].fit_transform(data[total_count_column].values.reshape(-1, 1))

features = np.column_stack([
    data['normalized_duration'].values,
    data['normalized_peak_count'].values,
    data['normalized_total_count'].values
])

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), :]
        X.append(a)
        Y.append(dataset[i + time_step, :])
    return np.array(X), np.array(Y)

time_step = 10
X, y = create_dataset(features, time_step)
X = X.reshape(X.shape[0], X.shape[1], 3)  # Three features

# Split data into training and validation sets
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# Build and train the model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(time_step, 3)))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(3))  
model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val), verbose=1)

# Evaluate the model
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print(f'Validation MSE: {mse}')
print(f'Validation MAE: {mae}')

def predict_future(model, data, time_step, n_days):
    future_predictions = []
    last_sequence = data[-time_step:]
    
    for _ in range(n_days):
        next_pred = model.predict(last_sequence.reshape(1, time_step, 3))
        future_predictions.append(next_pred[0])
        last_sequence = np.append(last_sequence[1:], next_pred, axis=0)
    
    return future_predictions

n_days = 30
future_data = predict_future(model, features, time_step, n_days)
future_data = np.array(future_data)

future_duration = scalers['duration'].inverse_transform(future_data[:, 0].reshape(-1, 1))
future_peak_count = scalers['peak_count'].inverse_transform(future_data[:, 1].reshape(-1, 1))
future_total_count = scalers['total_count'].inverse_transform(future_data[:, 2].reshape(-1, 1))

last_date = data[time_column].iloc[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]

plt.figure(figsize=(12, 6))

plt.plot(data[time_column], data[peak_count_column], label='Historical Peak Count')
plt.plot(future_dates, future_peak_count, label='Predicted Future Peak Count', linestyle='dashed')

plt.xlabel('Date')
plt.ylabel('Peak Count')
plt.title('Solar Flare Prediction')
plt.legend()
plt.show()

print("Future Predictions:")
for date, duration, peak_count, total_count in zip(future_dates, future_duration, future_peak_count, future_total_count):
    print(f"Date: {date.strftime('%Y-%m-%d')}, Duration: {duration[0]:.2f}, Peak Count: {peak_count[0]:.2f}, Total Count: {total_count[0]:.2f}")
