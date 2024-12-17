import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

file_path = 'hkd_exchange_rate.csv'  # Replace with your file path
data = pd.read_csv(file_path)

data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data.dropna(subset=['Close'], inplace=True)

exchange_rate = data['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
exchange_rate_scaled = scaler.fit_transform(exchange_rate)


train_size = int(len(exchange_rate_scaled) * 0.8)
train_data = exchange_rate_scaled[:train_size]
test_data = exchange_rate_scaled[train_size:]

print(f"train_data size: {train_data.shape}")
print(f"test_data size: {test_data.shape}")

def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])  # Append time_step days of data
        Y.append(dataset[i + time_step, 0])  # Predict the next day
    return np.array(X), np.array(Y)

time_step = 10  # Use the past 10 days to predict the next day (instead of 60)

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print(f"X_train size: {X_train.shape}")
print(f"y_train size: {y_train.shape}")
print(f"X_test size: {X_test.shape}")
print(f"y_test size: {y_test.shape}")

if len(X_train.shape) == 1:
    X_train = X_train.reshape(-1, time_step)
if len(X_test.shape) == 1:
    X_test = X_test.reshape(-1, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"X_train shape after reshaping: {X_train.shape}")
print(f"X_test shape after reshaping: {X_test.shape}")

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),  # Add Dropout to avoid overfitting
    LSTM(50, return_sequences=False),
    Dropout(0.2),  # Add Dropout to avoid overfitting
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')


model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=1)


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)


train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train_actual = scaler.inverse_transform([y_train])
y_test_actual = scaler.inverse_transform([y_test])

plt.figure(figsize=(14, 6))

plt.plot(data['Date'], exchange_rate, label='Actual Exchange Rate', color='blue')

train_predict_plot = np.empty_like(exchange_rate)
train_predict_plot[:, :] = np.nan
train_predict_plot[time_step:len(train_predict) + time_step] = train_predict[:len(train_predict)]  # 修正这一行
plt.plot(data['Date'], train_predict_plot, label='Train Predictions', color='green')


test_predict_plot = np.empty_like(exchange_rate)
test_predict_plot[:, :] = np.nan

test_predict_plot[train_size + time_step:train_size + time_step + len(test_predict)] = test_predict[:len(test_predict)]
plt.plot(data['Date'], test_predict_plot, label='Test Predictions', color='red')

plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.title('HKD Exchange Rate Prediction with LSTM')
plt.legend()

plt.xticks(rotation=45)  # Rotate the x-axis labels for better visibility
plt.grid(True)
plt.tight_layout()
plt.show()
