import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from tensorflow.keras import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer


# 데이터 전처리 및 생성 함수
def make_data(data, seq_length, column='Close/Last'):
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date').reset_index(drop=True)

    for col in ['Close/Last', 'Volume', 'Open', 'High', 'Low']:
        if data[col].dtype == 'object':
            data[col] = data[col].str.replace('$', '').astype(float)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[[column]])

    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i + seq_length])
        y.append(scaled_data[i + seq_length])

    return np.array(X), np.array(y), scaler


# LSTM 모델 생성 함수
def create_lstm_model(input_shape, batch_size=None, stateful=False, units=50, dropout_rate=0.1):
    if stateful:
        model = Sequential([
            InputLayer(batch_input_shape=(batch_size, input_shape[0], input_shape[1])),
            LSTM(units, return_sequences=True, stateful=True),
            Dropout(dropout_rate),
            LSTM(units, return_sequences=False, stateful=True),
            Dropout(dropout_rate),
            Dense(1)
        ])
    else:
        model = Sequential([
            InputLayer(input_shape=input_shape),
            LSTM(units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(1)
        ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# LSTM 모델 상태 초기화 함수
def reset_model_states(model):
    for layer in model.layers:
        if isinstance(layer, LSTM) and layer.stateful:
            layer.reset_states()

# LSTM 모델 학습 함수
def train_lstm(model, X, y, batch_size, epochs, stateful=False, validation_data=None):
    if stateful:
        # 학습 데이터 크기를 batch_size의 배수로 조정
        trim_size = len(X) - (len(X) % batch_size)
        X, y = X[:trim_size], y[:trim_size]

        # 검증 데이터도 batch_size의 배수로 조정
        if validation_data:
            X_val, y_val = validation_data
            trim_size_val = len(X_val) - (len(X_val) % batch_size)
            X_val, y_val = X_val[:trim_size_val], y_val[:trim_size_val]
            validation_data = (X_val, y_val)

        # 상태 초기화
        reset_model_states(model)

    # 모델 학습
    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=not stateful,  # Stateful 모델은 셔플 금지
        validation_data=validation_data,
        verbose=1
    )

    if stateful:
        reset_model_states(model)

    return history

# Stateful과 Stateless LSTM 비교 함수
def compare_cases(X, y, batch_size, epochs, input_shape=None):
    # Train/Test Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Stateful LSTM
    print("[ Stateful LSTM ]")
    stateful_model = create_lstm_model(input_shape, batch_size=batch_size, stateful=True)
    stateful_history = train_lstm(stateful_model, X_train, y_train, batch_size=batch_size, epochs=epochs,
                                  stateful=True, validation_data=(X_val, y_val))

    # Stateless LSTM
    print("\n[ Stateless LSTM ]")
    stateless_model = create_lstm_model(input_shape, stateful=False)
    stateless_history = train_lstm(stateless_model, X_train, y_train, batch_size=batch_size, epochs=epochs,
                                   stateful=False, validation_data=(X_val, y_val))

    # Plot Loss Comparison with Subplots
    plt.figure(figsize=(12, 8))

    # Stateful LSTM Loss
    plt.subplot(2, 1, 1)
    plt.plot(stateful_history.history['loss'], label='Train Loss')
    plt.plot(stateful_history.history.get('val_loss', []), label='Val Loss')
    plt.title("Stateful LSTM Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Stateless LSTM Loss
    plt.subplot(2, 1, 2)
    plt.plot(stateless_history.history['loss'], label='Train Loss')
    plt.plot(stateless_history.history.get('val_loss', []), label='Val Loss')
    plt.title("Stateless LSTM Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

    # RMSE Comparison
    trim_size_val = len(X_val) - (len(X_val) % batch_size)
    X_val, y_val = X_val[:trim_size_val], y_val[:trim_size_val]
    stateful_preds = stateful_model.predict(X_val, batch_size=batch_size)
    stateless_preds = stateless_model.predict(X_val)

    stateful_rmse = np.sqrt(mean_squared_error(y_val, stateful_preds))
    stateless_rmse = np.sqrt(mean_squared_error(y_val, stateless_preds))

    print(f"Stateful RMSE: {stateful_rmse}")
    print(f"Stateless RMSE: {stateless_rmse}")

    return {
        "stateful": {"train_loss": stateful_history.history['loss'], "rmse": stateful_rmse},
        "stateless": {"train_loss": stateless_history.history['loss'], "rmse": stateless_rmse}
    }

# 실행 블록
if __name__ == '__main__':
    # 데이터 로드
    raw_data = pd.read_csv('HistoricalData_1732122014814.csv')

    # 데이터 생성
    seq_length = 10
    X, y, scaler = make_data(raw_data, seq_length, column='Close/Last')

    # Stateful과 Stateless 모델 비교
    batch_size = 16
    input_shape = (X.shape[1], X.shape[2])

    results = compare_cases(X, y, batch_size=batch_size, epochs=20, input_shape=input_shape)

    # 결과 출력
    print("\nResults:")
    print(f"Stateful Train Loss: {results['stateful']['train_loss'][-1]:.4f}, RMSE: {results['stateful']['rmse']:.4f}")
    print(f"Stateless Train Loss: {results['stateless']['train_loss'][-1]:.4f}, RMSE: {results['stateless']['rmse']:.4f}")