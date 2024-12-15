import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 下载股票数据
def fetch_stock_data(ticker, start_date, end_date):
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError("No data found. Please check the ticker or date range.")
    print("Data downloaded successfully!")
    return data

# 准备数据用于 LSTM 模型
def prepare_data(data, prediction_days=60):
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, scaler

# 构建 LSTM 模型
def build_lstm_model():
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
        LSTM(units=50, return_sequences=False),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 预测未来股价
def predict_future(data, scaler, model, prediction_days=60, future_days=30):
    last_60_days = data['Close'].values[-prediction_days:].reshape(-1, 1)
    last_60_scaled = scaler.transform(last_60_days)

    X_test = np.array([last_60_scaled])
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    predictions = []
    for _ in range(future_days):
        pred_price = model.predict(X_test)[0, 0]
        predictions.append(pred_price)
        X_test = np.append(X_test[0, 1:], [[pred_price]], axis=0).reshape((1, prediction_days, 1))

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 生成未来日期索引
def get_future_dates(last_date, future_days):
    """生成从历史数据最后日期开始的未来日期索引"""
    return pd.date_range(start=last_date, periods=future_days + 1, freq='B')[1:]

# 主程序
if __name__ == "__main__":
    try:
        print("Program started!")
        stock_ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
        start_date = input("Enter start date (YYYY-MM-DD): ")
        end_date = input("Enter end date (YYYY-MM-DD): ")

        print("Fetching stock data...")
        stock_data = fetch_stock_data(stock_ticker, start_date, end_date)

        print(stock_data.head())

        # 准备训练数据
        X_train, y_train, scaler = prepare_data(stock_data)

        # 构建和训练模型
        print("Building and training the LSTM model...")
        lstm_model = build_lstm_model()
        lstm_model.fit(X_train, y_train, batch_size=32, epochs=10)
        print("Model training completed!")

        # 预测未来股价
        print("Predicting future stock prices...")
        future_prices = predict_future(stock_data, scaler, lstm_model)

        # 生成未来日期
        future_dates = get_future_dates(stock_data.index[-1], len(future_prices))

        # 绘制结果
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data.index, stock_data['Close'], label="Historical Prices", color='blue')
        plt.plot(future_dates, future_prices, label="Predicted Prices", color='orange')
        plt.title(f"{stock_ticker} Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Stock Price (USD)")
        plt.legend()
        plt.grid()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


