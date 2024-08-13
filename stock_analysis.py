import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


def get_stock_data(company_name):
    # Create a directory for storing CSV files if it doesn't exist
    csv_dir = 'stock_data'
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)

    # Define the CSV file path
    csv_file = os.path.join(csv_dir, f"{company_name}_data.csv")
    st.write(f"Downloading new data for {company_name}")
    stock = yf.Ticker(company_name)
    data = stock.history(period='max')
    data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)

    # Save the data to CSV
    data.to_csv(csv_file)

    return data, stock


def calculate_rsi(prices, period=14):
    # Calculate RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_indicators(data):
    # Calculate technical indicators
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    data['SMA50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])

    # Calculate Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=20).mean()
    data['BB_std'] = data['Close'].rolling(window=20).std()
    data['BB_upper'] = data['BB_middle'] + (data['BB_std'] * 2)
    data['BB_lower'] = data['BB_middle'] - (data['BB_std'] * 2)
    data['bullish'] = (data['Close'].shift(-1) > data['Close']).astype(int)

    return data


def create_visualization(data, company_name):
    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-90:], data['Close'][-90:], label='Close Price')
    plt.plot(data.index[-90:], data['SMA20'][-90:], label='20-day SMA')
    plt.plot(data.index[-90:], data['SMA50'][-90:], label='50-day SMA')
    plt.plot(data.index[-90:], data['BB_upper'][-90:],
             label='Upper Bollinger Band', linestyle='--')
    plt.plot(data.index[-90:], data['BB_lower'][-90:],
             label='Lower Bollinger Band', linestyle='--')
    plt.fill_between(data.index[-90:], data['BB_upper']
                     [-90:], data['BB_lower'][-90:], alpha=0.1)
    plt.title('Stock Price, Indicators, and Bollinger Bands (Last 3 Months)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)
    plt.close()


def prepare_data(data, look_back=100):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


def create_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def make_prediction(data):
    look_back = 100
    X, y, scaler = prepare_data(data, look_back)

    model = create_lstm_model(look_back)
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)

    last_100_days = data['Close'][-look_back:].values.reshape(-1, 1)
    last_100_days_scaled = scaler.transform(last_100_days)

    future_predictions = []
    current_batch = last_100_days_scaled.reshape((1, look_back, 1))

    for _ in range(30):  # Predict next 30 days
        current_pred = model.predict(current_batch)[0]
        future_predictions.append(current_pred)
        current_batch = np.roll(current_batch, -1, axis=1)
        current_batch[0, -1, 0] = current_pred

    future_predictions = scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1))
    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(days=1), periods=30)

    return future_dates, future_predictions.flatten()


def prepare_data_for_bullish_prediction(data):
    features = ['Close', 'SMA20', 'SMA50', 'RSI', 'BB_upper', 'BB_lower']
    X = data[features].dropna()
    y = data['bullish'].loc[X.index]
    return X, y


def create_bullish_prediction_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, accuracy, report


def predict_tomorrow_bullish(model, latest_data):
    features = ['Close', 'SMA20', 'SMA50', 'RSI', 'BB_upper', 'BB_lower']
    X = latest_data[features].iloc[-1].values.reshape(1, -1)
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]  # Probability of being bullish
    return prediction, probability


def generate_report(company_name, data, stock, bullish_model, bullish_accuracy, bullish_report, future_dates, future_prices):
    # Generate analysis report
    report = f"Technical Analysis Report for {company_name}\n\n"
    report += f"Current Price: {data['Close'].iloc[-1]:.2f}\n"
    report += f"50-day SMA: {data['SMA50'].iloc[-1]:.2f}\n"
    report += f"RSI: {data['RSI'].iloc[-1]:.2f}\n"
    report += f"Bollinger Bands:\n"
    report += f"  Upper: {data['BB_upper'].iloc[-1]:.2f}\n"
    report += f"  Middle: {data['BB_middle'].iloc[-1]:.2f}\n"
    report += f"  Lower: {data['BB_lower'].iloc[-1]:.2f}\n\n"
    report += f"Company Fundamentals:\n"
    report += f"Market Cap: {stock.info.get('marketCap', 'N/A')}\n"
    report += f"P/E Ratio: {stock.info.get('trailingPE', 'N/A')}\n"
    report += f"Dividend Yield: {stock.info.get('dividendYield', 'N/A')}\n\n"

    # Add bullish prediction
    prediction, probability = predict_tomorrow_bullish(bullish_model, data)
    report += f"Bullish Prediction for Tomorrow:\n"
    report += f"Prediction: {'Bullish' if prediction ==
                             1 else 'Not Bullish'}\n"
    report += f"Probability of being Bullish: {probability:.2f}\n\n"
    report += f"Bullish Prediction Model Accuracy: {bullish_accuracy:.2f}\n"
    report += f"Bullish Prediction Model Report:\n{bullish_report}\n"
    report += "LSTM-based Prediction for the next 30 days:\n"
    report += f"Predicted Price (NEXT DAY): {future_prices[0]:.2f}\n"
    report += f"Predicted Price (30 Days): {future_prices[-1]:.2f}\n"
    return report


def main():
    # Set up the API key for OpenAI
    open_api_key = "YOUR_API_KEY_HERE"
    os.environ["OPENAI_API_KEY"] = open_api_key

    # Streamlit title and input
    st.title('Stock Analysis and Prediction')
    company_name = st.text_input(
        "Enter the company stock symbol (e.g., AAPL for Apple):")

    if company_name:
        data, stock = get_stock_data(company_name)
        data = calculate_indicators(data)
        create_visualization(data, company_name)
        future_dates, future_prices = make_prediction(data)

        X, y = prepare_data_for_bullish_prediction(data)
        bullish_model, bullish_accuracy, bullish_report = create_bullish_prediction_model(
            X, y)

        report = generate_report(company_name, data, stock, bullish_model, bullish_accuracy, bullish_report,
                                 future_dates, future_prices)
        st.text_area("Generated Report", report)

        # Initialize ChatOpenAI for summary
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chat_messages = [
            SystemMessage(content='You are an expert financial advisor'),
            HumanMessage(
                content=f'Please provide a short and concise summary of the following report and impact of each indicator for investment purpose:\n TEXT: {report}')
        ]
        summary = llm(chat_messages).content
        st.subheader('Summary')
        st.write(summary)


if __name__ == "__main__":
    main()
