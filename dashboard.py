import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import pickle
import datetime
import matplotlib.pyplot as plt


ARIMA_PATH = "D:/personal-projects/Nextock/arima_models"
XGBOOST_PATH = "D:/personal-projects/Nextock/xgboost_models"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "GS", "BAC", "C", "V", "MA",
    "XOM", "CVX", "BP",
    "JNJ", "PFE", "UNH",
    "WMT", "COST", "KO", "PEP", "PG",
    "SPY", "QQQ", "DIA"
]

PERIODS = {"Next Day": 1, "Next Week": 7, "Next Month": 30}



@st.cache_data
def fetch_data(ticker):
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=180)  
    df = yf.download(ticker, start=start_date, end=end_date)
    return df



def predict_arima(ticker, forecast_days):
    predicted_ohlcv = {}
    ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"]
    for col in ohlcv_columns:
        model_filename = os.path.join(ARIMA_PATH, f"{ticker}_{col}_arima.pkl")
        if os.path.exists(model_filename):
            with open(model_filename, "rb") as f:
                model = pickle.load(f)
            predicted_ohlcv[col] = model.predict(n_periods=forecast_days)
        else:
            st.error(f"Missing ARIMA model for {ticker} - {col}")
            return None
    return pd.DataFrame(predicted_ohlcv)



def predict_xgboost(ticker, arima_predictions):
    model_filename = os.path.join(XGBOOST_PATH, f"{ticker}_xgboost.pkl")
    if os.path.exists(model_filename):
        with open(model_filename, "rb") as f:
            xgb_model = pickle.load(f)
        X_test = arima_predictions.drop(columns=["Close"]).values
        predicted_close = xgb_model.predict(X_test)
        return predicted_close
    else:
        st.error(f"Missing XGBoost model for {ticker}")
        return None



st.title("📈 Nextock - Your One Stop Stock Prediction Solution")


selected_ticker = st.selectbox("Select a Stock Ticker", TICKERS)


selected_period = st.selectbox("Select Forecast Period", list(PERIODS.keys()))
forecast_days = PERIODS[selected_period]


st.subheader(f"Last 6 Months Data for {selected_ticker}")
historical_data = fetch_data(selected_ticker)

if historical_data is not None and not historical_data.empty:
    st.line_chart(historical_data["Close"])
else:
    st.warning("Failed to fetch historical data.")


if st.button("Predict Next Close Price"):
    arima_predictions = predict_arima(selected_ticker, forecast_days)
    if arima_predictions is not None:
        predicted_close_prices = predict_xgboost(selected_ticker, arima_predictions)
        if predicted_close_prices is not None:
            predicted_df = pd.DataFrame({
                "Date": pd.date_range(start=historical_data.index[-1], periods=forecast_days + 1, freq="D")[1:],
                "Predicted Close": predicted_close_prices
            })
            st.subheader(f"Predicted Close Price for {selected_ticker} ({selected_period})")
            st.line_chart(predicted_df.set_index("Date"))
            st.write(predicted_df)
        else:
            st.error("XGBoost prediction failed.")
    else:
        st.error("ARIMA prediction failed.")

