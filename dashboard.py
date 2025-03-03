# This script creates a dashboard using Streamlit for the visualisation of models. 
# It takes the ticker and the forecast period as input and then fetches data from yfinance API
# to feed to the ARIMA models whose predictions are fed to the XGBoost models and the final
# Predicted Close price along with the fetched data is displayed as a table and visualised in a line graph. 
# 
# The following steps are performed:
# 1. Inputing Stock Ticker and Forecast Period: The Stock Ticker and the Forecast Ticker are taken 
# from the user as input and the "Predict Next Close Price" button is clicked.
# 2. Data Loading: Data of the past 6 months for that Ticker is fetched.
# 3. ARIMA Forecasting: ARIMA models forecast OHLCV values, which are then used as input features for XGBoost to predict the next closing price..
# 4. Final Close Price Prediction using XGBoost: The XGBoost model infers on that data and returns the Predicted Close price to be visualised.
# 5. Data Display and Visualisaion: The data is displayed as a table and visualised as a line graph.
# 
# Output:
# - The Predicted Close Price for the next selected forecast period visualised as a line graph.
#
# Dependencies:
# - pandas
# - yfinance
# - pickle
# - datetime 
# - os
# - streamlit
# - time

import streamlit as st
import pandas as pd
import yfinance as yf
import os
import pickle
import datetime
import time



ARIMA_PATH = "D:/personal-projects/Nextock/arima_models" # path to the arima_models/ folder
XGBOOST_PATH = "D:/personal-projects/Nextock/xgboost_models" # path to the xgboost_models/ folder

# list containing the tickers
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "GS", "BAC", "C", "V", "MA",
    "XOM", "CVX", "BP",
    "JNJ", "PFE", "UNH",
    "WMT", "COST", "KO", "PEP", "PG",
    "SPY", "QQQ", "DIA"
]

# dictionary containing the forecast period
PERIODS = {"Next Day": 1, "Next Week": 7, "Next Month": 30}


# function to fetch data from the yfinance API
@st.cache_data
def fetch_data(ticker):
    time.sleep(1)  # lag to prevent surpassing the API rate limit 
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=180)  
    df = yf.download(ticker, start=start_date, end=end_date)
    return df


# function to infer on trained ARIMA models
def predict_arima(ticker, forecast_days):
    predicted_ohlcv = {} # dictionary to store OHLCV predictions
    ohlcv_columns = ["Open", "High", "Low", "Close", "Volume"] # list containing OHLCV
    for col in ohlcv_columns:
        model_filename = os.path.join(ARIMA_PATH, f"{ticker}_{col}_arima.pkl")
        if os.path.exists(model_filename):
            with open(model_filename, "rb") as f:
                model = pickle.load(f)
            predicted_ohlcv[col] = model.predict(n_periods=forecast_days)
        else: # error handling in case the ARIMA model is missing
            st.error(f"Missing ARIMA model for {ticker} - {col}")
            return None
    return pd.DataFrame(predicted_ohlcv)


# function to infer on trained XGBoost models
def predict_xgboost(ticker, arima_predictions):
    model_filename = os.path.join(XGBOOST_PATH, f"{ticker}_xgboost.pkl")
    if os.path.exists(model_filename):
        with open(model_filename, "rb") as f:
            xgb_model = pickle.load(f)
        X_test = arima_predictions.drop(columns=["Close"]).values
        predicted_close = xgb_model.predict(X_test)
        return predicted_close
    else: # error handling in case the XGBoost model is missing
        st.error(f"Missing XGBoost model for {ticker}")
        return None




# |-------------------------------------Streamlit UI--------------------------------------------|

st.title("📈 Nextock - Your One Stop Stock Prediction Solution")

# select box to choose Ticker
selected_ticker = st.selectbox("Select a Stock Ticker", TICKERS)

# select box to choose forecast period
selected_period = st.selectbox("Select Forecast Period", list(PERIODS.keys()))
forecast_days = PERIODS[selected_period]


st.subheader(f"Last 6 Months Data for {selected_ticker}")
historical_data = fetch_data(selected_ticker) # fetching the data 

if historical_data is not None and not historical_data.empty:
    st.line_chart(historical_data["Close"])
else: # if data fetching fails
    st.warning("Failed to fetch historical data.")


if st.button("Predict Next Close Price"):
    arima_predictions = predict_arima(selected_ticker, forecast_days) # infer on ARIMA models
    if arima_predictions is not None:
        predicted_close_prices = predict_xgboost(selected_ticker, arima_predictions) # infer on XGBoost models
        if predicted_close_prices is not None:
            predicted_df = pd.DataFrame({
# DataFrame contains Date and Predicted Close
                "Date": pd.date_range(start=historical_data.index[-1], periods=forecast_days + 1, freq="D")[1:],
                "Predicted Close": predicted_close_prices
            })
            st.subheader(f"Predicted Close Price for {selected_ticker} ({selected_period})")
            st.line_chart(predicted_df.set_index("Date")) # visualise data as a line graph
            st.write(predicted_df) # display data as a table
        else: # if XGBoost inference fails
            st.error("XGBoost prediction failed.")
    else: # if ARIMA inference fails
        st.error("ARIMA prediction failed.")

