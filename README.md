# 📈 Nextock - Your One Stop Stock Prediction Solution
Nextock is a Stock Price Prediction Dashboard built using Streamlit. It allows users to select a stock ticker and a forecast period to predict the next closing price using ARIMA and XGBoost models.


## 🚀 Features
1. Fetches real-time stock data using yfinance API
2. Uses ARIMA models to generate OHLCV (Open, High, Low, Close, Volume) predictions
3. Feeds ARIMA predictions into XGBoost models to predict the next Close price
4. Displays actual and predicted stock prices in an interactive line chart
5. Supports 25 major stock tickers



## 📌 How It Works
1. Select a stock ticker from the dropdown menu.
2. Choose a forecast period (Next Day, Next Week, Next Month).
3. Click "Predict Next Close Price".
4. View Results:
The last 6 months' actual stock prices -
- Predicted close prices for the selected period.
- Data displayed in both table format and graph format.


## 🛠 Installation
**1. Clone the repository:**
            
    git clone https://github.com/greyshadow37/Nextock
   
    cd nextock

**2. Install dependencies:**

    pip install -r requirements.txt
**3. Run the Streamlit dashboard:**

    streamlit run dashboard.py


## 📶 Model Workflow
**1. Data Collection:** Fetches 6 months' historical data from yfinance
**2. ARIMA Model Prediction:** Generates OHLCV forecasts
**3. XGBoost Model Prediction:** Uses ARIMA outputs to predict the next Close price
**4. Visualization:** Plots actual vs predicted stock prices in a Streamlit dashboard

## 💹 Supported Stock Tickers
    AAPL, MSFT, GOOGL, AMZN, NVDA, JPM, GS, BAC, C, V, MA, XOM, CVX, BP, JNJ, PFE, UNH, WMT, COST, KO, PEP, PG, SPY, QQQ, DIA

## ⏳ Future Improvements
Here are some planned enhancements for Nextock:

**1. 📊 Advanced Analytics Options** – Introduce more in-depth analytics, including trend detection and sentiment analysis.

**2. 📅 Forecasting Beyond One Month** – Extend predictions to cover longer periods, such as 3 months, 6 months, or even a year.

**3. 📈 Improved Accuracy & Error Rate Display** – Enhance model performance and provide real-time error metrics (e.g., MAE, RMSE, MAPE).

**4.🧠 Neural Networks & LSTMs** – Integrate deep learning models like LSTMs and transformers for more accurate predictions.

**5. 📡 Support for More Tickers** – Expand coverage beyond the current 25 stock tickers.

**6. 🎨 Improved UI/UX** – Optimize the dashboard for a smoother and more user-friendly experience.