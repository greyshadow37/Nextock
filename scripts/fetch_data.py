# This script fetches OHLCV (Open, High, Low, Close, Volume) data from the yfinance API
# with the date range of 1st January 2000 to 1st January 2025 (25 years)
# and stores them in the "data/" folder for further analysis.  
# The following tickers represent stocks from various sectors, selected for their stability and availability 
# of historical data:
#
#     - Technology: AAPL (Apple Inc.), MSFT (Microsoft Corporation), GOOGL (Alphabet Inc.), AMZN (Amazon.com), NVDA (NVIDIA)
#     - Finance: JPM (JPMorgan Chase), GS (Goldman Sachs), BAC (Bank of America), C (Citigroup), V (Visa), MA (Mastercard)
#     - Energy: XOM (Exxon Mobil), CVX (Chevron), BP (BP p.l.c.)
#     - Healthcare: JNJ (Johnson & Johnson), PFE (Pfizer), UNH (UnitedHealth)
#     - Consumer Goods: WMT (Walmart), COST (Costco), KO (Coca-Cola), PEP (PepsiCo), PG (Procter & Gamble)
#     - Index ETFs: SPY (S&P 500 ETF), QQQ (Invesco QQQ), DIA (Dow Jones ETF)
# 
# Output:
# - The downloaded stock data will be saved as individual CSV files in the specified 'data/' directory.
#
# Dependencies:
# - yfinance
# - pandas
# - time

import yfinance as yf
import pandas as pd
import time

# tickers contain the OHLCV data of their respective organisations
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", # Technology
    "JPM", "GS", "BAC", "C", "V", "MA", # Finance
    "XOM", "CVX", "BP", # Energy
    "JNJ", "PFE", "UNH", # Healthcare
    "WMT", "COST", "KO", "PEP", "PG", # Consumer Goods
    "SPY", "QQQ", "DIA" # Index ETFs
]

# fetch data from 1st January 2000 - 1st January 2025
start_date = "2000-01-01"
end_date = "2025-01-01"

data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker') # download bulk data

output_dir = r"D:\personal-projects\Nextock\data" # data/ folder path

for ticker in tickers:
    try:
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date) # fetching data for individual tickers
        
        if df.empty: # in case of empty data
            print(f"No data found for {ticker}, skipping.")
            continue
        
        df.to_csv(f"{output_dir}/{ticker}.csv")
        print(f"Saved {ticker}.csv successfully!")

        time.sleep(1) # lag to prevent surpassing the API rate limit 

    except Exception as e: # in case of any unprecedented error 
        print(f"Error fetching {ticker}: {e}")

print("All stock data downloaded and saved.")
