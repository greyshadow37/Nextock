import yfinance as yf
import pandas as pd
import time

# tickers are basically organisations whose stock we are fetching
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", # Technology
    "JPM", "GS", "BAC", "C", "V", "MA", # Finance
    "XOM", "CVX", "BP", # Energy
    "JNJ", "PFE", "UNH", # Healthcare
    "WMT", "COST", "KO", "PEP", "PG", # Consumer Goods
    "SPY", "QQQ", "DIA" # Index ETFs
]

start_date = "2000-01-01"
end_date = "2025-01-01"

data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
output_dir = r"D:\personal-projects\Nextock\data"

for ticker in tickers:
    try:
        print(f"Fetching data for {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        
        if df.empty: 
            print(f"No data found for {ticker}, skipping.")
            continue
        
        df.to_csv(f"{output_dir}/{ticker}.csv")
        print(f"Saved {ticker}.csv successfully!")

        time.sleep(1) 

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")

print("All stock data downloaded and saved.")
