# This script loads stock data from the "data/" folder into a dictionary, preprocesses the data for each ticker, 
# and saves the cleaned data into the "preprocessed_data/" folder.
#
# The following steps are performed:
# 1. Data Loading: The data is read from the CSV files in the "data/" folder into a dictionary for processing.
# 2. Stationarity Check: The Augmented Dickey-Fuller (ADF) test is applied to check if the data is stationary.
#     - If the data is non-stationary (p-value >= 0.05), differencing is applied to make the data stationary.
#     - If necessary, second-order differencing is performed for better stationarity.
# 3. Interpolation and Missing Data Handling: Time interpolation is applied to fill in any missing data points. 
#     - After differencing, any additional missing values are filled using forward-fill and backward-fill methods.
# 4. Data Saving: The processed data for each ticker is saved as a CSV file in the "preprocessed_data/" folder.
#
# Output:
# - CSV files containing Preprocessed Data of each ticker 
#
# Dependencies:
# - pandas
# - statsmodels
# - os

import pandas as pd
import os
from statsmodels.tsa.stattools import adfuller

df_dict = {} # dictionary to load data
path = r"D:\personal-projects\Nextock\data" # path of the data/ folder
files = os.listdir(path) 
# loading data for each ticker
for file in files:
    file_path = os.path.join(path, file)
    df_dict[file] = pd.read_csv(file_path, skiprows=[1, 2])
    print(f"loaded {file_path}")

# function to check the p-value using the Augmented Dickey-Fuller (adfuller) test
def adfuller_check(data, col):
    result = adfuller(data[col].dropna())
    return result[1] # returns the p-value

# function to apply differencing
def differencing(data, col):
    data[col] = data[col].diff()  
    if adfuller_check(data.dropna(subset=[col]), col) >= 0.05: 
        # if the adfuller test does not yield satisfactory result
        # we apply second order differencing 
        data[col] = data[col].diff()  
    return data[col]

processed_df_dict = {}  # dictionary to store preprocessed data

for key, df in df_dict.items():
    print(f"preprocessing {key}.")
    df_copy = df.copy()  # copy of the data to prevent the original data being lost due to any error
    df_copy.rename(columns={"Price": "Date"}, inplace=True)
    print("renamed date column.")
# converting date column to datetime format and setting it as index
    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    df_copy.sort_index(inplace=True)
    print("converted date to datetime format and set and sorted it as index.")
# applying time interpolation
    df_copy.interpolate(method="time", inplace=True)
    print("Filled missing values!")
# list containing the parameters: OHLCV (Open, High, Low, Close, Volume)
    cols = ["Close", "High", "Low", "Open", "Volume"]
# checking the p-value of adfuller test and applying differencing, as per requirement
    for col in cols:
        if col in df_copy.columns:  
            if adfuller_check(df_copy, col) >= 0.05:
                print(f"{col} of {key} is not stationary, applying differencing...")
                df_copy[col] = differencing(df_copy, col)
                print(f"Successfully differentiated {col} of {key}!")
# filling missing values(if any), due to differencing using forward and backward fills
    df_copy.fillna(method="bfill", inplace=True)
    df_copy.fillna(method="ffill", inplace=True)
    
    print(f"successfully preprocessed {key}!")
    processed_df_dict[key] = df_copy 

print("preprocessing complete!")

output_path = r"D:\personal-projects\Nextock\preprocessed_data" # path to store preprocessed data
# saving data as CSV files in the preprocessed_data/ folder
for key, df in processed_df_dict.items():
    print(f"Converting {key} into CSV...")
    file_name = os.path.splitext(key)[0] + ".csv"
    file_path = os.path.join(output_path, file_name)
    df.to_csv(file_path, index=True)
    print(f"Converted {key} into CSV!")
