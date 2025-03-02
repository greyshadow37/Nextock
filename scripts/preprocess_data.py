import pandas as pd
import numpy as np
import os
from statsmodels.tsa.stattools import adfuller

df_dict = {}
path = r"D:\personal-projects\Nextock\data"
files = os.listdir(path)
for file in files:
    file_path = os.path.join(path, file)
    df_dict[file] = pd.read_csv(file_path, skiprows=[1, 2])
    print(f"loaded {file_path}")

def adfuller_check(data, col):
    result = adfuller(data[col].dropna())
    return result[1]

def differencing(data, col):
    data[col] = data[col].diff()  
    if adfuller_check(data.dropna(subset=[col]), col) >= 0.05:  
        data[col] = data[col].diff()  
    return data[col]

processed_df_dict = {}  

for key, df in df_dict.items():
    print(f"preprocessing {key}.")
    df_copy = df.copy()  
    df_copy.rename(columns={"Price": "Date"}, inplace=True)
    print("renamed date column.")

    df_copy['Date'] = pd.to_datetime(df_copy['Date'])
    df_copy.set_index('Date', inplace=True)
    df_copy.sort_index(inplace=True)
    print("converted date to datetime format and set and sorted it as index.")

    df_copy.interpolate(method="time", inplace=True)
    print("Filled missing values!")

    cols = ["Close", "High", "Low", "Open", "Volume"]

    for col in cols:
        if col in df_copy.columns:  
            if adfuller_check(df_copy, col) >= 0.05:
                print(f"{col} of {key} is not stationary, applying differencing...")
                df_copy[col] = differencing(df_copy, col)
                print(f"Successfully differentiated {col} of {key}!")

    df_copy.fillna(method="bfill", inplace=True)
    df_copy.fillna(method="ffill", inplace=True)
    
    print(f"successfully preprocessed {key}!")
    processed_df_dict[key] = df_copy 

print("preprocessing complete!")

output_path = r"D:\personal-projects\Nextock\preprocessed_data"
for key, df in processed_df_dict.items():
    print(f"Converting {key} into CSV...")
    file_name = os.path.splitext(key)[0] + ".csv"
    file_path = os.path.join(output_path, file_name)
    df.to_csv(file_path, index=True)
    print(f"Converted {key} into CSV!")
