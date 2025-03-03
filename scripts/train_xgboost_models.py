# This script loads the predictions of the OHLCV data made by the ARIMA models from
# "arima_predictions/" folder and then stores them in predictions dictionary after
# combining the data of each ticker into a single key. It then predicts the Close Price using 
# an XGBoost model for each ticker and validates them using the following metrics:
# Mean Squared Error(MSE), Mean Absolute Error(MAE),
# Root Mean Squared Error(RMSE) and Mean Absolute Percentage Error(MAPE) and 
# stores the models, predictions and validation metrics in "xgboost_models/",
# "xgboost_predictions/" and "xgboost_validations/" folders.
# 
# The following steps are performed:
# 1. Data Loading: The predictions made by the ARIMA models are read from the CSV files in the 
# "arima_predictions/" folder and the OHLCV data is combined into a single dictionary for processing.
# 2. Train-Test Split: The data is split into training and testing data in 3:1 ratio using scikit-learn.
# 3. Hyperparameter Tuning Model Training: Hyperparameter tuning is done for each of those 25 XGBoost models using 
# optuna and they are trained in parallel using xgboost and joblib Parallel and the models are saved in models dictionary. 
# 4. Saving the Models and their Predictions and Validations: The models, their predictions and their validations
# are saved in "xgboost_models/", "xgboost_predictions" and "xgboost_validations" respectively. 
# 
# Outputs:
# - 25 .pkl files of XGBoost models for each ticker
# - 25 CSV files containing the predictions of each of those models
# - 25 validation files for each of those XGBoost models
# 
# Dependencies:
# - numpy
# - pandas
# - scikit-learn
# - xgboost
# - optuna
# - joblib
# - pickle
# - warnings
# - os
# Note: The XGBoost models are trained on GPU. So ensure your machine has one while running the script.

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna
from joblib import Parallel, delayed
import pickle
import warnings


warnings.filterwarnings("ignore") # suppressing warnings

# list containing ticker values
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
    "JPM", "GS", "BAC", "C", "V", "MA", 
    "XOM", "CVX", "BP", 
    "JNJ", "PFE", "UNH",
    "WMT", "COST", "KO", "PEP", "PG", 
    "SPY", "QQQ", "DIA"
]

# list containing OHLCV 
OHLCV = ["Open","High","Low","Close","Volume"]

predictions = {} # dictionary to store fetched predictions
path = r"D:\personal-projects\Nextock\arima_predictions" # path to the arima_predictions/ folder

# loading ARIMA predictions for each ticker
for ticker in tickers:
    ticker_predictions = {} # dictionary to store the predictions temporarily for recombination
    for ohlcv in OHLCV:
        file_name = f"{ticker}_{ohlcv}_Predictions.csv"
        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            ticker_predictions[ohlcv] = df["Predictions"] # storing the individual predictions of each ticker
            print(f"Loaded predictions for {ticker} - {ohlcv}")
    if len(ticker_predictions) == len(OHLCV):  
# converting the dictionary into a DataFrame and adding it to the predictions dictionary
        predictions[ticker] = pd.DataFrame(ticker_predictions)
        print(f"Created dataframe for {ticker} with OHLCV data.")
    else:
        print(f"Missing OHLCV data for {ticker}, skipping.")

print(f"Loaded predictions for: {list(predictions.keys())}")


df_train = {} # dictionary to store training data 
df_test = {} # dictionary to store testing data
# train test split of the ARIMA predictions
for ticker, df in predictions.items():
    X = df.drop(columns=["Close"])  # OHLV are the input features
    y = df["Close"] # Close is the target variable
# the data is split in 3:1 ratio using scikit-learn's train test split function with shuffling disabled
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=0.75, random_state=42, shuffle=False)
    df_train[ticker] = {'X': X_train, 'y': y_train}
    df_test[ticker] = {'X': X_test, 'y': y_test}


# function to tune hyperparameters
def objective(study, X_train, y_train):
    params = {
        'n_estimators': study.suggest_int('n_estimators', 100, 500),
        'learning_rate': study.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': study.suggest_int('max_depth', 3, 10),
        'subsample': study.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': study.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_jobs': -1,
        'tree_method': 'gpu_hist'  # GPU is enabled 
    }
    model = xgb.XGBRegressor(**params) # XGBoost model
    # computing the cross validation score
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return score


# function to train the XGBoost models
def train_xgboost(ticker, X_train, y_train):
    print(f"Training XGBoost for {ticker} on GPU...")
    study = optuna.create_study(direction='minimize')   
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50, timeout=300)
    best_params = study.best_params
    best_params['n_jobs'] = -1
    best_params['tree_method'] = 'gpu_hist'  
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train.ravel())    
    print(f"Trained XGBoost for {ticker} on GPU.")
    return ticker, model

# training models in parallel using joblib Parallel
results = Parallel(n_jobs=-1)(delayed(train_xgboost)(ticker, data['X'], data['y']) for ticker, data in df_train.items())
df_models = dict(results)

df_predictions = {} # dictionary to store the predictions
df_validations = {} # dictionary to store the validation metrics

# generate predictions and validation metrics for each ticker 
for ticker, model in df_models.items():
    if ticker in df_test:
        print(f"generating predictions for {ticker}...")
        test = df_test[ticker]
        predictions = model.predict(test["X"])
        actual_values = test["y"].flatten() # flattening the y values to ensure consistent shape
        df_predictions[ticker] = predictions
        print(f"generated predictions for {ticker}...")
        print(f"validating {ticker}...")
# calculating validation metrics
        mae = mean_absolute_error(actual_values, predictions)
        mse = mean_squared_error(actual_values, predictions)
        rmse = mse ** 0.5
        mape = np.mean(np.abs((actual_values - predictions) / np.where(actual_values != 0, actual_values, 1))) * 100
        df_validations[ticker] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}
        print(f"validated {ticker}")

print("Model training and evaluation complete!")



# saving the models in xgboost_models/ folder
model_path = r"D:\personal-projects\Nextock\xgboost_models"
os.makedirs(model_path, exist_ok=True)
for ticker, model in df_models.items():
    model_filename = os.path.join(model_path, f"{ticker}_xgboost.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_filename}")


# saving the predictions in xgboost_predictions/ folder
predictions_path = r"D:\personal-projects\Nextock\xgboost_predictions"
os.makedirs(predictions_path, exist_ok=True)
for ticker, predictions in df_predictions.items():
    predictions_filename = os.path.join(predictions_path, f"{ticker}__xgboost_predictions.csv")
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(predictions_filename, index=False)
    print(f"Saved predictions: {predictions_filename}")


# saving validation metrics in xgboost_validations/ folder
validation_path = r"D:\personal-projects\Nextock\xgboost_validations"
os.makedirs(validation_path, exist_ok=True)
for ticker, metrics in df_validations.items():
    validation_filename = os.path.join(validation_path, f"{ticker}_xgboost_validation.csv")
    pd.DataFrame([metrics]).to_csv(validation_filename, index=False)
    print(f"Saved validation metrics: {validation_filename}")
