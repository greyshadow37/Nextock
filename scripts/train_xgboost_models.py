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


warnings.filterwarnings("ignore")

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
    "JPM", "GS", "BAC", "C", "V", "MA", 
    "XOM", "CVX", "BP", 
    "JNJ", "PFE", "UNH",
    "WMT", "COST", "KO", "PEP", "PG", 
    "SPY", "QQQ", "DIA"
]

OHLCV = ["Open","High","Low","Close","Volume"]

predictions = {}
path = r"D:\personal-projects\Nextock\arima_predictions"


for ticker in tickers:
    ticker_predictions = {}
    for ohlcv in OHLCV:
        file_name = f"{ticker}_{ohlcv}_Predictions.csv"
        file_path = os.path.join(path, file_name)
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            ticker_predictions[ohlcv] = df["Predictions"]
            print(f"Loaded predictions for {ticker} - {ohlcv}")
    if len(ticker_predictions) == len(OHLCV):  
        predictions[ticker] = pd.DataFrame(ticker_predictions)
        print(f"Created dataframe for {ticker} with OHLCV data.")
    else:
        print(f"Missing OHLCV data for {ticker}, skipping.")

print(f"Loaded predictions for: {list(predictions.keys())}")


df_train = {}
df_test = {}
for ticker, df in predictions.items():
    X = df.drop(columns=["Close"])  
    y = df["Close"]     
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=0.75, random_state=42, shuffle=False)
    df_train[ticker] = {'X': X_train, 'y': y_train}
    df_test[ticker] = {'X': X_test, 'y': y_test}


def objective(study, X_train, y_train):
    params = {
        'n_estimators': study.suggest_int('n_estimators', 100, 500),
        'learning_rate': study.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': study.suggest_int('max_depth', 3, 10),
        'subsample': study.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': study.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_jobs': -1,
        'tree_method': 'gpu_hist'  
    }
    model = xgb.XGBRegressor(**params)
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error', n_jobs=-1).mean()
    return score


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

results = Parallel(n_jobs=-1)(delayed(train_xgboost)(ticker, data['X'], data['y']) for ticker, data in df_train.items())
df_models = dict(results)

df_predictions = {}
df_validations = {}

for ticker, model in df_models.items():
    if ticker in df_test:
        print(f"generating predictions for {ticker}...")
        test = df_test[ticker]
        predictions = model.predict(test["X"])
        actual_values = test["y"].flatten()
        df_predictions[ticker] = predictions
        print(f"generated predictions for {ticker}...")
        print(f"validating {ticker}...")
        mae = mean_absolute_error(actual_values, predictions)
        mse = mean_squared_error(actual_values, predictions)
        rmse = mse ** 0.5
        mape = np.mean(np.abs((actual_values - predictions) / np.where(actual_values != 0, actual_values, 1))) * 100
        df_validations[ticker] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}
        print(f"validated {ticker}")

print("Model training and evaluation complete!")




model_path = r"D:\personal-projects\Nextock\xgboost_models"
os.makedirs(model_path, exist_ok=True)
for ticker, model in df_models.items():
    model_filename = os.path.join(model_path, f"{ticker}_xgboost.pkl")
    with open(model_filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model: {model_filename}")


predictions_path = r"D:\personal-projects\Nextock\xgboost_predictions"
os.makedirs(predictions_path, exist_ok=True)
for ticker, predictions in df_predictions.items():
    predictions_filename = os.path.join(predictions_path, f"{ticker}__xgboost_predictions.csv")
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(predictions_filename, index=False)
    print(f"Saved predictions: {predictions_filename}")


validation_path = r"D:\personal-projects\Nextock\xgboost_validations"
os.makedirs(validation_path, exist_ok=True)
for ticker, metrics in df_validations.items():
    validation_filename = os.path.join(validation_path, f"{ticker}_xgboost_validation.csv")
    pd.DataFrame([metrics]).to_csv(validation_filename, index=False)
    print(f"Saved validation metrics: {validation_filename}")
