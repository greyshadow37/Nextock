# This script loads preprocessed data from "preprocessed_data/" folder, trains 5 ARIMA models using auto_arima
# for each feature per ticker and validates them using the following metrics:
# Mean Squared Error(MSE), Mean Absolute Error(MAE),
# Root Mean Squared Error(RMSE) and Mean Absolute Percentage Error(MAPE) and 
# then saves the models in "arima_models/" folder and the predictions in
# "arima_predictions/" folder and the validation metrics in "arima_validations/" folder.
# 
# The following steps are performed:
# 1. Data Loading: The preprocessed data is read from the CSV files in the "data/" folder into a dictionary for processing.
# 2. Train-Test Split: The data is split into training and testing data in 4:1 ratio sequentially preserving the order.
# 3. Model Training: 125 ARIMA models are trained in parallel using pmdarima and joblib Parallel and the models are saved 
# in models dictionary.
# 4. Model Prediction and Validation: The models generates predictions on the test data and validates them and saves the
# predictions in df_predictions and df_validation.
# 5. Saving the Models and their Predictions and Validations: The models, their predictions and their validations are 
# saved in "arima_models/", "arima_predictions" and "arima_validations" respectively. 
# 
# Outputs:
# - 125 .pkl files of ARIMA models of OHLCV feature for each ticker
# - 125 CSV files containing the predictions of each of those models
# - 125 validation files for each of those ARIMA models
# 
# Dependencies:
# - pandas
# - scikit-learn 
# - pmdarima   
# - pickle
# - joblib
# - os
# - warnings

import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import pickle
from joblib import Parallel, delayed
import warnings


warnings.filterwarnings("ignore") # suppressing warnings

df_dict = {} # dictionary to load the preprocessed data

path = r"D:\personal-projects\Nextock\preprocessed_data" # preprocessed data file
files = os.listdir(path)
# loading preprocessed data for each ticker
for file in files:
    if file.endswith(".csv"):  
        file_path = os.path.join(path, file)
        print(f"Loading {file_path}")
        df_name = os.path.splitext(file)[0]  
        df_dict[df_name] = pd.read_csv(file_path)


cols = ["Close", "Open", "High", "Low", "Volume"] # list to store OHLCV as items
models = {col: {} for col in cols} # dictionary to store models
df_dict_train = {} # dictionary to store training data for each feature of each ticker
df_dict_test = {} # dictionary to store test data
df_predictions = {} # dictionary to store predictions
df_validation = {} # dictionary to store validation metrics

# train test split of the preprocessed data
for key, df in df_dict.items():
    train_size = int(len(df) * 0.8)  
    train, test = df[:train_size], df[train_size:]
    df_dict_train[key] = train
    df_dict_test[key] = test

# function to train models
def train_model(df, col, key):
    print(f"Training auto_arima for {key} - {col}")
    df = df.copy() # making a copy to preserve the original data in case of errors
    df[col] = df[col].fillna(method="ffill").fillna(method="bfill") # forward and backward filling the data to make sure no NaN values exist
    # condition to check for variance
    if df[col].nunique() <= 1:
        # skip training if there is insufficient variance 
        print(f"Skipping {key} - {col} due to insufficient variance.")
        return key, col, None
    # use auto_arima for automatic hyperparameter tuning
    model = auto_arima(df[col], 
                   seasonal=False, 
                   trace=True, 
                   suppress_warnings=True, 
                   max_p=5, max_q=5, max_d=2)
    return key, col, model

# train ARIMA models in parallel using joblib
results = Parallel(n_jobs=4, backend="loky")(delayed(train_model)(df, col, key) 
                              for key, df in df_dict_train.items() 
                              for col in cols)
# save the models in models dictionary
for key, col, model in results:
    models[col][key] = model

# generate predictions and validation metrics for each model
for col in cols:
    df_predictions[col] = {}
    df_validation[col] = {}
# for each model trained on each feature of each ticker
    for key, model in models[col].items():
# and if the ticker exists in the test set
        if key in df_dict_test:
# we generate predictions and validation metrics
            print(f"generating predictions for {col} of {key}...")
            test = df_dict_test[key]
            predictions = model.predict(n_periods=len(test))
            df_predictions[col][key] = predictions
            print(f"generated predictions for {col} of {key}.")
            print(f"validating predictions for {col} of {key}...")
# calculating validation metrics            
            actual_values = test[col].values
            mae = mean_absolute_error(actual_values, predictions)
            mse = mean_squared_error(actual_values, predictions)
            rmse = mse ** 0.5
            mask = actual_values != 0
            nonzero_actuals = actual_values[mask]
            nonzero_predictions = predictions[mask]
            if len(nonzero_actuals) > 0:
                mape = (abs((nonzero_actuals - nonzero_predictions) / nonzero_actuals)).mean() * 100
            else:
                mape = None if len(nonzero_actuals) == 0 else (abs((nonzero_actuals - nonzero_predictions) / nonzero_actuals)).mean() * 100

            df_validation[col][key] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "MAPE": mape}
            print(f"validated {col} of {key}.")


print("Model training and evaluation complete!")

# saving the models in arima_models/ folder
model_path = r"D:\personal-projects\Nextock\arima_models"
os.makedirs(model_path, exist_ok=True)
for col in cols:
    for key, model in models[col].items():
        model_filename = os.path.join(model_path, f"{key}_{col}_arima.pkl")
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model: {model_filename}")

# saving the predictions in arima_predictions/ folder
predictions_path = r"D:\personal-projects\Nextock\arima_predictions"
os.makedirs(predictions_path, exist_ok=True)
for col in cols:
    for key, predictions in df_predictions[col].items():
        predictions_filename = os.path.join(predictions_path, f"{key}_{col}_predictions.csv")
        pd.DataFrame(predictions, columns=["Predictions"]).to_csv(predictions_filename, index=False)
        print(f"Saved predictions: {predictions_filename}")

# saving validation metrics in arima_validations/ folder
validation_path = r"D:\personal-projects\Nextock\arima_validations"
os.makedirs(validation_path, exist_ok=True)
for col in cols:
    for key, metrics in df_validation[col].items():
        validation_filename = os.path.join(validation_path, f"{key}_{col}_validation.csv")
        pd.DataFrame([metrics]).to_csv(validation_filename, index=False)
        print(f"Saved validation metrics: {validation_filename}")