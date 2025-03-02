import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
import pickle
from joblib import Parallel, delayed
import warnings


warnings.filterwarnings("ignore")

df_dict = {}

path = r"D:\personal-projects\Nextock\preprocessed_data"
files = os.listdir(path)
for file in files:
    if file.endswith(".csv"):  
        file_path = os.path.join(path, file)
        print(f"Loading {file_path}")
        df_name = os.path.splitext(file)[0]  
        df_dict[df_name] = pd.read_csv(file_path)


cols = ["Close", "Open", "High", "Low", "Volume"]
models = {col: {} for col in cols}
df_dict_train = {}
df_dict_test = {}
df_predictions = {}
df_validation = {}

for key, df in df_dict.items():
    train_size = int(len(df) * 0.8)  
    train, test = df[:train_size], df[train_size:]
    df_dict_train[key] = train
    df_dict_test[key] = test

def train_model(df, col, key):
    print(f"Training auto_arima for {key} - {col}")
    df = df.copy()
    df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
    if df[col].nunique() <= 1:
        print(f"Skipping {key} - {col} due to insufficient variance.")
        return key, col, None
    model = auto_arima(df[col], 
                   seasonal=False, 
                   trace=True, 
                   suppress_warnings=True, 
                   max_p=5, max_q=5, max_d=2)
    return key, col, model

results = Parallel(n_jobs=4, backend="loky")(delayed(train_model)(df, col, key) 
                              for key, df in df_dict_train.items() 
                              for col in cols)
for key, col, model in results:
    models[col][key] = model


df_validation = {}
for col in cols:
    df_predictions[col] = {}
    df_validation[col] = {}
    
    for key, model in models[col].items():
        if key in df_dict_test:
            
            print(f"generating predictions for {col} of {key}...")
            test = df_dict_test[key]
            predictions = model.predict(n_periods=len(test))
            df_predictions[col][key] = predictions
            print(f"generated predictions for {col} of {key}.")
            print(f"validating predictions for {col} of {key}...")
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

model_path = r"D:\personal-projects\Nextock\arima_models"
os.makedirs(model_path, exist_ok=True)
for col in cols:
    for key, model in models[col].items():
        model_filename = os.path.join(model_path, f"{key}_{col}_arima.pkl")
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)
        print(f"Saved model: {model_filename}")

predictions_path = r"D:\personal-projects\Nextock\arima_predictions"
os.makedirs(predictions_path, exist_ok=True)
for col in cols:
    for key, predictions in df_predictions[col].items():
        predictions_filename = os.path.join(predictions_path, f"{key}_{col}_predictions.csv")
        pd.DataFrame(predictions, columns=["Predictions"]).to_csv(predictions_filename, index=False)
        print(f"Saved predictions: {predictions_filename}")

validation_path = r"D:\personal-projects\Nextock\arima_validations"
os.makedirs(validation_path, exist_ok=True)
for col in cols:
    for key, metrics in df_validation[col].items():
        validation_filename = os.path.join(validation_path, f"{key}_{col}_validation.csv")
        pd.DataFrame([metrics]).to_csv(validation_filename, index=False)
        print(f"Saved validation metrics: {validation_filename}")