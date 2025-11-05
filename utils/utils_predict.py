import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error


def load_data(csv_path, timestamp_col="timestamp", id_col="id", freq="D"):

    df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df = df.set_index(timestamp_col).asfreq(freq)
    df[id_col] = "main"
    df = df.reset_index()
    return df

def metrics(y_true, y_pred):

    rmse = round(root_mean_squared_error(y_true, y_pred), 2)
    mape = round(mean_absolute_percentage_error(y_true, y_pred), 2)
    mae = round(mean_absolute_error(y_true, y_pred), 2)
    r2 = round(r2_score(y_true, y_pred), 2)    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def generate_ew_windows(df, timestamp_col, train_end_date, prediction_length):

    test_start = pd.Timestamp(train_end_date) + pd.Timedelta(days=1)
    test_end = df[timestamp_col].max()

    for start_date in pd.date_range(test_start, test_end, freq=f"{prediction_length}D"):
        end_date = min(start_date + pd.Timedelta(days=prediction_length - 1), test_end)
        context_df = df[df[timestamp_col] < start_date]
        future_df = df[(df[timestamp_col] >= start_date) & (df[timestamp_col] <= end_date)]
        yield  context_df, future_df



def generate_sw_windows(df, timestamp_col, wtype, prediction_length):

    if wtype == "SW-1Y7D":
        context_size = 365
        train_end = pd.Timestamp("2018-12-31")
    elif wtype == "SW-2Y7D":
        context_size = 730
        train_end = pd.Timestamp("2018-12-31")
    else:
        raise ValueError("wtype inválido")

    test_start = train_end + pd.Timedelta(days=1)
    test_end = df[timestamp_col].max()

    for start_date in pd.date_range(test_start, test_end, freq=f"{prediction_length}D"):
        end_date = min(start_date + pd.Timedelta(days=prediction_length - 1), test_end)
        context_df = df[
            (df[timestamp_col] >= start_date - pd.Timedelta(days=context_size)) &
            (df[timestamp_col] < start_date)
        ]
        future_df = df[(df[timestamp_col] >= start_date) & (df[timestamp_col] <= end_date)]
        yield  context_df, future_df
