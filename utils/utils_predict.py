import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def load_data(csv_path, timestamp_col="timestamp", id_col="id", freq="D"):

    df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df = df.set_index(timestamp_col).asfreq(freq)
    df[id_col] = "main"
    df = df.reset_index()
    return df

def metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    abs_errors = np.abs(y_true - y_pred)
    sq_errors  = (y_true - y_pred) ** 2
    ape        = abs_errors / np.abs(y_true)

    mae  = round(float(abs_errors.mean()), 2)
    mae_std  = round(float(abs_errors.std()), 2)

    rmse = round(float(np.sqrt(sq_errors.mean())), 2)
    rmse_std = round(float(np.sqrt(sq_errors.std())), 2)

    mape = round(float(ape.mean()), 4)
    mape_std = round(float(ape.std()), 4)

    r2 = round(float(r2_score(y_true, y_pred)), 2)

    return {
        "MAE": mae, "MAE_std": mae_std,
        "RMSE": rmse, "RMSE_std": rmse_std,
        "MAPE": mape, "MAPE_std": mape_std,
        "R2": r2,
    }


def metrics_from_csv(csv_path, ytrue_col="ytrue", yhat_col="yhat",
                     save_json=True, json_path=None):

    df = pd.read_csv(csv_path)
    m = metrics(df[ytrue_col], df[yhat_col])

    if save_json:
        if json_path is None:
            json_path = os.path.splitext(csv_path)[0] + "_metrics.json"
        with open(json_path, "w") as f:
            json.dump(m, f, indent=4)
        print(f"Métricas salvas em: {json_path}")

    return m


def generate_ew_windows(df, timestamp_col, train_end_date, prediction_length, initial_context_size=730):

    test_start = pd.Timestamp(train_end_date) + pd.Timedelta(days=1)
    test_end = df[timestamp_col].max()

    # início fixo: 730 dias antes do começo do teste (tamanho inicial da janela)
    ew_start = test_start - pd.Timedelta(days=initial_context_size)

    for start_date in pd.date_range(test_start, test_end, freq=f"{prediction_length}D"):
        end_date = min(start_date + pd.Timedelta(days=prediction_length - 1), test_end)
        context_df = df[(df[timestamp_col] >= ew_start) & (df[timestamp_col] < start_date)]
        future_df = df[(df[timestamp_col] >= start_date) & (df[timestamp_col] <= end_date)]
        yield context_df, future_df



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

