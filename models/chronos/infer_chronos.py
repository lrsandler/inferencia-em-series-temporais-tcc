import os
import json
import torch
import numpy as np
import pandas as pd
from utils.utils_predict import load_data, metrics, generate_ew_windows, generate_sw_windows
from chronos import BaseChronosPipeline, Chronos2Pipeline
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*") #ignora aviso carrega na cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "./data/dataset.csv"
target_col = "water_produced"
timestamp_col = "timestamp"
id_col = "id"
pred_length = 7 

def load_model():
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "s3://autogluon/chronos-2/",
        device_map=device
    )
    return pipeline

def predict_with_window(pipeline, df, save_dir, wtype="EW", years=None):
    results = []

    if wtype == "EW":
        train_end_date = "2018-12-31"
        window_gen = generate_ew_windows(df, timestamp_col, train_end_date, prediction_length=pred_length)
        fname = "EW-7D"
    elif wtype == "SW":
        window_gen = generate_sw_windows(df, timestamp_col, f"SW-{years}Y7D", prediction_length=pred_length)
        fname = f"SW-{years}Y7D"

    for context_df, future_df in window_gen:
        pred_df = pipeline.predict_df(
            context_df,
            prediction_length=len(future_df),
            id_column=id_col,
            timestamp_column=timestamp_col,
            target=target_col
        )

        y_true = future_df[target_col].values
        y_pred = pred_df["predictions"].values[:len(y_true)]

        fold_res = pd.DataFrame({
            "timestamp": future_df[timestamp_col].values,
            "ytrue": y_true,
            "yhat": y_pred
        })
        results.append(fold_res)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(f"{save_dir}/chronos_{fname}.csv", index=False)

    m = metrics(results_df["ytrue"], results_df["yhat"])
    with open(f"{save_dir}/{fname}_metrics.json", "w") as f:
        json.dump(m, f, indent=4)
    print(f"Métricas {fname}:", m)
    return results_df

def predict_full_year(pipeline, df, save_dir):
    context_df = df[df[timestamp_col] <= "2018-12-31"]
    future_df = df[df[timestamp_col] >= "2019-01-01"]

    pred_df = pipeline.predict_df(
        context_df,
        prediction_length=len(future_df),
        id_column=id_col,
        timestamp_column=timestamp_col,
        target=target_col
    )

    y_true = future_df[target_col].values
    y_pred = pred_df["predictions"].values[:len(y_true)]

    results_df = pd.DataFrame({
        "timestamp": future_df[timestamp_col].values,
        "ytrue": y_true,
        "yhat": y_pred
    })

    os.makedirs(save_dir, exist_ok=True)
    results_df.to_csv(f"{save_dir}/chronos_full_2019.csv", index=False)

    m = metrics(y_true, y_pred)
    with open(f"{save_dir}/full_2019_metrics.json", "w") as f:
        json.dump(m, f, indent=4)

    print(f"Métricas full-year:", m)
    return results_df



def run_chronos():
    pipeline = load_model()
    df = load_data(csv_path)
    save_dir = "./results/chronos"
    os.makedirs(save_dir, exist_ok=True)

    predict_with_window(pipeline, df, save_dir, wtype="EW")
    predict_with_window(pipeline, df, save_dir, wtype="SW", years=1)
    predict_with_window(pipeline, df, save_dir, wtype="SW", years=2)
    predict_full_year(pipeline, df, save_dir)

    # Backtest igual ao repositorio  
    # https://github.com/jesuinovieira/bachelor-thesis/blob/main/src/processor.py

if __name__ == "__main__":
    run_chronos()


