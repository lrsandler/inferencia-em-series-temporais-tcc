import os
import torch
import pandas as pd
from utils.utils_predict import load_data, metrics_from_csv, generate_ew_windows, generate_sw_windows
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
        "amazon/chronos-2", 
        device_map=device)
    
    # pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
    #     "s3://autogluon/chronos-2/",
    #     device_map=device)

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
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=id_col,
            timestamp_column=timestamp_col,
            target=target_col
        )

        y_true = future_df[target_col].values
        y_pred = pred_df["predictions"].values[:len(y_true)]
        q10 = pred_df["0.1"].values[:len(y_true)]
        q90 = pred_df["0.9"].values[:len(y_true)]

        fold_res = pd.DataFrame({
            "timestamp": future_df[timestamp_col].values,
            "ytrue": y_true,
            "yhat": y_pred,
            "q10": q10,
            "q90": q90,
        })
        results.append(fold_res)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(f"{save_dir}/chronos_{fname}.csv", index=False)
    return fname

covariate_cols = [
    "temperature_mean", "temperature_std",
    "radiation_mean", "radiation_std",
    "relative_humidity_mean", "relative_humidity_std",
    "precipitation_mean", "precipitation_std", "dayofweek",
    "is_weekend", "season",
    "is_holiday_ctba_gtba_jve", "is_carnival", "is_school_recess_pr"
]

def predict_with_window_covariates(pipeline, df, save_dir, wtype="EW", years=None):
    results = []

    if wtype == "EW":
        train_end_date = "2018-12-31"
        window_gen = generate_ew_windows(df, timestamp_col, train_end_date, prediction_length=pred_length)
        fname = "EW-7D-cov"
    elif wtype == "SW":
        window_gen = generate_sw_windows(df, timestamp_col, f"SW-{years}Y7D", prediction_length=pred_length)
        fname = f"SW-{years}Y7D-cov"
    else:
        raise ValueError(f"wtype inválido: {wtype}")

    future_cols = [id_col, timestamp_col] + covariate_cols

    for context_df, future_df in window_gen:
        future_cov_df = future_df[future_cols].copy()

        pred_df = pipeline.predict_df(
            context_df,
            future_df=future_cov_df,
            prediction_length=len(future_df),
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=id_col,
            timestamp_column=timestamp_col,
            target=target_col
        )

        y_true = future_df[target_col].values
        y_pred = pred_df["predictions"].values[:len(y_true)]
        q10 = pred_df["0.1"].values[:len(y_true)]
        q90 = pred_df["0.9"].values[:len(y_true)]

        fold_res = pd.DataFrame({
            "timestamp": future_df[timestamp_col].values,
            "ytrue": y_true,
            "yhat": y_pred,
        })
        results.append(fold_res)

    results_df = pd.concat(results, ignore_index=True)
    results_df.to_csv(f"{save_dir}/chronos_{fname}.csv", index=False)
    return fname


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
    return "full_2019"



def run_chronos():
    pipeline = load_model()
    df = load_data(csv_path)
    save_dir = "./results/chronos"
    os.makedirs(save_dir, exist_ok=True)

    fnames = []
    fnames.append(predict_with_window(pipeline, df, save_dir, wtype="EW"))
    fnames.append(predict_with_window(pipeline, df, save_dir, wtype="SW", years=1))
    fnames.append(predict_with_window(pipeline, df, save_dir, wtype="SW", years=2))
    #fnames.append(predict_full_year(pipeline, df, save_dir))

    fnames.append(predict_with_window_covariates(pipeline, df, save_dir, wtype="EW"))
    fnames.append(predict_with_window_covariates(pipeline, df, save_dir, wtype="SW", years=1))
    fnames.append(predict_with_window_covariates(pipeline, df, save_dir, wtype="SW", years=2))

    for fname in fnames:
        csv_file = f"{save_dir}/chronos_{fname}.csv"
        m = metrics_from_csv(csv_file, json_path=f"{save_dir}/{fname}_metrics.json")
        print(f"Métricas {fname}:", m)

    # Backtest igual ao repositorio
    # https://github.com/jesuinovieira/bachelor-thesis/blob/main/src/processor.py

if __name__ == "__main__":
    run_chronos()