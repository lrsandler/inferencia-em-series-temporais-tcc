import os
import json
import torch
import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from huggingface_hub import hf_hub_download
from utils.utils_predict import load_data, metrics, generate_ew_windows, generate_sw_windows
from lag_llama.gluon.estimator import LagLlamaEstimator
import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "./data/dataset.csv"
target_col = "water_produced"
timestamp_col = "timestamp"
id_col = "id"
prediction_length = 7
num_samples = 100  #  n simulações 

ckpt_path = hf_hub_download(
    repo_id="time-series-foundation-models/Lag-Llama",
    filename="lag-llama.ckpt",
    local_dir="./lag-llama",
    local_dir_use_symlinks=False,
)


def load_lag_llama_model(ckpt_path, context_length=32, num_samples=100):

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args.get("time_feat", False),
        batch_size=1,
        num_parallel_samples=num_samples,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    return predictor


def predict_fold_lagllama(predictor, context_df, future_df, num_samples=num_samples):
    
    # TODO: adicionar como logs
    if future_df is None or len(future_df) == 0:
        return None
    if context_df is None or len(context_df) < 1:
        return None

    start_timestamp = context_df[timestamp_col].iloc[0]
    target_series = context_df[target_col].values.astype(np.float32)

    dataset = ListDataset(
        [{"start": start_timestamp, "target": target_series}],
        freq="D",)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples,
    )
    forecasts = list(forecast_it)
    mean_pred = forecasts[0].samples.mean(axis=0)
    y_pred = mean_pred[-len(future_df):]
    y_true = future_df[target_col].values

    return pd.DataFrame({
        "timestamp": future_df[timestamp_col].values,
        "ytrue": y_true,
        "yhat": y_pred
    })


def predict_with_window_llama(predictor, df, save_dir, wtype="EW", years=None):
    results = []

    if wtype == "EW":
        train_end_date = "2018-12-31"
        window_gen = generate_ew_windows(df, timestamp_col, train_end_date, prediction_length)
        fname = "EW-7D"
    elif wtype == "SW":
        wtype_str = f"SW-{years}Y7D"
        window_gen = generate_sw_windows(df, timestamp_col, wtype_str, prediction_length)
        fname = wtype_str

    for context_df, future_df in window_gen:
        fold_res = predict_fold_lagllama(predictor, context_df, future_df)
        results.append(fold_res)


    results_df = pd.concat(results, ignore_index=True)
    os.makedirs(save_dir, exist_ok=True)
    results_df.to_csv(f"{save_dir}/lagllama_{fname}.csv", index=False)

    m = metrics(results_df["ytrue"].values, results_df["yhat"].values)
    with open(f"{save_dir}/{fname}_metrics.json", "w") as f:
        json.dump(m, f, indent=4)

    print(f"Métricas globais {fname}: {m}")
    return results_df


def run_lag_llama():
    df = load_data(csv_path, timestamp_col=timestamp_col, id_col=id_col, freq="D")
    save_dir = "./results/lag_llama"
    os.makedirs(save_dir, exist_ok=True)


    # EW (3Y)
    predictor_ew = load_lag_llama_model(ckpt_path=ckpt_path, context_length=1095, num_samples=num_samples)
    predict_with_window_llama(predictor_ew, df, save_dir, wtype="EW")

    # SW 1Y
    predictor_sw1 = load_lag_llama_model(ckpt_path=ckpt_path, context_length=365, num_samples=num_samples)
    predict_with_window_llama(predictor_sw1, df, save_dir, wtype="SW", years=1)

    # SW 2Y
    predictor_sw2 = load_lag_llama_model(ckpt_path=ckpt_path, context_length=730, num_samples=num_samples)
    predict_with_window_llama(predictor_sw2, df, save_dir, wtype="SW", years=2)

if __name__ == "__main__":
    run_lag_llama()

#modelo foi treinado com context length  32
#We see that when context length is increased to  256 , the model's performance drops.
#  This indicates that tuning the context length for each dataset/task is very important,
#  and the largest possible context length is not always the best in the case of Lag-Llama at the moment.