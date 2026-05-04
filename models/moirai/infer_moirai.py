import os
import json
import torch
import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from utils.utils_predict import load_data, metrics, generate_ew_windows, generate_sw_windows
import warnings

warnings.filterwarnings("ignore", message=".*pin_memory.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "./data/dataset.csv"
target_col = "water_produced"
timestamp_col = "timestamp"
id_col = "id"
pred_length = 7

def load_moirai_model(context_length=512):

    module = Moirai2Module.from_pretrained(
        "Salesforce/moirai-2.0-R-small"
    )

    model = Moirai2Forecast(
        module=module,
        prediction_length=pred_length,
        context_length=context_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0
    )

    return model.create_predictor(batch_size=1)

def predict_fold_moirai(predictor, context_df, future_df):

    dataset = ListDataset(
        [{
            "start": context_df[timestamp_col].iloc[0],
            "target": context_df[target_col].values.astype(np.float32),
        }],
        freq="D"
    )

    forecast = list(predictor.predict(dataset))[0]

    y_pred = forecast.quantile(0.5)[:len(future_df)]
    y_true = future_df[target_col].values

    return pd.DataFrame({
        "timestamp": future_df[timestamp_col].values,
        "ytrue": y_true,
        "yhat": y_pred
    })

def predict_with_window_moirai(predictor, df, save_dir, wtype="EW", years=None):

    results = []

    if wtype == "EW":
        train_end_date = "2018-12-31"
        window_gen = generate_ew_windows(df, timestamp_col, train_end_date, prediction_length=pred_length)
        fname = "EW-7D"
    elif wtype == "SW":
        window_gen = generate_sw_windows(df, timestamp_col, f"SW-{years}Y7D", prediction_length=pred_length)
        fname = f"SW-{years}Y7D"
    else:
        raise ValueError("wtype inválido")

    for context_df, future_df in window_gen:
        fold_res = predict_fold_moirai(predictor, context_df, future_df)
        results.append(fold_res)

    results_df = pd.concat(results, ignore_index=True)

    os.makedirs(save_dir, exist_ok=True)
    results_df.to_csv(f"{save_dir}/moirai_{fname}.csv", index=False)

    m = metrics(results_df["ytrue"].values, results_df["yhat"].values)
    with open(f"{save_dir}/{fname}_metrics.json", "w") as f:
        json.dump(m, f, indent=4)

    print(f"Métricas {fname}: {m}")

    return fname

def run_moirai():

    df = load_data(csv_path, timestamp_col=timestamp_col, id_col=id_col, freq="D")
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    save_dir = "./results/moirai"
    os.makedirs(save_dir, exist_ok=True)

    fnames = []

    # EW
    predictor = load_moirai_model(context_length=1095)
    fnames.append(predict_with_window_moirai(predictor, df, save_dir, "EW"))

    # SW 1Y
    predictor = load_moirai_model(context_length=365)
    fnames.append(predict_with_window_moirai(predictor, df, save_dir, "SW", years=1))

    # SW 2Y
    predictor = load_moirai_model(context_length=730)
    fnames.append(predict_with_window_moirai(predictor, df, save_dir, "SW", years=2))


if __name__ == "__main__":
    run_moirai()