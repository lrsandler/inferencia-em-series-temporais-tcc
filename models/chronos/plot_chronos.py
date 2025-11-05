import os
import pandas as pd
import matplotlib.pyplot as plt

timestamp_col = "timestamp"
target_col = "water_produced"
save_dir = "./plots"
os.makedirs(save_dir, exist_ok=True) 

def plot_real_vs_pred_series(real_df, pred_df, title, save_subdir):

    ts_real = real_df.set_index(timestamp_col)[target_col]
    ts_pred = pred_df.set_index(timestamp_col)["yhat"]

    plt.figure(figsize=(16,5))
    plt.plot(ts_real, label="Real 2019", color="black", linewidth=1.8)
    plt.plot(ts_pred, label="Previsão", color="tab:blue", linewidth=2)
    plt.title(title)
    plt.xlabel("Data")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_subdir, exist_ok=True)
    plt.savefig(os.path.join(save_subdir, f"{title}.png"), dpi=300)
    plt.show()
    plt.close()

def plot_initial_context_and_forecast(real_df, pred_df, df, window_type, save_dir):

    if window_type == "SW-1Y7D":
        context_df = df[(df[timestamp_col] >= "2018-01-01") ]
    elif window_type == "SW-2Y7D":
        context_df = df[(df[timestamp_col] >= "2017-01-01")]
    elif window_type == "EW-7D":
        context_df = df


    plt.figure(figsize=(14,5))
    plt.plot(context_df.set_index(timestamp_col)[target_col], label="Contexto", color="gray", linewidth=1.5)
    plt.plot(real_df.set_index(timestamp_col)[target_col], label="Real", color="black", linewidth=1.8)
    plt.plot(pred_df.set_index(timestamp_col)["yhat"], label="Previsão", color="tab:blue", linewidth=2)
    plt.title(f"{window_type}: Contexto e previsão")
    plt.xlabel("Data")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{window_type}_initial_context.png"), dpi=300)
    plt.show()
    plt.close()

#TODO arrumar hardcoding dos paths
#plot de comparação?

csv_path = "/home/luana/Downloads/TCC/data/dataset.csv"
df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
df = df.sort_values(timestamp_col).reset_index(drop=True)
real_2019 = df[(df[timestamp_col] >= "2019-01-01") & (df[timestamp_col] <= "2019-12-31")][[timestamp_col, target_col]]


forecast_csvs = {
    "SW-1Y7D": "/home/luana/Downloads/TCC/results/chronos/chronos_SW-1Y7D.csv",
    "SW-2Y7D": "/home/luana/Downloads/TCC/results/chronos/chronos_SW-2Y7D.csv",
    "EW-7D": "/home/luana/Downloads/TCC/results/chronos/chronos_EW-7D.csv",
}

for label, csv in forecast_csvs.items():
    df_pred = pd.read_csv(csv, parse_dates=[timestamp_col])

    plot_real_vs_pred_series(real_2019, df_pred, f"{label} vs Real 2019", save_dir)

    plot_initial_context_and_forecast(real_2019, df_pred, df, label, save_dir)

plot_real_vs_pred_series(real_2019, pd.read_csv("/home/luana/Downloads/TCC/results/chronos/chronos_full_2019.csv", parse_dates=[timestamp_col]), "Full Year vs Real 2019", save_dir)