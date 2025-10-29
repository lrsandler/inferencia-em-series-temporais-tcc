import torch
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from chronos import BaseChronosPipeline, Chronos2Pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_path = "./data/dataset.csv"
target_col = "water_produced"
timestamp_col = "timestamp"
id_col = "id"
pred_length = 96  #livre escolha
pred_length = 365

save_dir = "./results"

def carregar_modelo():
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "s3://autogluon/chronos-2/",
        device_map=device
    )
    return pipeline

def preparar_dados(csv_path):
    df = pd.read_csv(csv_path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df = df.set_index(timestamp_col)
    df = df.asfreq("D")  # força índice diário contínuo
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(limit_direction="both")
    df = df.reset_index()
    df[id_col] = "main"
    freq = pd.infer_freq(df[timestamp_col])
    print(f"Frequência inferida: {freq}")
    return df

def criar_contexto_future(df, pred_length, target_col, usar_covariaveis=True):
    context_df = df.iloc[:-pred_length].copy()
    if usar_covariaveis:
        future_df = df.iloc[-pred_length:].drop(columns=[target_col]).copy()
    else:
        future_df = None
    print(f"Contexto shape: {context_df.shape}")
    if future_df is not None:
        print(f"Futuro shape (covariáveis): {future_df.shape}")
    else:
        print("Sem future_df (apenas série histórica)")
    return context_df, future_df

def prever(pipeline, context_df, future_df=None):
    pred_df = pipeline.predict_df(
        context_df,
        future_df=future_df,
        prediction_length=pred_length,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column=id_col,
        timestamp_column=timestamp_col,
        target=target_col
    )
    
    return pred_df

def plot_prediction(context_df, pred_df, target_col, timestamp_col, title="Previsão Chronos-2"):

    ts_context = context_df.set_index(timestamp_col)[target_col]
    ts_pred = pred_df.set_index(timestamp_col)
    plt.figure(figsize=(14,5))
    plt.plot(ts_context[-500:], label="Histórico", color="black", linewidth=1.8)
    plt.plot(ts_pred["predictions"], label="Previsão (mediana)", color="tab:blue", linewidth=2)
    plt.fill_between(ts_pred.index, ts_pred["0.1"], ts_pred["0.9"], alpha=0.3, color="tab:blue", label="Intervalo 10%-90%")
    plt.axvline(ts_context.index[-1], color="gray", linestyle="--", alpha=0.7)
    plt.title(title)
    plt.xlabel("Tempo")
    plt.ylabel(target_col)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def metrics(df, pred_df, target_col, pred_length):
    y_true = df.iloc[-pred_length:][target_col].values
    y_pred = pred_df["predictions"].values
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    print("\n🔹 Métricas de Desempenho:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²:   {r2:.4f}")
    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "r2": r2}

def main():
    pipeline = carregar_modelo()
    df = preparar_dados(csv_path)
    all_metrics = {}

    # com covariáveis futuras
    context_df, future_df = criar_contexto_future(df, pred_length, target_col, usar_covariaveis=True)
    pred_df_cov = prever(pipeline, context_df, future_df)
    plot_prediction(context_df, pred_df_cov, target_col, timestamp_col, title="Chronos-2 com covariáveis")
    metrics(df, pred_df_cov, target_col, pred_length)
    all_metrics["with_covariates"] = metrics(df, pred_df_cov, target_col, pred_length)

    # apenas com série histórica 
    context_df, future_df = criar_contexto_future(df, pred_length, target_col, usar_covariaveis=False)
    pred_df_hist = prever(pipeline, context_df, future_df=None)
    plot_prediction(context_df, pred_df_hist, target_col, timestamp_col, title="Chronos-2 apenas série de demanda")
    metrics(df, pred_df_hist, target_col, pred_length)
    all_metrics["historical_only"] = metrics(df, pred_df_hist, target_col, pred_length)


    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.json")

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

if __name__ == "__main__":
    main()

