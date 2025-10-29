import torch
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tempo.models.TEMPO import TEMPO
from inspect import signature


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
csv_path = "./data/dataset.csv"
target_col = "water_produced"
pred_length = 96
step = pred_length // 3 #66% de sobreposição
window_size = 336
years_test = [2018, 2019]
save_dir = "./results"

os.makedirs(save_dir, exist_ok=True)
scaler_y = StandardScaler()

def load_data(csv_path, target_col, years_test):
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df_years = df[df["year"].isin(years_test)].reset_index(drop=True)

    target_norm = scaler_y.fit_transform(df[[target_col]]).flatten()
    target_norm_years = np.concatenate([
        scaler_y.transform(df[df["year"] == y][[target_col]]).flatten()
        for y in years_test])
        
    return df, df_years, target_norm, target_norm_years


def predict_tempo(model, target_norm, window_size=512, pred_length=96, scaler_y=None):
    
    start = len(target_norm) - window_size - pred_length
    x = target_norm[start:-pred_length]

    with torch.no_grad():
        pred = model.predict(x=x, pred_length=pred_length)

    if scaler_y is not None:
        pred = scaler_y.inverse_transform(pred.reshape(-1, 1)).flatten()
    return pred

def plot_prediction(true_values, preds, window_size, pred_length, target_col, title="Previsão com Modelo TEMPO"):
    plt.figure(figsize=(14,6))
    plt.plot(true_values, label="Real", color="black", linewidth=2)

    pred_start = len(true_values) - pred_length
    pred_axis = np.arange(pred_start, pred_start + pred_length)

    plt.plot(pred_axis, preds, label="Previsão", color="tab:blue", linewidth=2)
    plt.axvspan(pred_start - window_size, pred_start, color="gray", alpha=0.15, label="Janela de entrada")

    plt.xlabel("Tempo")
    plt.ylabel(target_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_tempo_averaged_predictions(true_values, preds, title=""):

    plt.figure(figsize=(14, 6))
    time_axis = np.arange(len(true_values))

    start_pred = true_values.shape[0] - preds.shape[0]
    end_pred = start_pred + len(preds)
    pred_axis = np.arange(start_pred, end_pred)

    if end_pred > len(true_values): #se previsao ultrapassar 
        preds = preds[:len(true_values) - start_pred]
        pred_axis = np.arange(start_pred, len(true_values))

    plt.plot(time_axis, true_values, label="Série Original", color="black", alpha=0.6)
    plt.plot(pred_axis, preds, label="Previsão TEMPO", color="tab:blue", linewidth=2)

    year_ticks = [0, 365, 730, 1095]  
    year_labels = ["2016", "2017", "2018", "2019"]  
    
    ymin, ymax = plt.ylim()
    stick_height = (ymax - ymin) * 0.02

    # for idx, label in zip(year_ticks, year_labels):
    #     plt.plot([idx, idx], [ymin, ymin + stick_height], color="black", linewidth=2)
    #     plt.text(idx, ymin + stick_height, label, horizontalalignment='center', verticalalignment='bottom', fontsize=10)

    plt.xticks()
    plt.title(title, fontsize=14)
    plt.xlabel("Índice temporal")
    plt.ylabel("Valor")
    plt.legend()
    plt.grid(True, linestyle="--")
    plt.tight_layout()
    plt.show()
   
    
    
def predict_tempo_averaged(model, target_norm, window_size, step, pred_length, scaler_y):
    total_len = len(target_norm)
    preds_sum = np.zeros(total_len)
    preds_count = np.zeros(total_len)

    last_start = total_len - window_size - pred_length

    for start in range(last_start, -1, -step):  #ao contrario 
        end = start + window_size
        x = target_norm[start:end]

        with torch.no_grad():
            pred = model.predict(x=x, pred_length=pred_length)

        pred = scaler_y.inverse_transform(pred.reshape(-1,1)).flatten()

        pred_start = end
        pred_end = end + pred_length  
        preds_sum[pred_start:pred_end] += pred
        preds_count[pred_start:pred_end] += 1

    preds_final = np.zeros_like(preds_sum)

    for i in range(len(preds_sum)):
        if preds_count[i] != 0:
            preds_final[i] = preds_sum[i] / preds_count[i]
        else:
            preds_final[i] = 0

    return preds_final, preds_count

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

def main():
    df, df_years, target_norm, target_norm_years = load_data(csv_path, target_col, years_test)
    tempo_model = TEMPO.load_pretrained_model(device=device, repo_id="Melady/TEMPO", filename="TEMPO-80M_v1.pth")
    tempo_model.eval()

    # previsões
    pred = predict_tempo(tempo_model, target_norm_years, window_size, pred_length, scaler_y)
    preds_final, preds_count = predict_tempo_averaged(tempo_model, target_norm, window_size, step, pred_length, scaler_y)
    preds_final_years, preds_count_years = predict_tempo_averaged(tempo_model, target_norm_years, window_size, 10, pred_length, scaler_y)

    # métricas
    metrics_dict_one_step = metrics(df_years[target_col].values[-pred_length:], pred)
    mask = preds_count != 0
    y_true = df[target_col].values[mask]
    y_pred = preds_final[mask]
    metrics_dict_multiple_steps = metrics(y_true, y_pred)

    mask_years = preds_count_years != 0
    y_true_years = df_years[target_col].values[mask_years]
    y_pred_years = preds_final_years[mask_years]
    metrics_dict_multiple_steps_years = metrics(y_true_years, y_pred_years)

    all_metrics = {
        "one_step": metrics_dict_one_step,
        "multi_step": metrics_dict_multiple_steps,
        "multi_step_2018_2019": metrics_dict_multiple_steps_years
    }

    # salvar em json
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nMétricas salvas em {save_dir}/metrics.json")

if __name__ == "__main__":
    main()

# Métricas de Previsão para 1 Passo:
#   MAE: 3051.746
#  RMSE: 4323.745
#  MAPE: 19.142
#    R2: -0.432

# Métricas de Previsão para Múltiplos Passos:
#   MAE: 3428.463
#  RMSE: 4302.420
#  MAPE: 26.509
#    R2: -0.536

# Métricas de Previsão para Múltiplos Passos em 2018 e 2019:
#   MAE: 3306.045
#  RMSE: 4100.920
#  MAPE: 25.914
#    R2: -0.090
