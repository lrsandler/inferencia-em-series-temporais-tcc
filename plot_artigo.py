import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.patches import ConnectionPatch

# --- Configurações de Diretórios ---
RESULTS_ROOT = "./results"
SAVE_DIR = "./plots/global"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Definição da Paleta e Modelos (Filtro Chronos-2 e Moirai) ---
PALETTE = "tab20"
PALETTE_POSITIONS = {
    "Chronos-2": 0.0,
    "Moirai":    0.3,
}

_model_names_ordered = ["Chronos-2", "Moirai"]
_cmap = plt.get_cmap(PALETTE)
_colors = [_cmap(PALETTE_POSITIONS[m]) for m in _model_names_ordered]

MODELS = {
    "Chronos-2": {"prefix": "chronos", "dir": "chronos", "color": _colors[0]},
    "Moirai":    {"prefix": "moirai",  "dir": "moirai",  "color": _colors[1]},
}

WINDOWS = ["EW-7D", "SW-1Y7D", "SW-2Y7D"]

WINDOW_LABELS = {
    "EW-7D":   "EW-7D\n",
    "SW-1Y7D": "SW-1Y7D\n",
    "SW-2Y7D": "SW-2Y7D\n"}

METRICS_LABELS = {
    "MAE":  "MAE (m³)",
    "RMSE": "RMSE (m³)",
    "MAPE": "MAPE",
    "R2":   "R²"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

def _load_metrics():
    data = {}
    for model, cfg in MODELS.items():
        data[model] = {}
        for w in WINDOWS:
            path = os.path.join(RESULTS_ROOT, cfg["dir"], f"{w}_metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    data[model][w] = json.load(f)
    return data

def _load_predictions():
    data = {}
    for model, cfg in MODELS.items():
        data[model] = {}
        for w in WINDOWS:
            path = os.path.join(RESULTS_ROOT, cfg["dir"], f"{cfg['prefix']}_{w}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=["timestamp"])
                data[model][w] = df
    return data

def _save(fig, name):
    path = os.path.join(SAVE_DIR, f"{name}.pdf")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Salvo: {path}")
    plt.close(fig)


def plot_metrics_bar():
    metrics_data = _load_metrics()
    model_names = list(MODELS.keys())
    colors = [MODELS[m]["color"] for m in model_names]
    x = np.arange(len(WINDOWS))
    width = 0.35 # Largura p 2 modelos

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax_idx, (metric_key, metric_label) in enumerate(METRICS_LABELS.items()):
        ax = axes[ax_idx]
        for i, (model, color) in enumerate(zip(model_names, colors)):
            vals = [metrics_data[model].get(w, {}).get(metric_key, np.nan) for w in WINDOWS]
            bars = ax.bar(x + i * width, vals, width, label=model, color=color,
                          edgecolor="white", linewidth=0.8, alpha=0.9)
            
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    fmt = f"{v:.2f}" if metric_key in ("MAPE", "R2") else f"{v:,.0f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * bar.get_height(),
                            fmt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([WINDOW_LABELS[w] for w in WINDOWS], fontsize=9.5)
        ax.set_ylabel(metric_label)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f"{v:.2f}" if metric_key in ("MAPE", "R2") else f"{v:,.0f}"
        ))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        if metric_key == "R2":
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)

    handles = [mpatches.Patch(color=MODELS[m]["color"], label=m) for m in model_names]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    _save(fig, "01_comparacao_metricas_barras")

def plot_timeseries_per_window():
    preds = _load_predictions()
    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=False)

    for ax, window in zip(axes, WINDOWS):
        df_ref = next((preds[m][window] for m in MODELS if window in preds.get(m, {})), None)
        if df_ref is not None:
            ax.plot(df_ref["timestamp"], df_ref["ytrue"],
                    color="black", linewidth=1.4, label="Observed", zorder=2)

        for model, cfg in MODELS.items():
            df = preds.get(model, {}).get(window)
            if df is not None:
                ax.plot(df["timestamp"], df["yhat"],
                         color=cfg["color"], linewidth=2.0, label=model, zorder=3)

        ax.set_ylabel("Water produced (m³)")
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", framealpha=0.8)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "02_previsao_vs_real_por_janela")

def plot_compact_panel():
    preds = _load_predictions()
    metrics_data = _load_metrics()
    model_names = list(MODELS.keys())
    colors = [MODELS[m]["color"] for m in model_names]

    fig, axes = plt.subplots(3, 2, figsize=(17, 12),
                             gridspec_kw={"width_ratios": [3, 1.2]})

    for row, window in enumerate(WINDOWS):
        ax_ts = axes[row, 0]
        ax_bar = axes[row, 1]

        df_ref = next((preds[m][window] for m in MODELS if window in preds.get(m, {})), None)
        if df_ref is not None:
            ax_ts.plot(df_ref["timestamp"], df_ref["ytrue"], color="black",
                       linewidth=1.4, label="Observed", zorder=2)

        for model, cfg in MODELS.items():
            df = preds.get(model, {}).get(window)
            if df is not None:
                ax_ts.plot(df["timestamp"], df["yhat"], color=cfg["color"],
                           linewidth=2.0, alpha=0.85, label=model, zorder=3)
            
        ax_ts.set_ylabel("Water produced (m³)")
        ax_ts.grid(alpha=0.2, linestyle="--")
        ax_ts.spines[["top", "right"]].set_visible(False)
        ax_ts.legend(loc="upper right", fontsize=9, framealpha=0.8)

        x = np.arange(2) 
        bar_width = 0.35
        for i, (model, color) in enumerate(zip(model_names, colors)):
            m = metrics_data[model].get(window, {})
            vals = [m.get("MAE", 0), m.get("RMSE", 0)]
            ax_bar.bar(x + i * bar_width, vals, bar_width, color=color,
                       alpha=0.85, label=model, edgecolor="white")

        r2_str = "  |  ".join([
            f"{model}: R²={metrics_data[model].get(window, {}).get('R2', 0):.3f}"
            for model in model_names
        ])
        ax_bar.text(0.5, 1.03, r2_str, ha="center", va="bottom",
                    transform=ax_bar.transAxes, fontsize=9, color="dimgray", fontweight="bold")

        ax_bar.set_xticks(x + bar_width / 2)
        ax_bar.set_xticklabels(["MAE", "RMSE"], fontsize=10)
        ax_bar.set_ylabel("m³")
        ax_bar.grid(axis="y", alpha=0.3, linestyle="--")
        ax_bar.spines[["top", "right"]].set_visible(False)

    axes[-1, 0].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "03_comparacao_compacta_janelas")


def _build_best_configs(rank_by="R2"):
    higher_is_better = rank_by == "R2"
    best = {}
    for model, cfg in MODELS.items():
        model_dir = os.path.join(RESULTS_ROOT, cfg["dir"])
        if not os.path.exists(model_dir): continue
        candidates = []
        for fname in os.listdir(model_dir):
            if not fname.endswith("_metrics.json"): continue
            window = fname.replace("_metrics.json", "")
            metrics_path = os.path.join(model_dir, fname)
            csv_path = os.path.join(model_dir, f"{cfg['prefix']}_{window}.csv")
            if os.path.exists(csv_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                if rank_by in metrics:
                    candidates.append((metrics[rank_by], window, csv_path, metrics))
        
        if candidates:
            candidates.sort(key=lambda t: t[0], reverse=higher_is_better)
            score, window, csv_path, metrics = candidates[0]
            best[model] = {"file": csv_path, "window": window, "color": cfg["color"], "metrics": metrics}
    return best

_BEST_CONFIGS = _build_best_configs(rank_by="R2")

def plot_forecast_helper(df, color, ax, title_extra="", show_ylabel=True, zoom=False):
    ax.plot(df["timestamp"], df["ytrue"], color="black", linewidth=1.2, label="Observed", zorder=1)
    ax.plot(df["timestamp"], df["yhat"], color=color, linewidth=1.4, alpha=0.88, label="Forecast", zorder=3)

    if zoom:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=12)
    if show_ylabel: ax.set_ylabel("Water produced (m³)", fontsize=10)
    ax.grid(alpha=0.22, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

def _best_worst_months(df):
    tmp = df.copy()
    tmp["period"] = tmp["timestamp"].dt.to_period("M")
    mae_monthly = tmp.groupby("period").apply(lambda g: (g["yhat"] - g["ytrue"]).abs().mean())
    return mae_monthly.idxmin(), mae_monthly.idxmax()

def _add_zoom_connectors(fig, ax_full, ax_zoom, df_period, position):
    ts0, ts1 = df_period["timestamp"].min(), df_period["timestamp"].max()
    x0, x1 = mdates.date2num(ts0), mdates.date2num(ts1)
    y0, y1 = ax_full.get_ylim()
    rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linestyle="--", edgecolor="black", alpha=0.3, zorder=4)
    ax_full.add_patch(rect)

    c_full = [(x0, y1), (x1, y1)] if position == "top" else [(x0, y0), (x1, y0)]
    c_zoom = [(0.0, 0.0), (1.0, 0.0)] if position == "top" else [(0.0, 1.0), (1.0, 1.0)]

    for (xf, yf), (xz, yz) in zip(c_full, c_zoom):
        con = ConnectionPatch(xyA=(xf, yf), coordsA=ax_full.transData, xyB=(xz, yz), coordsB=ax_zoom.transAxes,
                              linestyle="--", color="black", linewidth=0.8, alpha=0.3, zorder=5)
        fig.add_artist(con)

def plot_zoom_panel():
    model_names = list(_BEST_CONFIGS.keys())
    dfs = {m: pd.read_csv(cfg["file"], parse_dates=["timestamp"]) for m, cfg in _BEST_CONFIGS.items()}
    
    n_models = len(model_names)
    fig = plt.figure(figsize=(6 * n_models + 2, 12))
    gs = fig.add_gridspec(3, n_models, height_ratios=[1.0, 1.5, 1.0], hspace=0.2, wspace=0.25)
    
    for col, model in enumerate(model_names):
        df, cfg = dfs[model], _BEST_CONFIGS[model]
        best_p, worst_p = _best_worst_months(df)
        
        #Top (Best Month), Mid (Full), Bot (Worst Month)
        ax_top = fig.add_subplot(gs[0, col])
        ax_mid = fig.add_subplot(gs[1, col])
        ax_bot = fig.add_subplot(gs[2, col])

        plot_forecast_helper(df[df["timestamp"].dt.to_period("M") == best_p], cfg["color"], ax_top, zoom=True, show_ylabel=(col==0))
        
        plot_forecast_helper(df, cfg["color"], ax_mid, show_ylabel=(col==0))

        plot_forecast_helper(df[df["timestamp"].dt.to_period("M") == worst_p], cfg["color"], ax_bot, zoom=True, show_ylabel=(col==0))

        fig.canvas.draw()
        _add_zoom_connectors(fig, ax_mid, ax_top, df[df["timestamp"].dt.to_period("M") == best_p], "top")
        _add_zoom_connectors(fig, ax_mid, ax_bot, df[df["timestamp"].dt.to_period("M") == worst_p], "bottom")

    _save(fig, "04_zoom_melhores_piores_meses")

def plot_monthly_error_boxplot():
    model_names = list(_BEST_CONFIGS.keys())
    dfs = {m: pd.read_csv(cfg["file"], parse_dates=["timestamp"]) for m, cfg in _BEST_CONFIGS.items()}
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    n_models = len(model_names)
    group_w, box_w = 0.7, 0.3 # Ajustado para 2 modelos

    fig, ax = plt.subplots(figsize=(16, 7))
    for m_idx, model in enumerate(model_names):
        cfg = _BEST_CONFIGS[model]
        df = dfs[model].copy()
        df["abs_err"] = (df["yhat"] - df["ytrue"]).abs() / 1000
        df["month"] = df["timestamp"].dt.month
        
        positions, data_by_month, means = [], [], []
        for mo in range(1, 13):
            pos = (mo - 1) + (m_idx - (n_models - 1) / 2) * box_w
            positions.append(pos)
            vals = df[df["month"] == mo]["abs_err"].values
            data_by_month.append(vals)
            means.append(vals.mean() if len(vals) else np.nan)

        bp = ax.boxplot(data_by_month, positions=positions, widths=box_w * 0.8,
                        patch_artist=True, manage_ticks=False,
                        boxprops=dict(facecolor=cfg["color"], alpha=0.7, edgecolor="black"),
                        medianprops=dict(color="white", linewidth=1.5),
                        flierprops=dict(marker='o', markersize=4, alpha=0.3))
        
        ax.scatter(positions, means, marker="o", s=40, color="white", edgecolors=cfg["color"], zorder=5, label=f"Mean {model}")

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=12)
    ax.set_ylabel("Absolute error (×10³ m³)",fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    
    handles = [mpatches.Patch(color=_BEST_CONFIGS[m]["color"], label=m) for m in model_names]
    ax.legend(handles=handles, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    
    _save(fig, "05_erro_mensal_boxplot")

if __name__ == "__main__":
    plot_metrics_bar()
    plot_timeseries_per_window()
    plot_compact_panel()
    plot_zoom_panel()
    plot_monthly_error_boxplot()
