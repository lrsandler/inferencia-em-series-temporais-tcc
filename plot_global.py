import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

RESULTS_ROOT = "./results"
SAVE_DIR = "./plots/global"
os.makedirs(SAVE_DIR, exist_ok=True)

PALETTE = "tab20"

PALETTE_POSITIONS = {
    "Chronos-2": 0.0,
    "Lag-Llama": 0.1,
    "Moirai":    0.3,
}

_model_names_ordered = ["Chronos-2", "Lag-Llama", "Moirai"]
_cmap   = plt.get_cmap(PALETTE)
_colors = [_cmap(PALETTE_POSITIONS[m]) for m in _model_names_ordered]

MODELS = {
    "Chronos-2":  {"prefix": "chronos",  "dir": "chronos",   "color": _colors[0]},
    "Lag-Llama":  {"prefix": "lagllama", "dir": "lag_llama", "color": _colors[1]},
    "Moirai":     {"prefix": "moirai",   "dir": "moirai",    "color": _colors[2]},
}

WINDOWS = ["EW-7D", "SW-1Y7D", "SW-2Y7D"]

WINDOW_LABELS = {
    "EW-7D":   "EW-7D\n",
    "SW-1Y7D": "SW-1Y7D\n",
    "SW-2Y7D": "SW-2Y7D\n",
}

METRICS_LABELS = {
    "MAE":  "MAE (m³)",
    "RMSE": "RMSE (m³)",
    "MAPE": "MAPE",
    "R2":   "R²",
}


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

def _load_metrics():
    """Retorna dict[model][window] = {MAE, RMSE, MAPE, R2}"""
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
    """Retorna dict[model][window] = DataFrame(timestamp, ytrue, yhat)"""
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
    width = 0.25

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax_idx, (metric_key, metric_label) in enumerate(METRICS_LABELS.items()):
        ax = axes[ax_idx]
        for i, (model, color) in enumerate(zip(model_names, colors)):
            vals = [metrics_data[model].get(w, {}).get(metric_key, np.nan) for w in WINDOWS]
            bars = ax.bar(x + i * width, vals, width, label=model, color=color,
                          edgecolor="white", linewidth=0.8, alpha=0.9)
            # anotações em cima das barras
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    fmt = f"{v:.2f}" if metric_key in ("MAPE", "R2") else f"{v:,.0f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * bar.get_height(),
                            fmt, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x + width)
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
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    _save(fig, "comparacao_metricas_chronos2_lag_llama_moirai")


def plot_timeseries_per_window():
    preds = _load_predictions()

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), sharex=False)

    for ax, window in zip(axes, WINDOWS):
        # ytrue primeiro para ficar abaixo das previsões
        df_ref = next((preds[m][window] for m in MODELS if window in preds.get(m, {})), None)
        if df_ref is not None:
            ax.plot(df_ref["timestamp"], df_ref["ytrue"],
                    color="black", linewidth=1.4, label="Observed", zorder=2)

        for model, cfg in MODELS.items():
            df = preds.get(model, {}).get(window)
            if df is None:
                continue
            ax.plot(df["timestamp"], df["yhat"],
                     color=cfg["color"], linewidth=2.0, label=model, zorder=3)
            

        ax.set_ylabel("Water produced (m³)")
        ax.grid(alpha=0.25, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", framealpha=0.8)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "previsao_vs_real_por_janela")



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

        # --- coluna esquerda: série temporal ---
        # ytrue primeiro para ficar abaixo das previsões
        df_ref = next((preds[m][window] for m in MODELS if window in preds.get(m, {})), None)
        if df_ref is not None:
            ax_ts.plot(df_ref["timestamp"], df_ref["ytrue"], color="black",
                       linewidth=1.4, label="Observed", zorder=2)

        for model, cfg in MODELS.items():
            df = preds.get(model, {}).get(window)
            if df is None:
                continue
            ax_ts.plot(df["timestamp"], df["yhat"], color=cfg["color"],
                       linewidth=2.0, alpha=0.85, label=model, zorder=3)
            
        ax_ts.set_ylabel("Water produced (m³)")
        ax_ts.grid(alpha=0.2, linestyle="--")
        ax_ts.spines[["top", "right"]].set_visible(False)
        ax_ts.legend(loc="upper right", fontsize=9, framealpha=0.8)
        if row < 2:
            ax_ts.set_xlabel("")

        x = np.arange(2)  # MAE e RMSE
        bar_width = 0.22
        for i, (model, color) in enumerate(zip(model_names, colors)):
            m = metrics_data[model].get(window, {})
            vals = [m.get("MAE", 0), m.get("RMSE", 0)]
            ax_bar.bar(x + i * bar_width, vals, bar_width, color=color,
                       alpha=0.85, label=model, edgecolor="white")

        # R2 como texto
        r2_str = "  ".join([
            f"{model[:3]}: R²={metrics_data[model].get(window, {}).get('R2', '—')}"
            for model in model_names
        ])
        ax_bar.text(0.5, 1.03, r2_str, ha="center", va="bottom",
                    transform=ax_bar.transAxes, fontsize=8, color="dimgray")

        ax_bar.set_xticks(x + bar_width)
        ax_bar.set_xticklabels(["MAE", "RMSE"], fontsize=10)
        ax_bar.set_ylabel("m³")
        ax_bar.grid(axis="y", alpha=0.3, linestyle="--")
        ax_bar.spines[["top", "right"]].set_visible(False)

    axes[-1, 0].set_xlabel("Date")
    fig.tight_layout()
    _save(fig, "comparacao_modelos_por_janela")



def _build_best_configs(rank_by="R2"):
    higher_is_better = rank_by == "R2"
    best = {}

    for model, cfg in MODELS.items():
        model_dir = os.path.join(RESULTS_ROOT, cfg["dir"])
        candidates = []

        for fname in os.listdir(model_dir):
            if not fname.endswith("_metrics.json"):
                continue

            window = fname.replace("_metrics.json", "")   # ex: "EW-7D-cov"
            metrics_path = os.path.join(model_dir, fname)
            csv_path     = os.path.join(model_dir, f"{cfg['prefix']}_{window}.csv")

            if not os.path.exists(csv_path):
                continue

            with open(metrics_path) as f:
                metrics = json.load(f)

            if rank_by not in metrics:
                continue

            candidates.append((metrics[rank_by], window, csv_path, metrics))

        if not candidates:
            continue

        candidates.sort(key=lambda t: t[0], reverse=higher_is_better)
        score, window, csv_path, metrics = candidates[0]

        # Monta label legível: "EW-7D-cov" → "EW-7D (c/ cov.)"
        label_window = window.replace("-cov", " (c/ cov.)")
        label = f"{model}  {label_window}"

        best[model] = {
            "file":    csv_path,
            "label":   label,
            "color":   cfg["color"],
            "metrics": metrics,
        }
        print(f"  [{model}] melhor janela: {window}  ({rank_by}={score})")

    return best


_BEST_CONFIGS = _build_best_configs(rank_by="R2")

def plot_forecast(df, color, ax, title_extra="", show_ylabel=True, zoom=False):

    import matplotlib.dates as mdates

    ax.plot(df["timestamp"], df["ytrue"],
            color="black", linewidth=1.2, label="Observed", zorder=1)

    if {"q10", "q90"}.issubset(df.columns):
        ax.fill_between(
            df["timestamp"], df["q10"], df["q90"],
            alpha=0.25, color=color, label="80% Interval", zorder=2,
        )

    ax.plot(df["timestamp"], df["yhat"],
            color=color, linewidth=1.4, alpha=0.88, label="Forecast", zorder=3)

    if zoom:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    if show_ylabel:
        ax.set_ylabel("Water produced (m³)", fontsize=9)
    ax.grid(alpha=0.22, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    if title_extra:
        ax.set_title(title_extra, fontsize=9)


def _best_worst_months(df):
    """Retorna (melhor_period, pior_period) baseado em MAE mensal."""
    tmp = df.copy()
    tmp["period"] = tmp["timestamp"].dt.to_period("M")
    mae_monthly = (
        tmp.groupby("period")[["yhat", "ytrue"]]
        .apply(lambda g: (g["yhat"] - g["ytrue"]).abs().mean())
    )
    return mae_monthly.idxmin(), mae_monthly.idxmax()


def _zoom_df(df, period):
    """Filtra o DataFrame para um único mês (pd.Period)."""
    return df[df["timestamp"].dt.to_period("M") == period].copy()


def _month_label(period):
    """Formata um pd.Period como 'Jan/2019'."""
    return period.to_timestamp().strftime("%b/%Y")


def _add_zoom_box_and_connectors(fig, ax_full, ax_zoom, df_period, position):
    
    import matplotlib.dates as mdates
    from matplotlib.patches import ConnectionPatch, FancyBboxPatch

    ts0 = df_period["timestamp"].min()
    ts1 = df_period["timestamp"].max()
    x0 = mdates.date2num(ts0)
    x1 = mdates.date2num(ts1)

    # y limits do eixo completo
    ax_full.autoscale_view()
    y0, y1 = ax_full.get_ylim()

    rect = plt.Rectangle(
        (x0, y0), x1 - x0, y1 - y0,
        fill=False, linestyle="--", edgecolor="black",
        linewidth=1.0, transform=ax_full.transData, zorder=4, clip_on=False,
    )
    ax_full.add_patch(rect)

    # Dois conectores: canto esquerdo e direito do rect → cantos do ax_zoom
    if position == "top":
        corners_full = [(x0, y1), (x1, y1)]   # topo do rect
        corners_zoom = [(0.0, 0.0), (1.0, 0.0)]  # base do ax_zoom
    else:
        corners_full = [(x0, y0), (x1, y0)]   # base do rect
        corners_zoom = [(0.0, 1.0), (1.0, 1.0)]  # topo do ax_zoom

    for (xf, yf), (xz, yz) in zip(corners_full, corners_zoom):
        con = ConnectionPatch(
            xyA=(xf, yf), coordsA=ax_full.transData,
            xyB=(xz, yz), coordsB=ax_zoom.transAxes,
            linestyle="--", color="black", linewidth=0.8,
            alpha=0.55, zorder=5, clip_on=False,
        )
        fig.add_artist(con)


def plot_zoom_panel():
    
    import matplotlib.dates as mdates

    model_names = list(_BEST_CONFIGS.keys())
    dfs = {
        m: pd.read_csv(cfg["file"], parse_dates=["timestamp"])
        for m, cfg in _BEST_CONFIGS.items()
    }

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1.0, 1.6, 1.0],
        hspace=0.08, wspace=0.28,
    )
    axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

    for col, model in enumerate(model_names):
        df   = dfs[model]
        cfg  = _BEST_CONFIGS[model]
        color = cfg["color"]
        m    = cfg["metrics"]

        best_p, worst_p = _best_worst_months(df)
        df_best  = _zoom_df(df, best_p)
        df_worst = _zoom_df(df, worst_p)
        mae_best  = (df_best["yhat"]  - df_best["ytrue"]).abs().mean()
        mae_worst = (df_worst["yhat"] - df_worst["ytrue"]).abs().mean()

        show_y = col == 0
        ax_top = axes[0][col]
        plot_forecast(df_best, color, ax_top, show_ylabel=show_y, zoom=True)
        ax_top.tick_params(labelbottom=False)
        ax_top.set_xlabel("")

        ax_mid = axes[1][col]
        plot_forecast(df, color, ax_mid, show_ylabel=show_y)
        ax_mid.tick_params(labelbottom=False)
        ax_mid.set_xlabel("")

        ax_bot = axes[2][col]
        plot_forecast(df_worst, color, ax_bot, show_ylabel=show_y, zoom=True)
        ax_bot.set_xlabel("Date", fontsize=9)

        # Caixas + conectores (desenhados APÓS os plots para ter ylim correto)
        fig.canvas.draw()  # força cálculo dos ylim
        _add_zoom_box_and_connectors(fig, ax_mid, ax_top, df_best,  position="top")
        _add_zoom_box_and_connectors(fig, ax_mid, ax_bot, df_worst, position="bottom")

    ax_mid_center = axes[1][1]
    legend_handles = [
        Line2D([0], [0], color="black", linewidth=1.6, label="Observed"),
    ] + [
        Line2D([0], [0], color=_BEST_CONFIGS[m]["color"], linewidth=1.6,
               alpha=0.88, label=m)
        for m in model_names
    ]
    ax_mid_center.legend(
        handles=legend_handles, loc="upper center",
        bbox_to_anchor=(0.5, 1.0), ncol=4,
        fontsize=9, framealpha=0.85,
    )

    _save(fig, "melhor_config_melhores_piores_meses")



def plot_monthly_error_boxplot():
    
    model_names = list(_BEST_CONFIGS.keys())
    dfs = {
        m: pd.read_csv(cfg["file"], parse_dates=["timestamp"])
        for m, cfg in _BEST_CONFIGS.items()
    }

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    n_months  = 12
    n_models  = len(model_names)
    group_w   = 0.75
    box_w     = group_w / n_models

    fig, ax = plt.subplots(figsize=(16, 6))

    for m_idx, model in enumerate(model_names):
        cfg = _BEST_CONFIGS[model]
        df  = dfs[model].copy()
        df["abs_err"] = (df["yhat"] - df["ytrue"]).abs() / 1000  # ×10³ m³
        df["month"]   = df["timestamp"].dt.month

        positions      = []
        data_by_month  = []
        means          = []

        for mo in range(1, n_months + 1):
            offset = (m_idx - (n_models - 1) / 2) * box_w
            pos    = (mo - 1) + offset
            positions.append(pos)
            vals = df[df["month"] == mo]["abs_err"].values
            data_by_month.append(vals)
            means.append(vals.mean() if len(vals) else np.nan)

        bp = ax.boxplot(
            data_by_month,
            positions=positions,
            widths=box_w * 0.82,
            patch_artist=True,
            manage_ticks=False,
            medianprops=dict(color="white", linewidth=2.0),
            whiskerprops=dict(linewidth=1.1, color="black"),
            capprops=dict(linewidth=1.1, color="black"),
            boxprops=dict(linewidth=1.1),
            flierprops=dict(
                marker="D", markersize=3.5, alpha=0.55,
                markerfacecolor=cfg["color"], markeredgewidth=0.4,
                markeredgecolor="gray",
            ),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(cfg["color"])
            patch.set_alpha(0.80)

        # Marcador de média (círculo branco com borda colorida)
        ax.scatter(
            positions, means,
            marker="o", s=28, zorder=6,
            color="white", edgecolors=cfg["color"], linewidths=1.5,
        )

    ax.set_xticks(range(n_months))
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Absolute error (×10³ m³)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.grid(axis="y", alpha=0.30, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.6, n_months - 0.4)

    handles = [
        mpatches.Patch(facecolor=_BEST_CONFIGS[m]["color"], alpha=0.80, label=m)
        for m in model_names
    ]
    handles.append(
        Line2D([0], [0], marker="o", color="gray", linestyle="None",
               markerfacecolor="white", markeredgecolor="gray",
               markersize=6, label="Mean")
    )
    ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.85)

    fig.tight_layout()
    _save(fig, "erro_absoluto_mensal_melhor_config")


if __name__ == "__main__":
    plot_metrics_bar()
    plot_timeseries_per_window()
    plot_compact_panel()
    plot_zoom_panel()
    plot_monthly_error_boxplot()
