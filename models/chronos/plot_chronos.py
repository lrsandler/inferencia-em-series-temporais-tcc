import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH   = os.path.join(ROOT, "data", "dataset.csv")
RESULTS_DIR = os.path.join(ROOT, "results", "chronos")
SAVE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

TIMESTAMP_COL = "timestamp"
TARGET_COL    = "water_produced"

FORECAST_CSVS = {
    "EW-7D":      os.path.join(RESULTS_DIR, "chronos_EW-7D.csv"),
    "SW-1Y7D":    os.path.join(RESULTS_DIR, "chronos_SW-1Y7D.csv"),
    "SW-2Y7D":    os.path.join(RESULTS_DIR, "chronos_SW-2Y7D.csv"),
    "EW-7D-cov":  os.path.join(RESULTS_DIR, "chronos_EW-7D-cov.csv"),
    "SW-1Y7D-cov":os.path.join(RESULTS_DIR, "chronos_SW-1Y7D-cov.csv"),
    "SW-2Y7D-cov":os.path.join(RESULTS_DIR, "chronos_SW-2Y7D-cov.csv"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

os.makedirs(SAVE_DIR, exist_ok=True)

def _fmt_xaxis(ax, zoom=False):
    if zoom:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b/%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)


def _save(fig, filename):
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Salvo: {path}")
    plt.close(fig)


def plot_real_vs_pred(real_df, pred_df, window_type):
    ts_real = real_df.set_index(TIMESTAMP_COL)[TARGET_COL]
    ts_pred = pred_df.set_index(TIMESTAMP_COL)["yhat"]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts_real, label="Real 2019", color="black", linewidth=1.8)
    ax.plot(ts_pred, label="Previsão",  color="tab:blue", linewidth=1.8, alpha=0.85)
    ax.set_title(f"Chronos — {window_type} vs Real 2019", fontweight="bold")
    ax.set_xlabel("Data")
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    _fmt_xaxis(ax)
    fig.tight_layout()
    _save(fig, f"{window_type} vs Real 2019.png")


def plot_initial_context_and_forecast(real_df, pred_df, full_df, window_type):
    base = window_type.replace("-cov", "")
    if base == "SW-1Y7D":
        context_df = full_df[full_df[TIMESTAMP_COL] >= "2018-01-01"]
    elif base == "SW-2Y7D":
        context_df = full_df[full_df[TIMESTAMP_COL] >= "2017-01-01"]
    else:  
        context_df = full_df

    ts_ctx  = context_df.set_index(TIMESTAMP_COL)[TARGET_COL]
    ts_real = real_df.set_index(TIMESTAMP_COL)[TARGET_COL]
    ts_pred = pred_df.set_index(TIMESTAMP_COL)["yhat"]

    # Remove sobreposição do contexto com 2019
    ts_ctx = ts_ctx[ts_ctx.index < ts_real.index.min()]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts_ctx,  label="Contexto histórico", color="gray",     linewidth=1.2, alpha=0.75)
    ax.plot(ts_real, label="Real 2019",           color="black",    linewidth=1.8)
    ax.plot(ts_pred, label="Previsão",             color="tab:blue", linewidth=1.8, alpha=0.85)
    ax.axvline(ts_real.index.min(), color="black", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_title(f"Chronos — {window_type}: Contexto e Previsão", fontweight="bold")
    ax.set_xlabel("Data")
    ax.set_ylabel(TARGET_COL)
    ax.legend()
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    _fmt_xaxis(ax)
    fig.tight_layout()
    _save(fig, f"{window_type}_initial_context.png")


def plot_forecast_with_interval(real_df, pred_df, full_df, window_type, history_length=256):
    """
    Estilo notebook Chronos: contexto histórico + real + previsão + intervalo de predição.

    Usa colunas q10/q90 do CSV (salvas pela inferência) para o fill_between cinza/lavanda.
    Se as colunas não existirem, plota sem intervalo.
    """
    base = window_type.replace("-cov", "")
    if base == "SW-1Y7D":
        context_df = full_df[full_df[TIMESTAMP_COL] >= "2018-01-01"]
    elif base == "SW-2Y7D":
        context_df = full_df[full_df[TIMESTAMP_COL] >= "2017-01-01"]
    else:  # EW-7D: todo o histórico
        context_df = full_df

    ts_ctx  = context_df.set_index(TIMESTAMP_COL)[TARGET_COL]
    ts_real = real_df.set_index(TIMESTAMP_COL)[TARGET_COL]
    ts_pred = pred_df.set_index(TIMESTAMP_COL)

    # Remove sobreposição e limita contexto a history_length pontos
    ts_ctx = ts_ctx[ts_ctx.index < ts_real.index.min()].iloc[-history_length:]

    has_interval = {"q10", "q90"}.issubset(ts_pred.columns) and ts_pred["q10"].notna().any()

    fig, ax = plt.subplots(figsize=(16, 5))

    ax.plot(ts_ctx,  label="Contexto histórico",  color="xkcd:azure",       linewidth=1.2, alpha=0.80)
    ax.plot(ts_real, label="Real 2019",            color="xkcd:grass green", linewidth=1.8)
    ax.plot(ts_pred["yhat"], label="Previsão",     color="xkcd:violet",      linewidth=1.8, alpha=0.88)

    if has_interval:
        ax.fill_between(
            ts_pred.index,
            ts_pred["q10"],
            ts_pred["q90"],
            alpha=0.35,
            color="xkcd:light lavender",
            label="Intervalo de predição (10%-90%)",
        )

    # Linha vertical marcando início da previsão
    ax.axvline(ts_real.index.min(), color="black", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_title(f"Chronos — {window_type}: Contexto, Previsão e Intervalo", fontweight="bold")
    ax.set_xlabel("Data")
    ax.set_ylabel(TARGET_COL)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    _fmt_xaxis(ax)
    fig.tight_layout()
    _save(fig, f"{window_type}_forecast_interval.png")

if __name__ == "__main__":
    df      = pd.read_csv(DATA_PATH, parse_dates=[TIMESTAMP_COL])
    df      = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    real_2019 = df[
        (df[TIMESTAMP_COL] >= "2019-01-01") &
        (df[TIMESTAMP_COL] <= "2019-12-31")
    ][[TIMESTAMP_COL, TARGET_COL]]

    for label, csv_path in FORECAST_CSVS.items():
        if not os.path.exists(csv_path):
            print(f"  [aviso] arquivo não encontrado: {csv_path}")
            continue
        df_pred = pd.read_csv(csv_path, parse_dates=[TIMESTAMP_COL])
        plot_real_vs_pred(real_2019, df_pred, label)
        plot_initial_context_and_forecast(real_2019, df_pred, df, label)
        plot_forecast_with_interval(real_2019, df_pred, df, label)
