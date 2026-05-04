import os
import argparse
from utils.utils_predict import metrics_from_csv

MODEL_CONFIGS = {
    "chronos":   {"dir": "results/chronos",   "prefix": "chronos_"},
    "lag-llama": {"dir": "results/lag_llama", "prefix": "lagllama_"},
    "moirai":    {"dir": "results/moirai",    "prefix": "moirai_"},
}

def compute_metrics(model):
    cfg = MODEL_CONFIGS[model]
    results_dir = cfg["dir"]
    prefix = cfg["prefix"]

    csv_files = [
        f for f in os.listdir(results_dir)
        if f.startswith(prefix) and f.endswith(".csv")
    ]

    if not csv_files:
        print(f"Nenhum CSV encontrado em {results_dir} com prefixo '{prefix}'")
        return

    for csv_file in sorted(csv_files):
        fname = csv_file[len(prefix):].replace(".csv", "")
        csv_path  = os.path.join(results_dir, csv_file)
        json_path = os.path.join(results_dir, f"{fname}_metrics.json")
        m = metrics_from_csv(csv_path, json_path=json_path)
        print(f"[{model}] {fname}: {m}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["chronos", "lag-llama", "moirai", "all"], required=True)
    parser.add_argument("--action", choices=["infer", "metrics"], default="infer")
    args = parser.parse_args()

    if args.action == "metrics":
        models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
        for model in models:
            compute_metrics(model)
        return

    if args.model == "chronos":
        from models.chronos.infer_chronos import run_chronos
        run_chronos()

    elif args.model == "lag-llama":
        from models.lagllama.infer_llama import run_lag_llama
        run_lag_llama()

    elif args.model == "moirai":
        from models.moirai.infer_moirai import run_moirai
        run_moirai()

if __name__ == "__main__":
    main()
