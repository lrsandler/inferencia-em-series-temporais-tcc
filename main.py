import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["chronos", "lag-llama"], required=True)
    args = parser.parse_args()

    if args.model == "chronos":
        from models.chronos.infer_chronos import run_chronos
        run_chronos()

    elif args.model == "lag-llama":
        from models.lagllama.infer_llama import run_lag_llama
        run_lag_llama()

if __name__ == "__main__":
    main()
