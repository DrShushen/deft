from __future__ import annotations

import hydra
from pathlib import Path

from src.baselines.dl_runner import run_dl_baselines_experiment


@hydra.main(config_path="../src/conf", config_name="dl_baselines", version_base=None)
def main(cfg) -> None:
    project_root = Path(__file__).resolve().parents[1]
    df_results, output_paths = run_dl_baselines_experiment(cfg, project_root)

    print("Saved deep-learning baseline results:")
    for model_name, output_path in output_paths.items():
        print(f"  {model_name}: {output_path}")

    if {"accuracy", "split", "method"}.issubset(df_results.columns):
        print("Mean accuracy by method/split:")
        print(df_results.groupby(["method", "split"])["accuracy"].mean())


if __name__ == "__main__":
    main()
