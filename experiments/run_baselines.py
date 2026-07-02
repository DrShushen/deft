from __future__ import annotations

import hydra
from pathlib import Path

from src.baselines.runner import run_baseline_experiment


@hydra.main(config_path="../src/conf", config_name="baselines", version_base=None)
def main(cfg) -> None:
    project_root = Path(__file__).resolve().parents[1]
    df_results, output_path = run_baseline_experiment(cfg, project_root)

    print(f"Saved baseline results to: {output_path}")
    if "accuracy" in df_results.columns and "split" in df_results.columns:
        print("Mean accuracy by split:")
        print(df_results.groupby("split")["accuracy"].mean())


if __name__ == "__main__":
    main()
