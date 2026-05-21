from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

from deft_mcp.models import RunDeftInput, RunDeftOutput
from deft_mcp.snapshot import make_snapshot

logger = logging.getLogger(__name__)

DEFT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFT_PYTHON = DEFT_REPO_ROOT / ".venv" / "bin" / "python"
DEFT_EXPERIMENT_SCRIPT = DEFT_REPO_ROOT / "experiments" / "deft_tree.py"

TREE_DILL_NAME = "tree.dill"


def _build_overrides(input: RunDeftInput, csv_path: Path, output_dir: Path) -> list[str]:
    """Build the Hydra override list passed to deft_tree.py."""
    return [
        "--config-name=config",
        "dataset=generic_csv",
        f"dataset.csv_path={csv_path}",
        f"dataset.sequence_column={input.sequence_column}",
        f"dataset.target_column={input.target_column}",
        f"dataset.test_size={input.test_size}",
        f"dataset.tree_filename={TREE_DILL_NAME}",
        f"llm_models={input.llm_models}",
        f"llm={input.llm}",
        f"max_depth={input.max_depth}",
        f"random_state={input.random_state}",
        f"min_samples_leaf_fraction={input.min_samples_leaf_fraction}",
        f"params_generation.n_reflections={input.n_reflections}",
        f"params_generation.n_samples_per_prompt={input.n_samples_per_prompt}",
        f"params_generation.target_name={input.target_name}",
        f"params_generation.dataset_info={input.dataset_info}",
        f"hydra.run.dir={output_dir}",
    ]


def _spawn_deft(overrides: list[str], output_dir: Path) -> tuple[int, str, str, float]:
    """Spawn the DEFT training subprocess. Returns (returncode, stdout, stderr, elapsed_s)."""
    if not DEFT_PYTHON.exists():
        raise FileNotFoundError(
            f"DEFT venv python not found at {DEFT_PYTHON}. "
            "Run `uv sync` from the DEFT repo root first."
        )
    if not DEFT_EXPERIMENT_SCRIPT.exists():
        raise FileNotFoundError(f"DEFT entry script not found: {DEFT_EXPERIMENT_SCRIPT}")

    cmd = [str(DEFT_PYTHON), str(DEFT_EXPERIMENT_SCRIPT), *overrides]
    logger.info("Spawning DEFT subprocess in %s", DEFT_REPO_ROOT)
    logger.debug("Command: %s", " ".join(cmd))

    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(DEFT_REPO_ROOT),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - start

    (output_dir / "deft_subprocess.stdout.log").write_text(proc.stdout or "")
    (output_dir / "deft_subprocess.stderr.log").write_text(proc.stderr or "")

    return proc.returncode, proc.stdout, proc.stderr, elapsed


def run_deft(input: RunDeftInput) -> RunDeftOutput:
    """Train a DEFT tree on the user's CSV and write a view-ready snapshot."""
    csv_path = Path(input.dataset_csv).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"dataset_csv not found: {csv_path}")

    output_dir = Path(input.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    overrides = _build_overrides(input, csv_path, output_dir)
    returncode, stdout, stderr, elapsed = _spawn_deft(overrides, output_dir)

    if returncode != 0:
        tail_err = (stderr or "")[-1500:]
        raise RuntimeError(
            f"DEFT subprocess failed with exit code {returncode}. "
            f"Logs written to {output_dir}/deft_subprocess.{{stdout,stderr}}.log. "
            f"stderr tail:\n{tail_err}"
        )

    tree_dill_path = output_dir / TREE_DILL_NAME
    if not tree_dill_path.is_file():
        raise FileNotFoundError(
            f"DEFT subprocess returned 0 but {tree_dill_path} was not written. "
            "Check deft_subprocess.{stdout,stderr}.log."
        )

    logger.info("DEFT run completed in %.1fs; writing snapshot.", elapsed)

    snapshot = make_snapshot(
        tree_path=tree_dill_path,
        csv_path=csv_path,
        sequence_column=input.sequence_column,
        target_column=input.target_column,
        test_size=input.test_size,
        random_state=input.random_state,
        output_dir=output_dir,
        input_meta=input.model_dump(),
        fit_seconds=elapsed,
    )

    return RunDeftOutput(
        tree_json_path=str(output_dir / "tree.json"),
        tree_dill_path=str(tree_dill_path),
        meta_yaml_path=str(output_dir / "_meta.yaml"),
        summary={
            "n_nodes": len(snapshot["nodes"]),
            "n_leaves": sum(1 for n in snapshot["nodes"] if n["is_leaf"]),
            "max_depth": snapshot["meta"]["max_depth"],
            "n_train": snapshot["meta"]["n_train"],
            "n_test": snapshot["meta"]["n_test"],
            "test_accuracy": snapshot["meta"].get("test_accuracy"),
            "test_auc": snapshot["meta"].get("test_auc"),
            "fit_seconds": round(elapsed, 1),
        },
    )
