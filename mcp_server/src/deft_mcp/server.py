from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from deft_mcp.models import RunDeftInput, RunDeftOutput
from deft_mcp.runner import run_deft

logger = logging.getLogger(__name__)

mcp = FastMCP("deft")


@mcp.tool()
def run_deft_tool(input: RunDeftInput) -> RunDeftOutput:
    """Train a DEFT decision tree on a CSV dataset.

    DEFT is an LLM-augmented top-down decision tree builder for DNA sequences:
    at every node it asks an LLM to propose candidate features (e.g.
    "GC content in 20-49", "TATA-box variant in 0-30"), scores each by Gini
    reduction on the node's data, and uses the best as the split. The result
    is a shallow, interpretable tree whose splits are biologically meaningful
    rather than per-position one-hot conditions.

    Input contract:
      - `dataset_csv` and `output_dir` are required.
      - The CSV must contain `sequence_column` (string DNA, e.g. ACGT...)
        and `target_column` (binary 0/1). Other columns are ignored.
      - An internal stratified train/test split is performed with the
        configured `test_size` and `random_state`.

    Outputs written into `output_dir`:
      - `tree.json`   -- view-friendly tree snapshot with per-node n / n_pos /
                         n_neg and per-leaf prediction; consumed by the
                         palestra view plugin.
      - `tree.dill`   -- raw fitted AdaptiveDecisionTree (requires DEFT to
                         re-load).
      - `_meta.yaml`  -- run metadata (config, dataset stats, test metrics).

    Defaults are tuned for a quick smoke (max_depth=4, n_reflections=0,
    n_samples_per_prompt=8). Raise n_reflections to 20 for paper-style runs;
    expect a ~20x cost / wall-time increase.
    """
    return run_deft(input)
