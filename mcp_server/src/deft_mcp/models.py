from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RunDeftInput(BaseModel):
    """Input schema for the run_deft MCP tool."""

    dataset_csv: str = Field(
        ..., description="Path to a CSV with a sequence column and a binary target column."
    )
    output_dir: str = Field(
        ..., description="Directory to write artifacts (tree.dill, tree.json, _meta.yaml)."
    )
    sequence_column: str = Field(
        "raw_sequence", description="Name of the sequence column in the CSV."
    )
    target_column: str = Field(
        "label", description="Name of the binary (0/1) target column in the CSV."
    )
    test_size: float = Field(
        0.2, description="Held-out test fraction for the internal stratified split.", ge=0.05, le=0.5
    )
    max_depth: int = Field(
        4, description="Tree max depth.", ge=1, le=10
    )
    n_reflections: int = Field(
        0,
        description=(
            "LLM reflection iterations K. 0 = skip reflections (smoke speed). "
            "Paper-style is 20. Each reflection roughly doubles LLM cost per node."
        ),
        ge=0,
    )
    random_state: int = Field(42, description="Seed for train/test split.")
    min_samples_leaf_fraction: float = Field(
        0.01, description="Min leaf size as fraction of training set.", gt=0.0, le=1.0
    )
    target_name: str = Field(
        "the binary target",
        description="Substituted into LLM prompts as the prediction target.",
    )
    dataset_info: str = Field(
        "DNA sequences",
        description="Substituted into LLM prompts as the dataset description.",
    )
    llm_models: str = Field(
        "azure_mine",
        description="Hydra override for the LLM endpoint group (conf/llm_models/<name>.yaml).",
    )
    llm: str = Field(
        "azure_gpt5",
        description="Hydra override for the LLM engine group (conf/llm/<name>.yaml).",
    )
    n_samples_per_prompt: int = Field(
        8,
        description="LLM completions per prompt. Capped at 8 for the gpt-5.4-es deployment.",
        ge=1,
        le=20,
    )


class RunDeftOutput(BaseModel):
    """Output schema for the run_deft MCP tool."""

    tree_json_path: str
    tree_dill_path: str
    meta_yaml_path: str
    summary: dict[str, Any]
