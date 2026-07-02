import logging
from pathlib import Path

import dill
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate

from src.data.dataset_plugins import instantiate_dataset_plugin


def _format_generation_prompts(cfg, prompt_context: dict) -> None:
    if not prompt_context:
        return

    cfg.params_generation.dataset_info = cfg.params_generation.dataset_info.format(
        **prompt_context
    )

    reflection_templates = cfg.params_generation.get("list_instructions_reflection", [])
    cfg.params_generation.list_instructions_reflection = [
        prompt_template.format(**prompt_context)
        for prompt_template in reflection_templates
    ]


def _load_training_bundle(cfg) -> dict:
    dataset_cfg = cfg.get("dataset")
    if dataset_cfg is None:
        raise ValueError("Missing required config block: `dataset`.")

    plugin = instantiate_dataset_plugin(dataset_cfg)
    bundle = plugin.load_for_deft()

    required_keys = ("X_train", "y_train", "tree_filename", "prompt_context")
    missing = [key for key in required_keys if key not in bundle]
    if missing:
        raise KeyError(
            "Dataset plugin output is missing keys: "
            f"{missing}. Received keys: {list(bundle.keys())}"
        )

    return bundle


def run_experiment(cfg) -> dict:
    logging.info("Running unified DEFT experiment.")

    bundle = _load_training_bundle(cfg)
    _format_generation_prompts(cfg, bundle["prompt_context"])

    X_train = bundle["X_train"]
    y_train = bundle["y_train"]
    tree_filename = bundle["tree_filename"]

    min_samples_leaf_fraction = cfg.get("min_samples_leaf_fraction")
    if min_samples_leaf_fraction is None:
        raise ValueError("Missing required config key: `min_samples_leaf_fraction`.")
    if not (0 < float(min_samples_leaf_fraction) <= 1):
        raise ValueError(
            "`min_samples_leaf_fraction` must be in the interval (0, 1]. "
            f"Received: {min_samples_leaf_fraction}"
        )

    llm_cfg = None
    if bool(cfg.get("use_gpt_oss", False)):
        llm_cfg = cfg.get("llm_gpt_oss")
    if llm_cfg is None:
        llm_cfg = cfg.get("llm")
    if llm_cfg is None:
        raise ValueError(
            "Missing LLM config. Provide `llm` in config defaults, or provide "
            "`llm_gpt_oss` when `use_gpt_oss=true`."
        )

    min_samples_leaf = max(1, int(float(min_samples_leaf_fraction) * len(y_train)))
    logging.info(
        "Min samples leaf: %d (fraction=%.4f, n_samples=%d)",
        min_samples_leaf,
        float(min_samples_leaf_fraction),
        len(y_train),
    )

    llm_engine = instantiate(llm_cfg)

    feature_finder_cfg = cfg.get("feature_finder")
    if feature_finder_cfg is None:
        raise ValueError("Missing required config block: `feature_finder`.")
    feature_finder = instantiate(feature_finder_cfg, llm_engine=llm_engine)

    log_dir = HydraConfig.get()["runtime"]["output_dir"]
    logging.info(f"Logging dir: {log_dir}")

    splitting_criterion_cfg = cfg.get("splitting_criterion")
    if splitting_criterion_cfg is None:
        raise ValueError("Missing required config block: `splitting_criterion`.")
    splitting_criterion = instantiate(splitting_criterion_cfg)

    tree_cfg = cfg.get("tree")
    if tree_cfg is None:
        raise ValueError("Missing required config block: `tree`.")

    tree = instantiate(
        tree_cfg,
        feature_finder=feature_finder,
        splitting_criterion=splitting_criterion,
        min_samples_leaf=min_samples_leaf,
        folder_path=log_dir,
    )

    selector = instantiate(cfg.selector)

    fit_succeeded = False
    tree_path = Path(log_dir) / tree_filename
    try:
        tree.fit(X_train, y_train, **cfg.params_generation, selector=selector)
        with open(tree_path, "wb") as f:
            dill.dump(tree, f)
        tree.pretty_print()
        fit_succeeded = True
    except Exception:
        logging.exception("An error occurred during tree fitting:")

    result = {
        "status": "fit_complete" if fit_succeeded else "fit_failed",
        "tree_path": str(tree_path) if fit_succeeded else None,
    }
    logging.info(f"Run result: {result}")
    return result
