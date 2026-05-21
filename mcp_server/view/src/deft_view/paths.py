"""Locating DEFT tree snapshots inside a palestra step.

The ``run_deft`` tool writes ``tree.json`` into a caller-chosen
``output_dir`` that lives directly under the step's ``workspace/`` (e.g.
``workspace/.deft/``, ``workspace/.deft_quick_run/``). The directory name
is not fixed, and a single step may accumulate several such directories --
one per ``run_deft`` invocation. So rather than hard-coding one path, we
discover every immediate ``workspace/*/tree.json`` and let the plugin
present a selector across them.
"""

from __future__ import annotations

from pathlib import Path

WORKSPACE_DIRNAME = "workspace"
TREE_JSON_NAME = "tree.json"


def workspace_dir(step_dir: Path) -> Path:
    """Return the agent workspace directory inside a palestra step."""
    return step_dir / WORKSPACE_DIRNAME


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def discover_tree_jsons(step_dir: Path) -> list[Path]:
    """Return all DEFT ``tree.json`` snapshots under the step's workspace.

    Globs the immediate children of ``workspace/`` (``workspace/*/tree.json``)
    -- matching how ``run_deft`` writes a snapshot into an ``output_dir``
    that is a direct child of the workspace. Results are sorted
    newest-first by the snapshot's mtime, so callers can default to the
    most recently finished run.

    Cheap and defensive: a single non-recursive glob, no JSON parsing,
    and never raises (a missing workspace or unreadable entry yields an
    empty list rather than an error).
    """
    ws = workspace_dir(step_dir)
    try:
        matches = [p for p in ws.glob(f"*/{TREE_JSON_NAME}") if p.is_file()]
    except OSError:
        return []
    matches.sort(key=_safe_mtime, reverse=True)
    return matches


def run_name(tree_json: Path) -> str:
    """Return a run's identifier: the name of the directory holding ``tree.json``.

    e.g. ``…/workspace/.deft_quick_run/tree.json`` -> ``.deft_quick_run``.
    """
    return tree_json.parent.name
