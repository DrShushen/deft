from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_snapshot(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def snapshot_label(tree_json: Path) -> str:
    """Human label for the run selector: the run's directory name + key metrics.

    Loads the snapshot's ``meta`` block best-effort to append depth / test
    AUC -- so similarly-named runs stay distinguishable. Falls back to the
    bare directory name if the file can't be read or parsed.
    """
    name = tree_json.parent.name
    try:
        meta = load_snapshot(tree_json).get("meta", {})
    except Exception:
        return name
    bits: list[str] = []
    max_depth = meta.get("max_depth")
    if max_depth is not None:
        bits.append(f"depth {max_depth}")
    test_auc = meta.get("test_auc")
    if isinstance(test_auc, (int, float)):
        bits.append(f"auc {test_auc:.3f}")
    return f"{name} -- {', '.join(bits)}" if bits else name


def _first_sentence(text: str, hard_cap: int = 200) -> str:
    if not text:
        return ""
    for i, ch in enumerate(text):
        if ch in ".?!" and (i + 1 == len(text) or text[i + 1].isspace()):
            return text[: i + 1].strip()
    return text[:hard_cap].strip()


def _find_extreme_leaves(nodes: list[dict]) -> tuple[str | None, str | None]:
    leaves = [n for n in nodes if n["is_leaf"] and n.get("prediction") is not None]
    if not leaves:
        return None, None
    return (
        min(leaves, key=lambda n: n["prediction"])["id"],
        max(leaves, key=lambda n: n["prediction"])["id"],
    )


def _path_to_root(node_id: str, parent_map: dict[str, str | None]) -> list[str]:
    path: list[str] = []
    cur: str | None = node_id
    while cur is not None:
        path.append(cur)
        cur = parent_map.get(cur)
    path.reverse()
    return path


def _node_label(node: dict) -> str:
    feature = node.get("feature") or {}
    n = node.get("n", 0)
    n_pos = node.get("n_pos", 0)
    n_neg = node.get("n_neg", 0)
    if node["is_leaf"]:
        pred = node.get("prediction")
        head = f"pred = {pred:.3f}" if pred is not None else "leaf"
    else:
        head = feature.get("name") or "node"
    return f"{head}\nn = {n}\n{n_pos} / {n_neg}"


def build_cytoscape_elements(snapshot: dict) -> tuple[list[dict], dict[str, Any]]:
    """Convert a tree.json snapshot into cytoscape elements + summary."""
    nodes: list[dict] = snapshot["nodes"]
    by_id = {n["id"]: n for n in nodes}
    parent_map: dict[str, str | None] = {n["id"]: n.get("parent_id") for n in nodes}

    min_leaf_id, max_leaf_id = _find_extreme_leaves(nodes)
    green_path = _path_to_root(min_leaf_id, parent_map) if min_leaf_id else []
    red_path = _path_to_root(max_leaf_id, parent_map) if max_leaf_id else []
    green_node_set = set(green_path)
    red_node_set = set(red_path)
    green_edges = {(green_path[i], green_path[i + 1]) for i in range(len(green_path) - 1)}
    red_edges = {(red_path[i], red_path[i + 1]) for i in range(len(red_path) - 1)}

    elements: list[dict] = []

    for n in nodes:
        feature = n.get("feature") or {}
        classes: list[str] = ["leaf" if n["is_leaf"] else "internal"]
        if n["id"] in green_node_set:
            classes.append("path-green")
        if n["id"] in red_node_set:
            classes.append("path-red")

        elements.append(
            {
                "data": {
                    "id": n["id"],
                    "label": _node_label(n),
                    "tooltip": _first_sentence(feature.get("description") or ""),
                    "prediction": n.get("prediction"),
                    "n": n["n"],
                    "n_pos": n["n_pos"],
                    "n_neg": n["n_neg"],
                    "is_leaf": n["is_leaf"],
                    "depth": n["depth"],
                    "feature_name": feature.get("name", ""),
                    "feature_description": feature.get("description", ""),
                    "feature_rationale": feature.get("rationale", ""),
                    "feature_code": feature.get("code", ""),
                    "feature_threshold": feature.get("threshold"),
                    "feature_score": feature.get("score"),
                },
                "classes": " ".join(classes),
            }
        )

    for n in nodes:
        feature = n.get("feature") or {}
        threshold = feature.get("threshold")
        for side, key in (("L", "left"), ("R", "right")):
            child_id = n.get(key)
            if not child_id or child_id not in by_id:
                continue
            op = "≤" if side == "L" else ">"
            label = f"{op} {threshold:.3g}" if threshold is not None else ""
            edge_classes: list[str] = []
            if (n["id"], child_id) in green_edges:
                edge_classes.append("path-green")
            if (n["id"], child_id) in red_edges:
                edge_classes.append("path-red")
            elements.append(
                {
                    "data": {
                        "id": f"{n['id']}->{child_id}",
                        "source": n["id"],
                        "target": child_id,
                        "label": label,
                    },
                    "classes": " ".join(edge_classes),
                }
            )

    summary = {
        "n_nodes": len(nodes),
        "n_leaves": sum(1 for n in nodes if n["is_leaf"]),
        "max_depth": snapshot.get("meta", {}).get("max_depth"),
        "min_leaf_id": min_leaf_id,
        "max_leaf_id": max_leaf_id,
    }
    return elements, summary
