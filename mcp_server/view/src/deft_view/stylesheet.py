"""Cytoscape stylesheet + layout for the DEFT tree view."""

from __future__ import annotations

CYTOSCAPE_STYLESHEET: list[dict] = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "text-wrap": "wrap",
            "text-valign": "center",
            "text-halign": "center",
            "text-max-width": 150,
            "shape": "round-rectangle",
            "background-color": "#FFFBE6",
            "border-color": "#555",
            "border-width": 1,
            "color": "#222",
            "font-size": 11,
            "width": 150,
            "height": 70,
            "padding": "6px",
        },
    },
    {
        "selector": "node.internal",
        "style": {"background-color": "#FFF3C4"},
    },
    {
        "selector": "node.leaf",
        "style": {
            "background-color": "#FFFFFF",
            "border-color": "#888",
            "width": 110,
            "height": 60,
            "font-size": 10,
        },
    },
    {
        "selector": "node.path-green",
        "style": {"border-color": "#1b9e3a", "border-width": 3},
    },
    {
        "selector": "node.leaf.path-green",
        "style": {"background-color": "#d4f0d8"},
    },
    {
        "selector": "node.path-red",
        "style": {"border-color": "#d63a3a", "border-width": 3},
    },
    {
        "selector": "node.leaf.path-red",
        "style": {"background-color": "#f8d4d4"},
    },
    {
        "selector": "node:selected",
        "style": {"border-color": "#1f6feb", "border-width": 3},
    },
    {
        "selector": "edge",
        "style": {
            "label": "data(label)",
            "font-size": 10,
            "color": "#333",
            "text-background-color": "#fff",
            "text-background-opacity": 0.85,
            "text-background-padding": 1,
            "curve-style": "bezier",
            "target-arrow-shape": "triangle",
            "target-arrow-color": "#666",
            "line-color": "#666",
            "width": 1.5,
        },
    },
    {
        "selector": "edge.path-green",
        "style": {
            "line-color": "#1b9e3a",
            "target-arrow-color": "#1b9e3a",
            "width": 3,
        },
    },
    {
        "selector": "edge.path-red",
        "style": {
            "line-color": "#d63a3a",
            "target-arrow-color": "#d63a3a",
            "width": 3,
        },
    },
]


CYTOSCAPE_LAYOUT: dict = {
    "name": "breadthfirst",
    "directed": True,
    "padding": 20,
    "spacingFactor": 1.25,
    "roots": "[id = 'root']",
    "animate": False,
    # Fit the viewport to the tree whenever this layout runs (initial
    # mount, run switch, and -- via the layout dropdown -- layout change).
    "fit": True,
}
