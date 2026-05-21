"""Dash layout pieces for the DEFT view plugin."""

from __future__ import annotations

from typing import Any

import dash_cytoscape as cyto
import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from deft_view.stylesheet import CYTOSCAPE_LAYOUT

NAME = "deft"

ID_CYTOSCAPE = f"plugin-{NAME}-cytoscape"
ID_DETAIL = f"plugin-{NAME}-detail"
ID_SUMMARY = f"plugin-{NAME}-summary"
ID_SNAPSHOT_STORE = f"plugin-{NAME}-snapshot-store"
ID_RUN_SELECTOR = f"plugin-{NAME}-run-selector"
ID_LAYOUT_DROPDOWN = f"plugin-{NAME}-layout-dropdown"
ID_FIT_BUTTON = f"plugin-{NAME}-fit-button"

# Curated to the layouts that make sense for a (binary) tree -- the
# hierarchical ones. Force-directed / geometric layouts (cose, circle,
# grid, concentric) would scramble the parent->child structure, so they
# are deliberately excluded. Dagre (top-down) is the default -- its
# layered spacing keeps binary trees tidy and non-overlapping;
# breadthfirst stays available, and dagre left-right suits deep trees
# that run off-screen vertically.
DEFAULT_LAYOUT_KEY = "dagre-TB"

LAYOUT_OPTIONS: list[dict[str, str]] = [
    {"label": "Breadthfirst", "value": "breadthfirst"},
    {"label": "Dagre (top-down)", "value": "dagre-TB"},
    {"label": "Dagre (left-right)", "value": "dagre-LR"},
]


def cytoscape_layout(value: str) -> dict:
    """Map a :data:`LAYOUT_OPTIONS` value to a cytoscape layout config.

    Unknown values fall back to the default (``breadthfirst``). The
    ``breadthfirst`` entry reuses :data:`CYTOSCAPE_LAYOUT` so the dropdown's
    default exactly matches the canvas's initial layout.
    """
    # ``fit: True`` re-fits the viewport every time the layout runs, so
    # switching layout (or run) always frames the whole tree.
    return {
        "breadthfirst": CYTOSCAPE_LAYOUT,
        "dagre-TB": {
            "name": "dagre",
            "rankDir": "TB",
            "spacingFactor": 1.1,
            "padding": 20,
            "animate": False,
            "fit": True,
        },
        "dagre-LR": {
            "name": "dagre",
            "rankDir": "LR",
            "spacingFactor": 1.1,
            "padding": 20,
            "animate": False,
            "fit": True,
        },
    }.get(value, CYTOSCAPE_LAYOUT)


def build_controls_overlay() -> Any:
    """Layout-select + Fit-button overlay, anchored top-left over the canvas.

    Mirrors palestra view's own graph-control overlays: absolutely
    positioned (zero layout height) with a Mantine body background so it
    stays readable over the cytoscape canvas in light and dark modes.
    """
    return dmc.Group(
        [
            dmc.Tooltip(
                dmc.Select(
                    id=ID_LAYOUT_DROPDOWN,
                    data=LAYOUT_OPTIONS,
                    value=DEFAULT_LAYOUT_KEY,
                    size="xs",
                    w=150,
                    allowDeselect=False,
                    comboboxProps={"withinPortal": True},
                ),
                label="Tree layout",
                position="bottom",
                withArrow=True,
                openDelay=300,
            ),
            dmc.Tooltip(
                dmc.ActionIcon(
                    DashIconify(icon="mdi:fit-to-screen"),
                    id=ID_FIT_BUTTON,
                    size="md",
                    variant="subtle",
                    **{"aria-label": "Fit tree to view"},
                ),
                label="Fit tree to view",
                position="bottom",
                withArrow=True,
                openDelay=300,
            ),
        ],
        gap="xs",
        align="center",
        style={
            "position": "absolute",
            "top": "4px",
            "left": "8px",
            "zIndex": 2,
            "padding": "2px 4px",
            "background": "var(--mantine-color-body)",
            "borderRadius": "var(--mantine-radius-sm)",
        },
    )


def _placeholder_detail() -> Any:
    return dmc.Stack(
        [
            dmc.Title("DEFT node details", order=4),
            dmc.Text(
                "Click a node in the tree to see its full feature description, "
                "rationale, generated Python code, and split score.",
                size="sm",
                c="dimmed",
            ),
        ],
        gap="xs",
    )


def build_summary_bar(summary: dict, meta: dict) -> Any:
    """Compact top bar with run summary (nodes/leaves/depth + test metrics)."""
    pills = []

    def pill(label: str, value: Any) -> Any:
        return dmc.Badge(f"{label}: {value}", color="gray", variant="light")

    pills.append(pill("nodes", summary.get("n_nodes")))
    pills.append(pill("leaves", summary.get("n_leaves")))
    pills.append(pill("max depth", summary.get("max_depth")))

    n_train = meta.get("n_train")
    n_test = meta.get("n_test")
    test_acc = meta.get("test_accuracy")
    test_auc = meta.get("test_auc")

    if n_train is not None:
        pills.append(pill("train", n_train))
    if n_test is not None:
        pills.append(pill("test", n_test))
    if test_acc is not None:
        pills.append(pill("test acc", f"{test_acc:.3f}"))
    if test_auc is not None:
        pills.append(pill("test auc", f"{test_auc:.3f}"))

    return dmc.Group(pills, gap="xs", wrap="wrap")


def build_run_selector(options: list[dict], value: str | None) -> Any:
    """Dropdown selecting which DEFT run (workspace output dir) to visualize.

    ``options`` is a list of ``{"value": <tree.json path>, "label": <text>}``;
    ``value`` is the path string of the currently shown run. The selector is
    always rendered (even for a single run) so it doubles as a label of which
    output directory the displayed tree came from.
    """
    return dmc.Select(
        id=ID_RUN_SELECTOR,
        data=options,
        value=value,
        allowDeselect=False,
        searchable=len(options) > 8,
        size="xs",
        placeholder="Select a DEFT run",
        style={"minWidth": "280px"},
        comboboxProps={"withinPortal": True},
    )


def build_cytoscape(elements: list[dict], stylesheet: list[dict], layout: dict) -> Any:
    return cyto.Cytoscape(
        id=ID_CYTOSCAPE,
        elements=elements,
        stylesheet=stylesheet,
        layout=layout,
        style={"width": "100%", "height": "100%", "min-height": "500px"},
        autoungrabify=False,
        userPanningEnabled=True,
        userZoomingEnabled=True,
    )


def render_error(message: str) -> Any:
    return dmc.Alert(message, title="DEFT view error", color="red")


def render_detail(tap_data: dict | None) -> Any:
    """Render the side panel for a tapped node."""
    if not tap_data:
        return _placeholder_detail()

    is_leaf = bool(tap_data.get("is_leaf"))
    n = tap_data.get("n")
    n_pos = tap_data.get("n_pos")
    n_neg = tap_data.get("n_neg")
    depth = tap_data.get("depth")
    prediction = tap_data.get("prediction")

    blocks: list[Any] = []
    blocks.append(dmc.Title(tap_data.get("feature_name") or ("leaf" if is_leaf else "node"), order=4))

    stat_pills = [
        dmc.Badge(f"depth: {depth}", variant="light", color="gray"),
        dmc.Badge(f"n: {n}", variant="light", color="gray"),
        dmc.Badge(f"n_pos: {n_pos}", variant="light", color="grape"),
        dmc.Badge(f"n_neg: {n_neg}", variant="light", color="blue"),
    ]
    if prediction is not None:
        stat_pills.append(dmc.Badge(f"pred: {prediction:.4f}", variant="filled", color="indigo"))
    blocks.append(dmc.Group(stat_pills, gap="xs", wrap="wrap"))

    if is_leaf:
        blocks.append(
            dmc.Text(
                "This is a leaf. Its prediction is the mean target of the routed training samples.",
                size="sm",
                c="dimmed",
            )
        )
        return dmc.Stack(blocks, gap="sm")

    threshold = tap_data.get("feature_threshold")
    score = tap_data.get("feature_score")
    split_pills = []
    if threshold is not None:
        split_pills.append(dmc.Badge(f"threshold: {threshold:.4g}", variant="light", color="cyan"))
    if score is not None:
        split_pills.append(dmc.Badge(f"score: {score:.4f}", variant="light", color="green"))
    if split_pills:
        blocks.append(dmc.Group(split_pills, gap="xs", wrap="wrap"))

    description = tap_data.get("feature_description") or ""
    if description:
        blocks.append(dmc.Divider(label="Description", labelPosition="left"))
        blocks.append(dmc.Text(description, size="sm"))

    rationale = tap_data.get("feature_rationale") or ""
    if rationale:
        blocks.append(dmc.Divider(label="Rationale", labelPosition="left"))
        blocks.append(dmc.Text(rationale, size="sm", c="dimmed"))

    code = tap_data.get("feature_code") or ""
    if code:
        blocks.append(dmc.Divider(label="Generated Python", labelPosition="left"))
        blocks.append(
            dmc.CodeHighlight(
                code=code,
                language="python",
                withCopyButton=True,
            )
        )

    return dmc.Stack(blocks, gap="sm")


def run_view_layout(
    run_selector: Any,
    summary_bar: Any,
    cytoscape: Any,
    detail_initial: Any,
    snapshot_store_data: dict | None,
) -> Any:
    """Two-pane layout: run selector + cytoscape on the left, detail on the right.

    The ``run_selector`` lives in the toolbar and is owned by the host's
    re-render path (its value is seeded from host_state on every mount), so
    the run-change callback never rewrites it -- avoiding a feedback loop.
    The ``summary_bar`` sits in its own ``ID_SUMMARY`` container, which the
    run-change callback swaps out when the user picks a different run.
    """
    toolbar = html.Div(
        [
            run_selector,
            html.Div(summary_bar, id=ID_SUMMARY, style={"flex": "1 1 auto", "minWidth": 0}),
        ],
        style={
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "center",
            "gap": "12px",
            "flexWrap": "wrap",
            "padding": "6px 8px",
        },
    )
    return html.Div(
        [
            dcc.Store(id=ID_SNAPSHOT_STORE, data=snapshot_store_data),
            html.Div(
                [
                    html.Div(
                        [
                            toolbar,
                            html.Div(
                                [build_controls_overlay(), cytoscape],
                                # ``position: relative`` anchors the absolutely
                                # positioned controls overlay to this canvas box.
                                style={
                                    "position": "relative",
                                    "flex": "1 1 auto",
                                    "minHeight": 0,
                                    "display": "flex",
                                },
                            ),
                        ],
                        style={
                            "flex": "1 1 60%",
                            "minWidth": "300px",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                    ),
                    html.Div(className="panel-splitter"),
                    html.Div(
                        detail_initial,
                        id=ID_DETAIL,
                        style={
                            "flex": "1 1 40%",
                            "minWidth": "260px",
                            "padding": "10px",
                            "overflowY": "auto",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "row",
                    "flex": "1 1 auto",
                    "minHeight": "0",
                },
            ),
        ],
        style={"display": "flex", "flexDirection": "column", "flex": "1 1 auto", "minHeight": "0"},
    )
