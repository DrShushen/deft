"""DeftPlugin: palestra view entry point for DEFT decision trees."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dash import Dash, Input, Output, html, no_update
from dash.development.base_component import Component
from palestra.view.plugin_api import plugin_host_id, plugin_state_store_id

from deft_view import layout as L
from deft_view.elements import build_cytoscape_elements, load_snapshot, snapshot_label
from deft_view.paths import discover_tree_jsons
from deft_view.stylesheet import CYTOSCAPE_STYLESHEET

logger = logging.getLogger(__name__)

NAME = "deft"
LABEL = "DEFT Tree"


def _run_options(snapshots: list[Path]) -> list[dict]:
    """Build dropdown options (value = tree.json path, label = run name + metrics)."""
    return [{"value": str(p), "label": snapshot_label(p)} for p in snapshots]


def _resolve_selection(host_state: dict | None, snapshots: list[Path]) -> Path:
    """Pick which run to show: the host_state choice if still present, else newest.

    ``snapshots`` is the discovery result, sorted newest-first, so
    ``snapshots[0]`` is the most recently finished run.
    """
    by_path = {str(p): p for p in snapshots}
    selected = (host_state or {}).get("selected")
    if isinstance(selected, str) and selected in by_path:
        return by_path[selected]
    return snapshots[0]


def _no_runs_view(name: str) -> Component:
    return html.Div(
        L.render_error(
            "No DEFT tree snapshot found under workspace/*/tree.json. Did the run_deft tool finish successfully?"
        ),
        id=plugin_host_id(name),
    )


def _build_tree_view(selected: Path) -> tuple[list[dict], Any, dict | None]:
    """Load ``selected`` and return ``(elements, summary_children, snapshot_store)``.

    On any load/parse error, returns empty elements + an error banner for the
    summary slot + ``None`` store -- so the caller can still render the
    selector and the user can switch to another run.
    """
    try:
        snapshot = load_snapshot(selected)
        elements, summary = build_cytoscape_elements(snapshot)
    except Exception as exc:
        logger.exception("Failed to load/build DEFT snapshot from %s", selected)
        return [], L.render_error(f"Failed to load {selected.name}: {exc}"), None
    meta = snapshot.get("meta", {})
    return elements, L.build_summary_bar(summary, meta), {"path": str(selected), "summary": summary}


class DeftPlugin:
    """Palestra view plugin that renders DEFT trees as a cytoscape graph.

    Discovery is tolerant: any ``workspace/*/tree.json`` makes the plugin
    applicable, regardless of the output-directory name the agent chose.
    When a step holds several runs, a selector lets the user switch between
    them; the choice is persisted in host_state so it survives plugin
    re-mounts (auto-refresh ticks, tab toggles, step changes).
    """

    name: str = NAME
    label: str = LABEL

    def is_applicable(self, step_dir: Path) -> bool:
        try:
            return bool(discover_tree_jsons(step_dir))
        except OSError:
            return False

    def host_state_initial(self) -> dict | None:
        # ``selected`` is the tree.json path of the run the user last chose;
        # ``None`` means "default to the newest run".
        return {"selected": None}

    def render(self, step_dir: Path, host_state: dict | None) -> Component:
        snapshots = discover_tree_jsons(step_dir)
        if not snapshots:
            return _no_runs_view(self.name)

        selected = _resolve_selection(host_state, snapshots)
        run_selector = L.build_run_selector(_run_options(snapshots), str(selected))
        elements, summary_children, snapshot_store = _build_tree_view(selected)

        cytoscape = L.build_cytoscape(elements, CYTOSCAPE_STYLESHEET, L.cytoscape_layout(L.DEFAULT_LAYOUT_KEY))
        inner = L.run_view_layout(
            run_selector=run_selector,
            summary_bar=summary_children,
            cytoscape=cytoscape,
            detail_initial=L.render_detail(None),
            snapshot_store_data=snapshot_store,
        )
        return html.Div(inner, id=plugin_host_id(self.name))

    def register_callbacks(self, app: Dash) -> None:
        # ---- Tap a node: render its detail panel. ----------------------------
        # Registered first so its plain ``ID_DETAIL`` output is the canonical
        # writer; the run-selector callback below targets ``ID_DETAIL`` too
        # and must therefore use ``allow_duplicate=True``.
        @app.callback(
            Output(L.ID_DETAIL, "children"),
            Input(L.ID_CYTOSCAPE, "tapNodeData"),
            prevent_initial_call=True,
        )
        def _on_tap_node(tap_data: dict | None) -> Any:
            if tap_data is None:
                return no_update
            try:
                return L.render_detail(tap_data)
            except Exception as exc:
                logger.exception("Failed to render detail panel")
                return L.render_error(f"Detail render failed: {exc}")

        # ---- Run selector: load the chosen run and swap the view in place. ----
        # A direct callback (not the host's re-render) gives an immediate
        # switch; the host only re-renders on view-state changes, so relying
        # on host_state alone would lag a poll tick. We also mirror the choice
        # into the host-level Store so a later re-mount restores it.
        @app.callback(
            Output(L.ID_CYTOSCAPE, "elements"),
            Output(L.ID_SUMMARY, "children"),
            Output(L.ID_DETAIL, "children", allow_duplicate=True),
            Output(L.ID_SNAPSHOT_STORE, "data"),
            Output(plugin_state_store_id(self.name), "data"),
            Input(L.ID_RUN_SELECTOR, "value"),
            prevent_initial_call=True,
        )
        def _on_run_change(selected_path: str | None) -> Any:
            if not selected_path:
                return no_update, no_update, no_update, no_update, no_update
            elements, summary_children, snapshot_store = _build_tree_view(Path(selected_path))
            return (
                elements,
                summary_children,
                L.render_detail(None),
                snapshot_store,
                {"selected": selected_path},
            )

        # ---- Layout dropdown: re-run the chosen cytoscape layout. ------------
        @app.callback(
            Output(L.ID_CYTOSCAPE, "layout"),
            Input(L.ID_LAYOUT_DROPDOWN, "value"),
            prevent_initial_call=True,
        )
        def _on_layout_change(value: str | None) -> Any:
            if not value:
                return no_update
            return L.cytoscape_layout(value)

        # ---- Fit the tree to the viewport (clientside). ----------------------
        # Fires on exactly three intents: a layout change, a run switch, and
        # the Fit button. cytoscape honours a layout's ``fit: true`` only on
        # its FIRST run -- re-running a layout on an existing graph repositions
        # nodes but leaves the viewport stale -- so we fit explicitly, deferred
        # two animation frames so the (synchronous, animate:false) re-layout
        # settles first. Clientside so it's instant and never hits the server.
        #
        # The run-switch trigger is the snapshot Store (written only by
        # ``_on_run_change``), deliberately NOT ``elements``: dash-cytoscape
        # writes node positions back into ``elements`` on every node drag
        # (``cy.on('dragfree', ...)``), so keying the fit off ``elements`` would
        # yank the viewport back whenever the user merely moves a node.
        app.clientside_callback(
            """
            function(layout, snapshot, nClicks) {
                var el = document.getElementById('"""
            + L.ID_CYTOSCAPE
            + """');
                if (el && el._cyreg && el._cyreg.cy) {
                    var cy = el._cyreg.cy;
                    requestAnimationFrame(function() {
                        requestAnimationFrame(function() { cy.fit(undefined, 20); });
                    });
                }
                return window.dash_clientside.no_update;
            }
            """,
            Output(L.ID_FIT_BUTTON, "n_clicks"),
            Input(L.ID_CYTOSCAPE, "layout"),
            Input(L.ID_SNAPSHOT_STORE, "data"),
            Input(L.ID_FIT_BUTTON, "n_clicks"),
            prevent_initial_call=True,
        )
