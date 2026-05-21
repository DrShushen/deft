from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from deft_mcp import __version__


def _load_env_file(env_path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ if not already set."""
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _autoload_deft_env() -> None:
    """If DEFT_API_KEY is not set, try to source <deft_repo>/.env once."""
    if os.environ.get("DEFT_API_KEY"):
        return
    deft_repo = Path(__file__).resolve().parents[3]
    _load_env_file(deft_repo / ".env")


def main(argv: list[str] | None = None) -> None:
    """Entry point for palestra-mcp-deft."""
    parser = argparse.ArgumentParser(
        prog="palestra-mcp-deft",
        description="MCP server wrapping DEFT (LLM-augmented decision-tree DNA classifier).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"palestra-mcp-deft {__version__}",
    )
    parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("DEFT_MCP_LOG_LEVEL", "INFO"),
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    )

    _autoload_deft_env()

    from deft_mcp.server import mcp

    mcp.run()
