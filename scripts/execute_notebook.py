#!/usr/bin/env python3
"""Execute a Jupyter notebook in-place and save with embedded cell outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def execute_notebook(
    notebook_path: Path,
    *,
    timeout: int = 900,
    kernel_name: str = "python3",
) -> None:
    """Run all cells and overwrite the notebook with outputs (plots, tables)."""
    notebook_path = notebook_path.resolve()
    if not notebook_path.exists():
        raise FileNotFoundError(notebook_path)

    with notebook_path.open(encoding="utf-8") as handle:
        nb = nbformat.read(handle, as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )
    client.execute()

    with notebook_path.open("w", encoding="utf-8") as handle:
        nbformat.write(nb, handle)

    print(f"Executed and saved: {notebook_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path, help="Path to .ipynb file")
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-notebook execution timeout in seconds (default: 900)",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name (default: python3)",
    )
    args = parser.parse_args(argv)

    try:
        execute_notebook(args.notebook, timeout=args.timeout, kernel_name=args.kernel)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
