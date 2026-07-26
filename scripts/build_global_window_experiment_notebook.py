#!/usr/bin/env python3
"""Build notebooks/global_window_experiment.ipynb (manual execution)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "global_window_experiment.ipynb"


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [text],
    }


def build_notebook() -> None:
    cells = [
        _md(
            "# GGTCE — Global GRU Temporal Context Experiment\n\n"
            "Input window ablation: G96, G192, G288 (output horizon fixed at 96).\n\n"
            "Documentation: `docs/global_window_experiment/`\n\n"
            "> Set `EXP_DIR` below to your authoritative run directory."
        ),
        _code(
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "REPO_ROOT = Path('..').resolve()\n"
            "sys.path.insert(0, str(REPO_ROOT))\n\n"
            "runs = sorted((REPO_ROOT / 'experiments').glob('ggtce_*'))\n"
            "EXP_DIR = runs[-1] if runs else None\n"
            "print('Experiment:', EXP_DIR)\n\n"
            "if EXP_DIR is None:\n"
            "    raise FileNotFoundError('No ggtce_* experiment — run scripts/run_ggtce_experiment.py')"
        ),
        _md("## 1. Introduction"),
        _code(
            "with (EXP_DIR / 'config' / 'ggtce_config_frozen.json').open() as fh:\n"
            "    config = json.load(fh)\n"
            "print('Protocol:', config['protocol_version'])\n"
            "print('RQ:', config['research_question'])\n"
            "print('Co-primary:', config['co_primary_question'])"
        ),
        _md("## 2. Methodology & frozen parity"),
        _code(
            "with (EXP_DIR / 'verification' / 'frozen_parity.json').open() as fh:\n"
            "    parity = json.load(fh)\n"
            "pd.DataFrame(parity['checks'])"
        ),
        _md("## 3. G96 replication gate"),
        _code(
            "with (EXP_DIR / 'verification' / 'g96_replication.json').open() as fh:\n"
            "    rep = json.load(fh)\n"
            "rep"
        ),
        _md("## 4. Training"),
        _code(
            "seq = pd.read_csv(EXP_DIR / 'sequence_analysis' / 'sequence_reduction.csv')\n"
            "seq"
        ),
        _md("## 5. Evaluation"),
        _code(
            "cohort = pd.read_csv(EXP_DIR / 'window_comparison' / 'cohort_comparison.csv')\n"
            "cohort"
        ),
        _md("## 6. Window comparison"),
        _code(
            "deltas = pd.read_csv(EXP_DIR / 'window_comparison' / 'deltas_g96_to_g288.csv')\n"
            "deltas.describe()"
        ),
        _md("## 7. Memory-aware analysis"),
        _code(
            "mem = pd.read_csv(EXP_DIR / 'memory_analysis' / 'merged_deltas_g96_g288.csv')\n"
            "mem[['container_id','memory_score','memory_class','delta_day1_mae']].head(10)"
        ),
        _md("## 8. Case studies"),
        _code(
            "with (EXP_DIR / 'case_studies' / 'case_studies.json').open() as fh:\n"
            "    cases = json.load(fh)\n"
            "cases['cases']"
        ),
        _md("## 9. Visualizations"),
        _code(
            "for name in sorted((EXP_DIR / 'plots').glob('*.png')):\n"
            "    display(Image(filename=str(name)))"
        ),
        _md("## 10. Discussion & conclusions"),
        _code(
            "with (EXP_DIR / 'reports' / 'final_report.json').open() as fh:\n"
            "    report = json.load(fh)\n"
            "for k, v in report['final_answers'].items():\n"
            "    print(f'{k}: {v}')"
        ),
    ]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
        },
        "cells": cells,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w") as fh:
        json.dump(nb, fh, indent=1)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
