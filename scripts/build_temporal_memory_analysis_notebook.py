#!/usr/bin/env python3
"""Build notebooks/temporal_memory_analysis.ipynb (manual execution)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "temporal_memory_analysis.ipynb"
EXP_DIR = "experiments/temporal_memory_analysis_2026-07-26_170052"


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
            "# Temporal Memory Analysis\n\n"
            "**Read-only diagnostic experiment.** Determines whether the fixed "
            "96-step input window captures temporal dependencies in Alibaba "
            "workloads, or whether useful memory exists beyond one day.\n\n"
            "> **Scope:** 99 evaluable validation containers. No model training. "
            "No Prophet.fit(), GRU, or Ridge fitting in this experiment."
        ),
        _md("## 1. Experiment overview"),
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
            f"EXP_DIR = REPO_ROOT / '{EXP_DIR}'\n"
            "CONFIG_PATH = EXP_DIR / 'config' / 'temporal_memory_analysis_config.json'\n\n"
            "with CONFIG_PATH.open() as fh:\n"
            "    config = json.load(fh)\n\n"
            "print('Experiment:', config['experiment_name'])\n"
            "print('Timestamp:', config['timestamp'])\n"
            "print('Containers:', config['n_containers'])\n"
            "print('Max lag:', config['maximum_lag'])"
        ),
        _md("## 2. Load frozen data"),
        _code(
            "summary_df = pd.read_csv(EXP_DIR / 'tables' / 'cohort_summary.csv')\n"
            "with (EXP_DIR / 'reports' / 'final_report.json').open() as fh:\n"
            "    final_report = json.load(fh)\n"
            "with (EXP_DIR / 'verification' / 'data_integrity.json').open() as fh:\n"
            "    verification = json.load(fh)\n\n"
            "summary_df"
        ),
        _md("## 3. CPU memory analysis"),
        _code(
            "with (EXP_DIR / 'statistics' / 'cpu_original_memory_length.json').open() as fh:\n"
            "    cpu_memory = json.load(fh)\n"
            "cpu_memory['distribution']"
        ),
        _md("## 4. Prophet residual memory"),
        _code(
            "with (EXP_DIR / 'statistics' / 'prophet_residual_memory_length.json').open() as fh:\n"
            "    prophet_memory = json.load(fh)\n"
            "prophet_memory['distribution']"
        ),
        _md("## 5. Hybrid residual memory"),
        _code(
            "with (EXP_DIR / 'statistics' / 'hybrid_post_forecast_residual_memory_length.json').open() as fh:\n"
            "    hybrid_memory = json.load(fh)\n"
            "hybrid_memory['distribution']"
        ),
        _md("## 6. Long-lag ACF"),
        _code(
            "display(Image(filename=str(EXP_DIR / 'plots' / 'cohort_acf_cpu_original.png')))\n"
            "display(Image(filename=str(EXP_DIR / 'plots' / 'acf_heatmap_cpu_original.png')))"
        ),
        _md("## 7. Long-lag PACF"),
        _code(
            "display(Image(filename=str(EXP_DIR / 'plots' / 'cohort_pacf_cpu_original.png')))"
        ),
        _md("## 8. FFT"),
        _code(
            "fft_cpu = pd.read_csv(EXP_DIR / 'fft' / 'cpu_original_summary.csv')\n"
            "fft_cpu[['dominant_period_steps', 'daily_freq_power_fraction']].describe()\n"
            "display(Image(filename=str(EXP_DIR / 'plots' / 'fft_psd_comparison.png')))"
        ),
        _md("## 9. Daily periodicity"),
        _code(
            "with (EXP_DIR / 'statistics' / 'cpu_original_daily_periodicity.json').open() as fh:\n"
            "    daily = json.load(fh)\n"
            "pd.DataFrame(daily['lags']).T\n"
            "display(Image(filename=str(EXP_DIR / 'plots' / 'daily_periodicity_cpu_original.png')))"
        ),
        _md("## 10. Memory length estimation"),
        _code(
            "display(Image(filename=str(EXP_DIR / 'plots' / 'memory_length_hist_cpu_original.png')))\n"
            "display(Image(filename=str(EXP_DIR / 'plots' / 'memory_decay_cpu_original.png')))"
        ),
        _md("## 11. Window sufficiency"),
        _code(
            "with (EXP_DIR / 'statistics' / 'cpu_original_window_sufficiency.json').open() as fh:\n"
            "    window_suff = json.load(fh)\n"
            "window_suff['cohort']\n"
            "display(Image(filename=str(EXP_DIR / 'plots' / 'window_sufficiency_comparison.png')))"
        ),
        _md("## 12. Case studies"),
        _code(
            "with (EXP_DIR / 'statistics' / 'case_study_containers.json').open() as fh:\n"
            "    case_studies = json.load(fh)\n"
            "case_studies"
        ),
        _code(
            "for role, cid in case_studies.items():\n"
            "    path = EXP_DIR / 'plots' / f'case_study_{role}_{cid}.png'\n"
            "    if path.exists():\n"
            "        display(Image(filename=str(path)))"
        ),
        _md("## 13. Discussion"),
        _code(
            "with (EXP_DIR / 'statistics' / 'cpu_original_paired_day1_vs_beyond.json').open() as fh:\n"
            "    paired = json.load(fh)\n"
            "paired"
        ),
        _md("## 14. Final conclusions"),
        _code(
            "for key, answer in final_report['explicit_answers'].items():\n"
            "    print(f'\\n{key}:\\n{answer}')\n"
            "print('\\nVerdict:', final_report['verdict'])"
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w") as handle:
        json.dump(notebook, handle, indent=1)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
