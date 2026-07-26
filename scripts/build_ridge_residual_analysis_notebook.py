#!/usr/bin/env python3
"""Build notebooks/ridge_residual_analysis.ipynb (manual execution)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "ridge_residual_analysis.ipynb"


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
            "# Ridge Residual Analysis\n\n"
            "**Read-only diagnostic experiment.** Determines whether temporal structure "
            "remains in Prophet residuals after Ridge regression — and whether a "
            "nonlinear learner (GRU) is still justified.\n\n"
            "> **Scope:** CSRLE B0-LC, 99 containers. Trains Ridge only. "
            "No GRU. No modification of prior experiments."
        ),
        _md("## 1. Experiment overview"),
        _code(
            "import json\n"
            "import pickle\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n\n"
            "REPO_ROOT = Path('..').resolve()\n"
            "sys.path.insert(0, str(REPO_ROOT))\n\n"
            "# Point to latest RRA run (update timestamp if needed)\n"
            "EXP_DIR = REPO_ROOT / 'experiments' / 'ridge_residual_analysis_2026-07-26_161500'\n"
            "CONFIG_PATH = EXP_DIR / 'config' / 'ridge_residual_analysis_config.json'\n\n"
            "with CONFIG_PATH.open() as fh:\n"
            "    config = json.load(fh)\n\n"
            "print('Experiment:', config['experiment_name'])\n"
            "print('Timestamp:', config['timestamp'])\n"
            "print('Condition:', config['primary_condition'])\n"
            "print('Containers:', config['dataset']['n_containers'])"
        ),
        _md("## 2. Load frozen artifacts"),
        _code(
            "eval_df = pd.read_csv(EXP_DIR / 'tables' / 'ridge_evaluation_per_container.csv')\n"
            "diag_df = pd.read_csv(EXP_DIR / 'diagnostics' / 'temporal_structure_per_container.csv')\n"
            "with (EXP_DIR / 'diagnostics' / 'white_noise_comparison.json').open() as fh:\n"
            "    white_noise = json.load(fh)\n"
            "with (EXP_DIR / 'reports' / 'final_report.json').open() as fh:\n"
            "    final_report = json.load(fh)\n"
            "with (EXP_DIR / 'diagnostics' / 'case_study_containers.json').open() as fh:\n"
            "    case_studies = json.load(fh)\n\n"
            "eval_df.head()"
        ),
        _md("## 3. Train Ridge\n\nRidge is trained offline by `scripts/run_ridge_residual_analysis.py`. "
             "Load metadata below."),
        _code(
            "with (EXP_DIR / 'ridge' / 'training' / 'ridge_training_metadata.json').open() as fh:\n"
            "    ridge_meta = json.load(fh)\n"
            "with (EXP_DIR / 'ridge' / 'training' / 'residual_stats.json').open() as fh:\n"
            "    res_stats = json.load(fh)\n\n"
            "ridge_meta"
        ),
        _md("## 4. Evaluate Ridge"),
        _code(
            "cohort = eval_df[['ridge_pearson_r', 'ridge_mae_scaled', 'remaining_std_ratio']].describe()\n"
            "cohort"
        ),
        _md("## 5. Remaining residual analysis"),
        _code(
            "eval_df[['container_id', 'prophet_mean', 'prophet_std', 'remaining_mean', "
            "'remaining_std', 'remaining_std_ratio', 'var_reduction_fraction']].describe()"
        ),
        _md("## 6. Distribution analysis"),
        _code(
            "from IPython.display import Image\n"
            "Image(filename=str(EXP_DIR / 'plots' / 'residual_distribution_comparison.png'))"
        ),
        _md("## 7. ACF"),
        _code(
            "print('Prophet mean |ACF| 1-10:', white_noise['prophet_mean_avg_abs_acf_1_10'])\n"
            "print('Remaining mean |ACF| 1-10:', white_noise['remaining_mean_avg_abs_acf_1_10'])\n"
            "print('Reduction:', white_noise['mean_acf_reduction'])\n"
            "Image(filename=str(EXP_DIR / 'plots' / 'acf_boxplot_comparison.png'))"
        ),
        _md("## 8. PACF"),
        _code(
            "print('Prophet mean |PACF| 1-10:', white_noise['prophet_mean_avg_abs_pacf_1_10'])\n"
            "print('Remaining mean |PACF| 1-10:', white_noise['remaining_mean_avg_abs_pacf_1_10'])\n"
            "Image(filename=str(EXP_DIR / 'plots' / 'pacf_boxplot_comparison.png'))"
        ),
        _md("## 9. Ljung–Box"),
        _code(
            "print('Prophet reject fraction (lag 20):', "
            "white_noise['prophet_ljung_reject_fraction_lag_20'])\n"
            "print('Remaining reject fraction (lag 20):', "
            "white_noise['remaining_ljung_reject_fraction_lag_20'])\n"
            "diag_df[['container_id', 'prophet_lb_p_lag_20', 'remaining_lb_p_lag_20']].head()"
        ),
        _md("## 10. FFT"),
        _code(
            "fft_df = pd.read_csv(EXP_DIR / 'tables' / 'fft_summary.csv')\n"
            "fft_df.describe()[['prophet_dominant_freq', 'remaining_dominant_freq']]"
        ),
        _md("## 11. White-noise comparison"),
        _code(
            "white_noise"
        ),
        _md("## 12. Case studies"),
        _code(
            "for label, cid in case_studies.items():\n"
            "    print(f'{label}: {cid}')\n"
            "    display(Image(filename=str(EXP_DIR / 'plots' / f'case_study_{label}_{cid}.png')))"
        ),
        _md("## 13. Justification for Ridge→GRU"),
        _code(
            "j = final_report['justification']\n"
            "for k, v in j.items():\n"
            "    print(f'{k}: {v}')"
        ),
        _md("## 14. Final conclusions"),
        _code(
            "print('Verdict:', final_report['verdict'])\n"
            "print()\n"
            "for q, ans in final_report['explicit_answers'].items():\n"
            "    print(q, '->', ans)"
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
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK_PATH.open("w") as fh:
        json.dump(nb, fh, indent=1)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
