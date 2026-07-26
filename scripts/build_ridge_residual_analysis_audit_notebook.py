#!/usr/bin/env python3
"""Build notebooks/ridge_residual_analysis_audit.ipynb (manual execution)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "ridge_residual_analysis_audit.ipynb"
RRA_EXP = "experiments/ridge_residual_analysis_2026-07-26_161500"
AUDIT_EXP = "experiments/ridge_residual_analysis_audit_2026-07-26_162500"


def _md(t: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [t]}


def _code(t: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [t],
    }


def build() -> None:
    cells = [
        _md(
            "# Ridge Residual Analysis — Read-Only Audit\n\n"
            "Verifies RRA v1.0 methodology after counterintuitive ACF increase. "
            "**No experiments modified. No GRU training.**"
        ),
        _md("## 1. Setup"),
        _code(
            "import json\nimport sys\nfrom pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\n"
            "REPO = Path('..').resolve()\nsys.path.insert(0, str(REPO))\n\n"
            f"RRA_EXP = REPO / '{RRA_EXP}'\n"
            f"AUDIT_EXP = REPO / '{AUDIT_EXP}'\n\n"
            "with (AUDIT_EXP / 'reports' / 'audit_report.json').open() as fh:\n"
            "    report = json.load(fh)\nreport.keys()"
        ),
        _md("## 2. Task 1 — Residual definition & scaling"),
        _code(
            "scale = pd.read_csv(AUDIT_EXP / 'verification' / 'task1_scaling_mismatch.csv')\n"
            "task1 = pd.read_csv(AUDIT_EXP / 'verification' / 'task1_residual_definition.csv')\n"
            "print('Identity holds (original RRA):', task1['original_identity_holds'].all())\n"
            "print('Sign: prophet - ridge (confirmed)')\n"
            "print('Scaling bug: raw prophet minus z-scored ridge')\n"
            "scale[['container_id','prophet_raw_std','ridge_pred_z_std','mean_abs_diff_raw_vs_z_pred']].head()"
        ),
        _md("## 3. Task 2 — Alignment (Day-1 block)"),
        _code(
            "with (AUDIT_EXP / 'verification' / 'task2_day1_alignment_sample.json').open() as fh:\n"
            "    align = json.load(fh)\nalign"
        ),
        _md("## 4. Task 3 — Data splits / leakage"),
        _code(
            "with (AUDIT_EXP / 'verification' / 'task3_data_splits.json').open() as fh:\n"
            "    splits = json.load(fh)\nsplits"
        ),
        _md("## 5. Task 4 — ACF replication (RRA vs corrected)"),
        _code(
            "rra = report['rra_reported']\n"
            "corr = report['corrected_summary']\n"
            "pd.DataFrame([\n"
            "    {'metric': 'mean |ACF| 1-10', 'prophet': rra['prophet_mean_acf_1_10'],\n"
            "     'rra_remaining': rra['rra_remaining_mean_acf_1_10'],\n"
            "     'corrected_remaining': corr['corrected_remaining_mean_acf_1_10']},\n"
            "    {'metric': 'Ljung reject lag20', 'prophet': rra['rra_prophet_lb_reject_20'],\n"
            "     'rra_remaining': rra['rra_remaining_lb_reject_20'],\n"
            "     'corrected_remaining': corr['corrected_remaining_lb_reject_20']},\n"
            "])"
        ),
        _md("## 6. Task 5 — Why ACF appeared to increase"),
        _code(
            "report['task5']"
        ),
        _md("## 7. Task 6 — Linear models on corrected remaining"),
        _code(
            "ar = pd.read_csv(AUDIT_EXP / 'tables' / 'linear_model_ar_summary.csv')\nar"
        ),
        _md("## 8. Task 7 — Decision"),
        _code(
            "report['decision']"
        ),
        _md("## 9. Interpretation"),
        _code(
            "print('Scaling bug confirmed:', report['decision']['scaling_bug_in_rra_v1'])\n"
            "print('Corrected ACF reduction:', report['corrected_summary']['corrected_acf_reduction'])\n"
            "print('Ridge→GRU justified:', report['decision']['q3_ridge_gru_justified_vs_more_linear_modelling'])"
        ),
    ]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    with NOTEBOOK.open("w") as fh:
        json.dump(nb, fh, indent=1)
    print(f"Wrote {NOTEBOOK}")


if __name__ == "__main__":
    build()
