#!/usr/bin/env python3
"""Build notebooks/hybrid_v2_context_ablation.ipynb (manual execution)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "hybrid_v2_context_ablation.ipynb"


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
            "# HCERL — Hybrid v2 Context-Enriched Residual Learning\n\n"
            "Controlled ablation: V0 (96×1) through V4 (96×5).\n\n"
            "Documentation: `docs/hybrid_v2_context/`\n\n"
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
            "# Update to your HCERL run:\n"
            "runs = sorted((REPO_ROOT / 'experiments').glob('hcerl_*'))\n"
            "EXP_DIR = runs[-1] if runs else None\n"
            "print('Experiment:', EXP_DIR)\n\n"
            "if EXP_DIR is None:\n"
            "    raise FileNotFoundError('No hcerl_* experiment found — run scripts/run_hcerl_experiment.py')"
        ),
        _md("## 1. Experiment overview"),
        _code(
            "with (EXP_DIR / 'config' / 'hcerl_config_frozen.json').open() as fh:\n"
            "    config = json.load(fh)\n"
            "print('Protocol:', config['protocol_version'])\n"
            "print('Research question:', config['research_question'])\n"
            "config['variant_order']"
        ),
        _md("## 2. Cohort comparison"),
        _code(
            "cohort = pd.read_csv(EXP_DIR / 'ablation' / 'cohort_comparison.csv')\n"
            "cohort"
        ),
        _md("## 3. Feature contribution ranking"),
        _code(
            "ranking = pd.read_csv(EXP_DIR / 'ablation' / 'feature_contribution_ranking.csv')\n"
            "ranking"
        ),
        _md("## 4. Ablation plots"),
        _code(
            "for name in [\n"
            "    '01_overall_comparison',\n"
            "    '02_ablation_waterfall_mae',\n"
            "    '03_metric_evolution',\n"
            "    '04_std_ratio_distribution',\n"
            "    '08_feature_contribution_pearson',\n"
            "]:\n"
            "    p = EXP_DIR / 'plots' / f'{name}.png'\n"
            "    if p.exists():\n"
            "        display(Image(filename=str(p)))"
        ),
        _md("## 5. Per-variant extended metrics"),
        _code(
            "frames = []\n"
            "for vid in config['variant_order']:\n"
            "    df = pd.read_csv(EXP_DIR / 'variants' / vid / 'evaluation' / 'evaluation_extended.csv')\n"
            "    df['variant_id'] = vid\n"
            "    frames.append(df)\n"
            "all_eval = pd.concat(frames, ignore_index=True)\n"
            "all_eval.groupby('variant_id')[['day1_mae','day1_rmse','residual_pearson_r','residual_std_ratio']].mean()"
        ),
        _md("## 6. Paired statistical tests"),
        _code(
            "with (EXP_DIR / 'ablation' / 'paired_tests.json').open() as fh:\n"
            "    paired = json.load(fh)\n"
            "rows = []\n"
            "for tr in paired['transitions']:\n"
            "    for metric, stats in tr['metrics'].items():\n"
            "        rows.append({\n"
            "            'transition': tr['transition'],\n"
            "            'metric': metric,\n"
            "            'mean_delta': stats['mean_delta'],\n"
            "            'cohens_d': stats['cohens_d'],\n"
            "            'bootstrap_ci_low': stats['bootstrap']['ci_low'],\n"
            "            'bootstrap_ci_high': stats['bootstrap']['ci_high'],\n"
            "            'wilcoxon_p': stats['wilcoxon']['pvalue'],\n"
            "        })\n"
            "pd.DataFrame(rows)"
        ),
        _md("## 7. Training curves"),
        _code(
            "display(Image(filename=str(EXP_DIR / 'plots' / '07_training_curves.png')))"
        ),
        _md("## 8. Case studies"),
        _code(
            "with (EXP_DIR / 'case_studies' / 'case_studies.json').open() as fh:\n"
            "    cases = json.load(fh)\n"
            "pd.DataFrame(cases['cases']).T"
        ),
        _code(
            "for name in ['largest_improvement','performance_degradation']:\n"
            "    p = list((EXP_DIR / 'plots').glob(f'09_case_study_{name}_*.png'))\n"
            "    if p:\n"
            "        display(Image(filename=str(p[0])))"
        ),
        _md("## 9. Final report"),
        _code(
            "with (EXP_DIR / 'reports' / 'final_report.json').open() as fh:\n"
            "    report = json.load(fh)\n"
            "print('Recommended:', report['recommended_variant'])\n"
            "print(report['recommendation_rationale'])\n"
            "print()\n"
            "for q, a in report['explicit_answers'].items():\n"
            "    print(f'{q}: {a}')"
        ),
        _md("## 10. Conclusions\n\nSee `reports/final_report.md` in the experiment directory."),
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
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
