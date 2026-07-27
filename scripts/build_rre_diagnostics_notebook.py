#!/usr/bin/env python3
"""Build notebooks/residual_representation_diagnostics.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "residual_representation_diagnostics.ipynb"


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


def build_notebook(exp_dir_placeholder: str = "experiments/rre_diagnostics_<timestamp>") -> None:
    cells = [
        _md(
            "# Residual Representation Diagnostic Study (Stage 0)\n\n"
            "**Read-only statistical analysis.** Compares causal residual "
            "representations R0–R3 before any GRU-based RRE experiment.\n\n"
            "> **No model training.** No GRU implementation. Evidence-based "
            "selection of the primary RRE candidate."
        ),
        _md("## 1. Setup and experiment directory"),
        _code(
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from scipy import stats\n\n"
            "REPO_ROOT = Path('..').resolve()\n"
            "sys.path.insert(0, str(REPO_ROOT))\n\n"
            "# Update after running scripts/run_rre_diagnostics.py\n"
            f"EXP_DIR = REPO_ROOT / '{exp_dir_placeholder}'\n\n"
            "CONFIG = json.loads((EXP_DIR / 'config' / 'rre_diagnostics_config.json').read_text())\n"
            "FINAL = json.loads((EXP_DIR / 'reports' / 'final_report.json').read_text())\n"
            "COMPARISON = pd.read_csv(EXP_DIR / 'comparison' / 'comparison_table.csv')\n"
            "REC = json.loads((EXP_DIR / 'comparison' / 'recommendation.json').read_text())\n\n"
            "print('Protocol:', CONFIG['protocol_version'])\n"
            "print('Containers:', CONFIG['n_containers'])\n"
            "print('Representations:', [r['id'] for r in CONFIG['representations']])"
        ),
        _md("## 2. Why Stage 0 exists"),
        _md(
            "Prior experiments (Hybrid baseline, peak-aware loss, context features, "
            "HCERL, LFHE, TMA) addressed **how** the GRU learns. They did not "
            "systematically compare **what signal** the GRU should model.\n\n"
            "Prophet explains ~78% of variance; residuals are centred, skewed, and "
            "heavy-tailed. Velocity (R1) was **not** assumed optimal — this study "
            "selects the primary RRE candidate from data."
        ),
        _md("## 3. Comparison table"),
        _code("COMPARISON"),
        _md("## 4. Distribution diagnostics (histogram, KDE, QQ)"),
        _code(
            "REP_IDS = ['R0', 'R1', 'R2', 'R3']\n"
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n"
            "for ax, rid in zip(axes.ravel(), REP_IDS):\n"
            "    dist = json.loads((EXP_DIR / 'representations' / rid / 'distribution.json').read_text())\n"
            "    bins = np.array(dist['histogram_bins'])\n"
            "    counts = np.array(dist['histogram_counts'])\n"
            "    if len(bins) > 1:\n"
            "        width = bins[1] - bins[0]\n"
            "        ax.bar(bins, counts, width=width * 0.9, alpha=0.6, label='Histogram')\n"
            "        values = np.repeat(bins, counts.astype(int))\n"
            "        if len(values) > 10:\n"
            "            xs = np.linspace(values.min(), values.max(), 200)\n"
            "            ax.plot(xs, stats.gaussian_kde(values)(xs) * len(values) * width, 'r-', label='KDE')\n"
            "    ax.set_title(f\"{rid}: skew={dist['skewness']:.2f}, kurt={dist['kurtosis']:.2f}\")\n"
            "    ax.set_xlabel('Value')\n"
            "    ax.legend(fontsize=8)\n"
            "plt.suptitle('Distribution profiles (train-period cohort pool)')\n"
            "plt.tight_layout()\n"
            "plt.show()\n\n"
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n"
            "for ax, rid in zip(axes.ravel(), REP_IDS):\n"
            "    qq = pd.read_csv(EXP_DIR / 'representations' / rid / 'qq_plot.csv')\n"
            "    ax.scatter(qq['theoretical_quantile'], qq['sample_quantile'], s=4, alpha=0.5)\n"
            "    lims = [qq[['theoretical_quantile', 'sample_quantile']].min().min(),\n"
            "            qq[['theoretical_quantile', 'sample_quantile']].max().max()]\n"
            "    ax.plot(lims, lims, 'k--', lw=1)\n"
            "    ax.set_title(f'{rid} QQ plot')\n"
            "    ax.set_xlabel('Theoretical quantiles')\n"
            "    ax.set_ylabel('Sample quantiles')\n"
            "plt.suptitle('Normality QQ plots (subsample)')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("## 5. Temporal structure (ACF / PACF)"),
        _code(
            "fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)\n"
            "for rid in REP_IDS:\n"
            "    acf_df = pd.read_csv(EXP_DIR / 'representations' / rid / 'acf_cohort.csv')\n"
            "    axes[0].plot(acf_df['lag'], acf_df['acf_mean'], label=rid)\n"
            "    pacf_df = pd.read_csv(EXP_DIR / 'representations' / rid / 'pacf_cohort.csv')\n"
            "    axes[1].plot(pacf_df['lag'], pacf_df['pacf_mean'], label=rid)\n"
            "axes[0].axhline(0, color='k', lw=0.5)\n"
            "axes[0].set_ylabel('ACF')\n"
            "axes[0].legend()\n"
            "axes[0].set_title('Cohort mean ACF')\n"
            "axes[1].axhline(0, color='k', lw=0.5)\n"
            "axes[1].set_ylabel('PACF')\n"
            "axes[1].set_xlabel('Lag (15-min steps)')\n"
            "axes[1].set_title('Cohort mean PACF')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("## 6. Frequency characteristics (PSD)"),
        _code(
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "for rid in REP_IDS:\n"
            "    psd = pd.read_csv(EXP_DIR / 'representations' / rid / 'psd.csv')\n"
            "    ax.semilogy(psd['freq'], psd['psd_mean'], label=rid)\n"
            "daily_f = 1.0 / 96.0\n"
            "ax.axvline(daily_f, color='gray', ls='--', alpha=0.5, label='Daily (1/96)' if rid == 'R0' else '')\n"
            "ax.set_xlabel('Frequency (1/steps)')\n"
            "ax.set_ylabel('Mean PSD')\n"
            "ax.set_title('Cohort mean power spectral density')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("## 7. Stationarity and information summary"),
        _code(
            "rows = []\n"
            "for rid in REP_IDS:\n"
            "    stat = json.loads((EXP_DIR / 'representations' / rid / 'stationarity.json').read_text())\n"
            "    info = json.loads((EXP_DIR / 'representations' / rid / 'information.json').read_text())\n"
            "    rows.append({'id': rid, **stat, **info})\n"
            "pd.DataFrame(rows)"
        ),
        _md("## 8. Evidence-based recommendation"),
        _code(
            "ans = REC['answers']\n"
            "print('1. Most temporal structure:', ans['q1_most_temporal_structure'])\n"
            "print('2. Most learnable (proxy):', ans['q2_most_learnable_proxy'])\n"
            "print('3. Best balance:', ans['q3_best_balance'])\n"
            "print('4. Primary candidate:', ans['q4_primary_candidate'], '-', REC['primary_recommendation_name'])\n"
            "print('5. Level baseline remains primary?', ans['q5_level_baseline_remains'])\n"
            "print()\n"
            "print(REC['rationale'])"
        ),
        _md("## 9. Radar-style rank visualization"),
        _code(
            "rankings = REC['rankings']\n"
            "metrics = ['temporal_structure', 'learnability_proxy', 'stability', 'balance']\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "x = np.arange(len(metrics))\n"
            "width = 0.2\n"
            "for i, rid in enumerate(REP_IDS):\n"
            "    vals = [rankings[m][rid] for m in metrics]\n"
            "    ax.bar(x + i * width, vals, width, label=rid)\n"
            "ax.set_xticks(x + width * 1.5)\n"
            "ax.set_xticklabels(metrics, rotation=15)\n"
            "ax.set_ylabel('Rank (1 = best)')\n"
            "ax.set_title('Representation ranks across diagnostic criteria')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("## 10. Consolidated conclusion"),
        _md(
            "Stage 0 informed RRE v1.0. The full two-stage conclusion is in "
            "`docs/hybrid_residual_investigation/CONCLUSION.md`.\n\n"
            "Stage 0 outcomes: R1 (velocity) rejected; R3 selected as challenger "
            "because it preserves the same ACF as R0. RRE v1.0 subsequently tested "
            "R3 vs R0 in training and found no forecast improvement — retain global z-score."
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
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {NOTEBOOK_PATH}")


def main() -> None:
    build_notebook()


if __name__ == "__main__":
    main()
