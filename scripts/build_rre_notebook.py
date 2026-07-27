#!/usr/bin/env python3
"""Build notebooks/residual_scaling_experiment.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "residual_scaling_experiment.ipynb"
EXP_PLACEHOLDER = "experiments/rre_<timestamp>"


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


def build_notebook(exp_dir: str = EXP_PLACEHOLDER) -> None:
    cells = [
        _md(
            "# Residual Scaling Experiment (RRE v1.0)\n\n"
            "**Research question:** Can a more optimisation-friendly scaling of the "
            "*same* Prophet residual improve Hybrid GRU learning without changing "
            "underlying temporal information?\n\n"
            "> This is **not** a temporal representation experiment. Stage 0 showed "
            "R3 (median/MAD) preserves identical ACF structure to R0 while improving "
            "numerical robustness."
        ),
        _md("## 1. Setup"),
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
            f"EXP_DIR = REPO_ROOT / '{exp_dir}'\n"
            "CONFIG = json.loads((EXP_DIR / 'config' / 'rre_config_frozen.json').read_text())\n"
            "FINAL = json.loads((EXP_DIR / 'reports' / 'final_report.json').read_text())\n"
            "COMP = json.loads((EXP_DIR / 'comparison' / 'paired_tests.json').read_text())\n"
            "ANSWERS = FINAL['answers']\n\n"
            "eval_r0 = pd.read_csv(EXP_DIR / 'variants' / 'R0' / 'evaluation' / 'evaluation_extended.csv')\n"
            "eval_r3 = pd.read_csv(EXP_DIR / 'variants' / 'R3' / 'evaluation' / 'evaluation_extended.csv')\n"
            "hist_r0 = pd.read_csv(EXP_DIR / 'variants' / 'R0' / 'training' / 'training_history.csv')\n"
            "hist_r3 = pd.read_csv(EXP_DIR / 'variants' / 'R3' / 'training' / 'training_history.csv')\n\n"
            "print('Protocol:', CONFIG['protocol_version'])\n"
            "print('Research question:', CONFIG['research_question'])"
        ),
        _md("## 2. Motivation and Stage 0 summary"),
        _md(
            "Prior work established that Prophet explains ~78% of CPU variance, Hybrid "
            "beats Prophet, and changes to loss/context/windows did not help. **Stage 0** "
            "compared residual representations statistically:\n\n"
            "- **R0** (level z-score): strongest temporal structure\n"
            "- **R1** (velocity): rejected — removes temporal dependence\n"
            "- **R2**: no clear advantage\n"
            "- **R3** (median/MAD): **same ACF as R0**, lower sparsity, more robust tails\n\n"
            "Therefore RRE v1.0 tests **optimisation conditioning**, not new temporal information."
        ),
        _md("## 3. Mathematical formulation"),
        _md(
            "Level Prophet residual: $r_t = y_t - \\hat{y}_t$\n\n"
            "**R0 (baseline):** $\\tilde{r}_t = (r_t - \\mu_{train}) / \\sigma_{train}$\n\n"
            "**R3 (robust):** $\\tilde{r}_t = (r_t - \\mathrm{median}_{train}) / \\mathrm{MAD}_{train}$\n\n"
            "Inverse at inference: $\\hat{r}_t = \\tilde{r}_t \\cdot s + c$ where $(c,s)$ are "
            "train-fitted centre and scale. GRU architecture, window, optimizer, and loss are frozen."
        ),
        _md("## 4. Cohort comparison table"),
        _code(
            "summary_rows = []\n"
            "for vid in ['R0', 'R3']:\n"
            "    s = FINAL['cohort_summaries'][vid]\n"
            "    summary_rows.append({\n"
            "        'variant': vid,\n"
            "        'mean_day1_mae': s['day1_mae']['mean'],\n"
            "        'mean_day1_rmse': s['day1_rmse']['mean'],\n"
            "        'mean_pearson_r': s['residual_pearson_r']['mean'],\n"
            "        'mean_std_ratio': s['residual_std_ratio']['mean'],\n"
            "    })\n"
            "pd.DataFrame(summary_rows)"
        ),
        _md("## 5. Training and validation loss curves"),
        _code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "for hist, label, color in [(hist_r0, 'R0', 'C0'), (hist_r3, 'R3', 'C1')]:\n"
            "    axes[0].plot(hist['epoch'], hist['loss'], label=f'{label} train', color=color)\n"
            "    axes[0].plot(hist['epoch'], hist['val_loss'], '--', label=f'{label} val', color=color)\n"
            "axes[0].set_xlabel('Epoch')\n"
            "axes[0].set_ylabel('MSE loss')\n"
            "axes[0].set_title('Training dynamics (optimisation focus)')\n"
            "axes[0].legend(fontsize=8)\n"
            "axes[0].grid(alpha=0.3)\n\n"
            "merged = eval_r0.merge(eval_r3, on='container_id', suffixes=('_R0', '_R3'))\n"
            "axes[1].scatter(merged['day1_mae_R0'], merged['day1_mae_R3'], alpha=0.6, s=20)\n"
            "lim = [0, max(merged['day1_mae_R0'].max(), merged['day1_mae_R3'].max()) * 1.05]\n"
            "axes[1].plot(lim, lim, 'k--', lw=1)\n"
            "axes[1].set_xlabel('R0 day1 MAE (%)')\n"
            "axes[1].set_ylabel('R3 day1 MAE (%)')\n"
            "axes[1].set_title('Per-container MAE: R3 vs R0')\n"
            "axes[1].grid(alpha=0.3)\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("## 6. Optimisation metrics"),
        _code(
            "opt = pd.DataFrame(FINAL['optimization_summaries']).T\n"
            "opt[['best_epoch', 'epochs_run', 'best_val_loss', 'generalisation_gap', 'convergence_speed']]"
        ),
        _md("## 7. Forecast metric distributions"),
        _code(
            "fig, axes = plt.subplots(2, 2, figsize=(11, 9))\n"
            "metrics = [\n"
            "    ('day1_mae', 'Day-1 MAE (%)'),\n"
            "    ('day1_rmse', 'Day-1 RMSE (%)'),\n"
            "    ('residual_pearson_r', 'Residual Pearson r'),\n"
            "    ('residual_std_ratio', 'Residual std ratio'),\n"
            "]\n"
            "for ax, (col, title) in zip(axes.ravel(), metrics):\n"
            "    ax.boxplot([eval_r0[col].dropna(), eval_r3[col].dropna()], tick_labels=['R0', 'R3'])\n"
            "    ax.set_title(title)\n"
            "    ax.grid(alpha=0.3)\n"
            "plt.suptitle('Cohort metric distributions')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("## 8. Statistical tests (bootstrap + Wilcoxon)"),
        _code(
            "rows = []\n"
            "for metric, block in COMP['forecast_metrics'].items():\n"
            "    rows.append({\n"
            "        'metric': metric,\n"
            "        'mean_delta_R3_minus_R0': block['mean_delta'],\n"
            "        'wilcoxon_p': block['wilcoxon']['pvalue'],\n"
            "        'bootstrap_ci_lower': block['bootstrap']['ci_lower'],\n"
            "        'bootstrap_ci_upper': block['bootstrap']['ci_upper'],\n"
            "        'fraction_improved': block['fraction_improved'],\n"
            "    })\n"
            "pd.DataFrame(rows)"
        ),
        _md("## 9. Case studies"),
        _code(
            "merged = eval_r0.merge(eval_r3, on='container_id', suffixes=('_R0', '_R3'))\n"
            "merged['delta_mae'] = merged['day1_mae_R3'] - merged['day1_mae_R0']\n"
            "best_r3 = merged.nsmallest(3, 'delta_mae')[['container_id', 'day1_mae_R0', 'day1_mae_R3', 'delta_mae']]\n"
            "worst_r3 = merged.nlargest(3, 'delta_mae')[['container_id', 'day1_mae_R0', 'day1_mae_R3', 'delta_mae']]\n"
            "print('Top R3 improvements:')\n"
            "display(best_r3)\n"
            "print('Top R3 regressions:')\n"
            "display(worst_r3)"
        ),
        _md("## 10. Prediction examples (case study containers)"),
        _code(
            "import pickle\n\n"
            "def plot_case(cid, ax):\n"
            "    cache_r0 = pickle.load(open(EXP_DIR / 'variants' / 'R0' / 'evaluation' / 'inference_cache.pkl', 'rb'))\n"
            "    cache_r3 = pickle.load(open(EXP_DIR / 'variants' / 'R3' / 'evaluation' / 'inference_cache.pkl', 'rb'))\n"
            "    actual = cache_r0[cid]['actual_day1_real']\n"
            "    pred0 = cache_r0[cid]['day1_final_real']\n"
            "    pred3 = cache_r3[cid]['day1_final_real']\n"
            "    prophet = cache_r0[cid]['day1_prophet_real']\n"
            "    t = np.arange(len(actual))\n"
            "    ax.plot(t, actual, 'k-', label='Actual', lw=1.5)\n"
            "    ax.plot(t, prophet, ':', color='gray', label='Prophet')\n"
            "    ax.plot(t, pred0, '--', label='Hybrid R0')\n"
            "    ax.plot(t, pred3, '-.', label='Hybrid R3')\n"
            "    ax.set_title(f'{cid}')\n"
            "    ax.set_xlabel('Step (15 min)')\n"
            "    ax.set_ylabel('CPU %')\n"
            "    ax.legend(fontsize=7)\n"
            "    ax.grid(alpha=0.3)\n\n"
            "case_ids = list(best_r3['container_id'].head(2)) + list(worst_r3['container_id'].head(1))\n"
            "fig, axes = plt.subplots(len(case_ids), 1, figsize=(11, 3 * len(case_ids)))\n"
            "if len(case_ids) == 1:\n"
            "    axes = [axes]\n"
            "for ax, cid in zip(axes, case_ids):\n"
            "    plot_case(cid, ax)\n"
            "plt.suptitle('Day-1 forecast case studies')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("## 11. Research question answers"),
        _code(
            "for key, block in ANSWERS.items():\n"
            "    print(f\"{key}: {block['answer']}\")\n"
            "    print(f\"  {block['evidence']}\")\n"
            "    print()"
        ),
        _md("## 12. Consolidated conclusion"),
        _md(
            "The full research arc — Stage 0 (representation diagnostics) and RRE v1.0 "
            "(scaling experiment) — is documented in:\n\n"
            "`docs/hybrid_residual_investigation/CONCLUSION.md`\n\n"
            "**Thesis statement:** The Prophet complement residual is best modelled in "
            "level form with global z-score normalisation. Velocity was rejected (Stage 0). "
            "Robust median/MAD scaling did not improve forecasts (RRE v1.0). Residual "
            "preprocessing is not the Hybrid bottleneck under the frozen protocol."
        ),
        _md("## 13. Discussion"),
        _md(
            "**Interpretation:** If R3 and R0 produce similar accuracy but different "
            "optimisation curves, the experiment supports the hypothesis that scaling "
            "affects gradient conditioning without changing learnable temporal content. "
            "If R3 does not improve accuracy or convergence, the frozen Hybrid z-score "
            "baseline remains scientifically justified.\n\n"
            "Negative results are valuable: they confirm that residual representation "
            "was not the bottleneck — consistent with HCERL, peak-aware, and window experiments."
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
