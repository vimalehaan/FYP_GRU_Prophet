#!/usr/bin/env python3
"""Build notebooks/context_feature_feasibility.ipynb (manual execution, no training)."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "context_feature_feasibility.ipynb"
DOCS_DIR = "docs/context_feature_feasibility"
TMA_EXP = "experiments/temporal_memory_analysis_2026-07-26_170052"


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
            "# Context Feature Feasibility Study (Hybrid v2 Design)\n\n"
            "**Read-only diagnostic.** No model training. No new experiments. "
            "No synthetic data.\n\n"
            "Determines whether a context-enriched Hybrid Prophet + GRU is "
            "scientifically justified using the existing Alibaba dataset and "
            "prior research (RPA, CSRLE, LFHE, TMA, RLLA).\n\n"
            "Documentation: `docs/context_feature_feasibility/`"
        ),
        _md("## 1. Setup"),
        _code(
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display, Markdown\n\n"
            "REPO_ROOT = Path('..').resolve()\n"
            "sys.path.insert(0, str(REPO_ROOT))\n\n"
            f"DOCS = REPO_ROOT / '{DOCS_DIR}'\n"
            f"FIGURES = DOCS / 'figures'\n"
            f"CACHE = DOCS / '_analysis_cache'\n"
            f"TMA_DIR = REPO_ROOT / '{TMA_EXP}'\n\n"
            "plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})\n"
            "print('Repo:', REPO_ROOT)"
        ),
        _md("## 2. Dataset inspection (Task 1)"),
        _code(
            "train_df = pd.read_parquet(REPO_ROOT / 'data/train_df.parquet')\n"
            "val_df = pd.read_parquet(REPO_ROOT / 'data/val_df.parquet')\n"
            "selected = np.load(REPO_ROOT / 'data/selected_containers.npy', allow_pickle=True)\n\n"
            "print('Train shape:', train_df.shape)\n"
            "print('Val shape:', val_df.shape)\n"
            "print('Columns:', list(train_df.columns))\n"
            "print('Selected containers:', len(selected))\n\n"
            "null_pct = (train_df.isna().mean() * 100).round(1)\n"
            "null_pct.sort_values(ascending=False)"
        ),
        _code(
            "schema = pd.DataFrame({\n"
            "    'column': train_df.columns,\n"
            "    'dtype': train_df.dtypes.astype(str).values,\n"
            "    'train_null_pct': (train_df.isna().mean() * 100).round(1).values,\n"
            "})\n"
            "hybrid_cols = {'container_id', 'time_stamp', 'cpu_scaled'}\n"
            "global_cols = hybrid_cols | {'cpu_mean', 'cpu_std'}\n"
            "schema['used_in_hybrid'] = schema['column'].isin(hybrid_cols)\n"
            "schema['used_in_global_gru'] = schema['column'].isin(global_cols)\n"
            "schema"
        ),
        _md("## 3. Prophet-equivalent residuals (read-only, no Prophet.fit)"),
        _code(
            "from utils.tma.data import prophet_equivalent_residuals\n\n"
            "cid = 'c_10032'\n"
            "tr = train_df[train_df.container_id == cid].sort_values('time_stamp')\n"
            "va = val_df[val_df.container_id == cid].sort_values('time_stamp')\n\n"
            "train_res, val_res, full_res = prophet_equivalent_residuals(\n"
            "    tr.cpu_scaled.values, va.cpu_scaled.values\n"
            ")\n"
            "print('Train residuals:', train_res.shape)\n"
            "print('Val residuals:', val_res.shape)\n"
            "print('Val residual mean:', val_res.mean(), 'std:', val_res.std())"
        ),
        _md("## 4. Feature derivation examples (Task 2)"),
        _code(
            "def derive_context_features(tr, va, y_tr, y_va, full_res):\n"
            "    \"\"\"Example causal feature derivation for one container (no leakage).\"\"\"\n"
            "    ts = pd.concat([tr.time_stamp, va.time_stamp], ignore_index=True)\n"
            "    cpu = np.concatenate([y_tr, y_va])\n"
            "    res = full_res\n"
            "    pred = cpu - res\n"
            "    s = pd.Series(res)\n"
            "    p90 = float(tr.cpu_scaled.quantile(0.9))\n"
            "    out = pd.DataFrame({\n"
            "        'time_stamp': ts,\n"
            "        'cpu_scaled': cpu,\n"
            "        'prophet_pred': pred,\n"
            "        'residual': res,\n"
            "        'hour': ts.dt.hour,\n"
            "        'hour_sin': np.sin(2 * np.pi * ts.dt.hour / 24),\n"
            "        'hour_cos': np.cos(2 * np.pi * ts.dt.hour / 24),\n"
            "        'res_roll_std_12': s.rolling(12, min_periods=1).std().fillna(0),\n"
            "        'res_energy_96': (s ** 2).rolling(96, min_periods=1).mean(),\n"
            "        'res_lag96': s.shift(96).fillna(0),\n"
            "        'cpu_std': tr.cpu_std.iloc[0],\n"
            "        'is_peak_p90': (cpu >= p90).astype(float),\n"
            "    })\n"
            "    return out\n\n"
            "feat_df = derive_context_features(tr, va, tr.cpu_scaled.values, va.cpu_scaled.values, full_res)\n"
            "feat_df.tail(8)"
        ),
        _md("## 5. Temporal Memory Analysis integration (Task 4)"),
        _code(
            "tma_report = TMA_DIR / 'reports' / 'final_report.json'\n"
            "if tma_report.exists():\n"
            "    with tma_report.open() as fh:\n"
            "        tma = json.load(fh)\n"
            "    key = tma.get('key_metrics', {})\n"
            "    pd.DataFrame([{\n"
            "        'signal': 'CPU original lag-96 ACF',\n"
            "        'value': key.get('cpu_lag96_acf_mean'),\n"
            "    }, {\n"
            "        'signal': 'Prophet-equivalent residual lag-96 ACF',\n"
            "        'value': key.get('prophet_lag96_acf_mean'),\n"
            "    }, {\n"
            "        'signal': 'Hybrid post-forecast decorrelation steps',\n"
            "        'value': key.get('hybrid_error_memory_steps'),\n"
            "    }])\n"
            "else:\n"
            "    print('TMA report not found — see docs/temporal_memory_analysis/')"
        ),
        _code(
            "acf_path = FIGURES / 'acf_cpu_vs_prophet_residual_example.png'\n"
            "if acf_path.exists():\n"
            "    display(Image(filename=str(acf_path)))\n"
            "else:\n"
            "    from statsmodels.tsa.stattools import acf\n"
            "    acf_cpu = acf(va.cpu_scaled.values, nlags=120, fft=True)\n"
            "    acf_res = acf(val_res, nlags=120, fft=True)\n"
            "    fig, ax = plt.subplots(figsize=(8, 3.5))\n"
            "    ax.plot(acf_cpu, label='cpu_scaled')\n"
            "    ax.plot(acf_res, label='Prophet-equivalent residual')\n"
            "    ax.axvline(96, color='gray', ls='--', label='lag 96')\n"
            "    ax.legend(); ax.set(xlabel='Lag', ylabel='ACF'); plt.show()"
        ),
        _md("## 6. Correlation analysis — context vs residual (Task 3, 6)"),
        _code(
            "ctx_csv = CACHE / 'context_feature_residual_correlation.csv'\n"
            "if ctx_csv.exists():\n"
            "    ctx_summary = pd.read_csv(ctx_csv)\n"
            "else:\n"
            "    rows = []\n"
            "    for cid in selected:\n"
            "        tr_i = train_df[train_df.container_id == cid].sort_values('time_stamp')\n"
            "        va_i = val_df[val_df.container_id == cid].sort_values('time_stamp')\n"
            "        if len(tr_i) < 200 or len(va_i) < 50:\n"
            "            continue\n"
            "        y_tr, y_va = tr_i.cpu_scaled.values, va_i.cpu_scaled.values\n"
            "        _, _, full = prophet_equivalent_residuals(y_tr, y_va)\n"
            "        fd = derive_context_features(tr_i, va_i, y_tr, y_va, full)\n"
            "        sub = fd.iloc[len(tr_i):]\n"
            "        for col in sub.columns:\n"
            "            if col in {'time_stamp', 'cpu_scaled', 'prophet_pred', 'residual'}:\n"
            "                continue\n"
            "            if sub[col].std() < 1e-12:\n"
            "                continue\n"
            "            r = np.corrcoef(sub[col], sub['residual'])[0, 1]\n"
            "            rows.append({'feature': col, 'pearson_r': r, 'container_id': cid})\n"
            "    ctx_summary = pd.DataFrame(rows).groupby('feature')['pearson_r'].agg(['mean','std']).reset_index()\n"
            "    ctx_summary['abs_mean'] = ctx_summary['mean'].abs()\n\n"
            "ctx_summary.sort_values('abs_mean', ascending=False)"
        ),
        _code(
            "fig_path = FIGURES / 'context_feature_residual_correlation.png'\n"
            "if fig_path.exists():\n"
            "    display(Image(filename=str(fig_path)))\n"
            "else:\n"
            "    plot_df = ctx_summary.sort_values('abs_mean')\n"
            "    fig, ax = plt.subplots(figsize=(8, 4))\n"
            "    ax.barh(plot_df['feature'], plot_df['mean'])\n"
            "    ax.axvline(0, color='k', lw=0.8)\n"
            "    ax.set_title('Context feature correlation with residual')\n"
            "    plt.show()"
        ),
        _md("## 7. Residual-derived redundancy (Task 5–6)"),
        _code(
            "red_path = CACHE / 'feature_residual_correlation_summary.csv'\n"
            "if red_path.exists():\n"
            "    redundant = pd.read_csv(red_path)\n"
            "    display(redundant.sort_values('abs_mean', ascending=False).head(12))\n"
            "    display(Image(filename=str(FIGURES / 'residual_derived_feature_correlation.png')))\n"
            "else:\n"
            "    Markdown('Run read-only analysis script to populate cache (see docs README).')"
        ),
        _md(
            "**Interpretation:** High correlation for rolling mean, diff, and lag features "
            "reflects algebraic dependence on the residual series — the GRU already receives "
            "96 consecutive residuals. These are **redundant channels**, not new information."
        ),
        _md("## 8. Inter-feature correlation — compact set (Task 6)"),
        _code(
            "compact_path = FIGURES / 'compact_feature_intercorrelation.png'\n"
            "if compact_path.exists():\n"
            "    display(Image(filename=str(compact_path)))\n"
            "else:\n"
            "    pool = []\n"
            "    for cid in selected[:10]:\n"
            "        tr_i = train_df[train_df.container_id == cid].sort_values('time_stamp')\n"
            "        va_i = val_df[val_df.container_id == cid].sort_values('time_stamp')\n"
            "        y_tr, y_va = tr_i.cpu_scaled.values, va_i.cpu_scaled.values\n"
            "        _, _, full = prophet_equivalent_residuals(y_tr, y_va)\n"
            "        fd = derive_context_features(tr_i, va_i, y_tr, y_va, full)\n"
            "        pool.append(fd.iloc[len(tr_i):][[\n"
            "            'prophet_pred','res_roll_std_12','res_energy_96','res_lag96','hour_sin','cpu_std'\n"
            "        ]])\n"
            "    cm = pd.concat(pool, ignore_index=True).corr()\n"
            "    fig, ax = plt.subplots(figsize=(7, 5))\n"
            "    im = ax.imshow(cm.values, cmap='RdBu_r', vmin=-1, vmax=1)\n"
            "    ax.set_xticks(range(len(cm.columns)), cm.columns, rotation=45, ha='right')\n"
            "    ax.set_yticks(range(len(cm.columns)), cm.columns)\n"
            "    for i in range(len(cm)):\n"
            "        for j in range(len(cm)):\n"
            "            ax.text(j, i, f'{cm.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)\n"
            "    ax.set_title('Compact candidate inter-correlation')\n"
            "    fig.colorbar(im, ax=ax, fraction=0.046)\n"
            "    plt.tight_layout(); plt.show()"
        ),
        _md("## 9. Comparison tables"),
        _code(
            "comparison = pd.DataFrame([\n"
            "    {'feature': 'residual_scaled', 'new_info': 'baseline', 'prophet_dup': 'No',\n"
            "     'tma_support': '96-step window OK', 'recommended': 'Yes (ch0)'},\n"
            "    {'feature': 'prophet_yhat_scaled', 'new_info': 'Level/regime', 'prophet_dup': 'Partial',\n"
            "     'tma_support': 'Level not in residual', 'recommended': 'Yes (ch1)'},\n"
            "    {'feature': 'res_roll_std_12', 'new_info': 'Local volatility', 'prophet_dup': 'No',\n"
            "     'tma_support': 'Short-memory vol', 'recommended': 'Yes (ch2)'},\n"
            "    {'feature': 'cpu_std', 'new_info': 'Container scale', 'prophet_dup': 'No',\n"
            "     'tma_support': 'Global GRU precedent', 'recommended': 'Yes (ch3)'},\n"
            "    {'feature': 'is_peak_p90', 'new_info': 'Peak regime', 'prophet_dup': 'No',\n"
            "     'tma_support': 'Peak-aware precedent', 'recommended': 'Yes (ch4)'},\n"
            "    {'feature': 'hour_sin/cos', 'new_info': 'Calendar', 'prophet_dup': 'High',\n"
            "     'tma_support': 'lag-96 ACF ~ 0.04', 'recommended': 'Optional ablation'},\n"
            "    {'feature': 'res_roll_mean_12', 'new_info': 'None (derivable)', 'prophet_dup': 'No',\n"
            "     'tma_support': 'Redundant with window', 'recommended': 'No'},\n"
            "    {'feature': 'mem_util_percent', 'new_info': 'Unknown', 'prophet_dup': 'No',\n"
            "     'tma_support': '98% missing', 'recommended': 'No'},\n"
            "])\n"
            "comparison"
        ),
        _md("## 10. Hybrid v2 input representation (Task 7)"),
        _code(
            "tensor_doc = pd.DataFrame({\n"
            "    'channel': [0, 1, 2, 3, 4],\n"
            "    'name': ['residual_scaled', 'prophet_yhat_scaled', 'res_roll_std_12', 'cpu_std', 'is_peak_p90'],\n"
            "    'shape_per_step': ['scalar', 'scalar', 'scalar', 'scalar (broadcast)', 'binary'],\n"
            "    'window': ['t-95..t', 't-95..t', 't-95..t', 'constant', 't-95..t'],\n"
            "})\n"
            "print('Input tensor shape per sequence: (96, 5)')\n"
            "tensor_doc"
        ),
        _code(
            "# Illustrative slice (last 5 steps of 96-step window)\n"
            "window = feat_df.iloc[len(tr)-5:len(tr)+1].copy()\n"
            "window['residual_scaled'] = (window['residual'] - train_res.mean()) / train_res.std()\n"
            "cols = ['residual_scaled', 'prophet_pred', 'res_roll_std_12', 'cpu_std', 'is_peak_p90']\n"
            "window[cols].round(3)"
        ),
        _md("## 11. Feature importance — conceptual only (Task 8)"),
        _md(
            "No model training in this study. Conceptual importance based on research chain:\n\n"
            "| Evidence | Implication |\n"
            "|----------|-------------|\n"
            "| RPA: Ljung–Box reject 73.7% | Structure exists — GRU should benefit if given right inputs |\n"
            "| CSRLE: GRU learns synthetic B0 | Architecture capable; real signal weaker |\n"
            "| LFHE: r did not improve with DA-MSE | Context alone may not fix objective issue |\n"
            "| TMA: residual lag-96 ≈ 0.04 | Calendar features low priority |\n"
            "| RLLA: linear r ≈ 0.02–0.06 | Upper bound on easy linear gains is low |\n\n"
            "**Conceptual ranking:** prophet level > volatility > peak regime > cpu_std > calendar"
        ),
        _md("## 12. Prior research metrics (frozen references)"),
        _code(
            "prior = pd.DataFrame([\n"
            "    {'study': 'RPA', 'metric': 'Ljung-Box reject rate', 'value': '~73.7%'},\n"
            "    {'study': 'RPA', 'metric': 'Hybrid GRU residual Pearson r', 'value': '~0.02'},\n"
            "    {'study': 'CSRLE', 'metric': 'GRU learns synthetic B0', 'value': 'Yes'},\n"
            "    {'study': 'LFHE', 'metric': 'DA-MSE std ratio change', 'value': '+0.73'},\n"
            "    {'study': 'LFHE', 'metric': 'DA-MSE Pearson r change', 'value': '0.113→0.087'},\n"
            "    {'study': 'TMA', 'metric': 'CPU lag-96 ACF', 'value': '~0.33'},\n"
            "    {'study': 'TMA', 'metric': 'Prophet residual lag-96 ACF', 'value': '~0.04'},\n"
            "    {'study': 'RLLA', 'metric': 'Ridge best Pearson r (real)', 'value': '~0.059'},\n"
            "    {'study': 'Ridge audit', 'metric': 'Ridge std ratio B0', 'value': '~0.22'},\n"
            "    {'study': 'Ridge audit', 'metric': 'GRU std ratio B0', 'value': '~0.07'},\n"
            "])\n"
            "prior"
        ),
        _md("## 13. Final conclusions"),
        _md(
            "### Explicit answers\n\n"
            "1. **Dataset sufficient?** Yes for CPU-side context; no for multivariate telemetry.\n\n"
            "2. **Genuinely new features?** `prophet_yhat_scaled`, `res_roll_std_12`, `cpu_std`, "
            "`is_peak_p90`. Calendar and residual-transform features are largely duplicative.\n\n"
            "3. **Recommended 3–5 features:** prophet_yhat_scaled, res_roll_std_12, cpu_std, "
            "is_peak_p90 (+ optional hour_sin/cos).\n\n"
            "4. **Scientifically justified?** **Yes, conditionally** — investigate compact Hybrid v2; "
            "expect incremental not breakthrough gains (RLLA, LFHE).\n\n"
            "See `docs/context_feature_feasibility/summary.md` for full narrative."
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
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    build_notebook()
