#!/usr/bin/env python3
"""Build production/notebooks/production_hybrid_model.ipynb."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK_PATH = REPO_ROOT / "production" / "notebooks" / "production_hybrid_model.ipynb"


def _md(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def _code(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def build_notebook() -> None:
    cells = [
        _md(
            "# Production Hybrid Prophet + GRU Model\n\n"
            "**Purpose:** Demonstrate the deployable production Hybrid model trained on "
            "the complete high-quality dataset with a fixed container-level holdout for "
            "generalization evaluation.\n\n"
            "**Important:** This notebook loads saved production artifacts and "
            "**generates all visualizations in-notebook**. It does not display pre-rendered PNGs.\n\n"
            "Training is executed only via `production/scripts/train_production_hybrid.py`."
        ),
        _code(
            "import json\n"
            "import pickle\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import tensorflow as tf\n"
            "from IPython.display import display\n\n"
            "# Resolve repo root whether the kernel cwd is repo/ or repo/notebooks/\n"
            "_candidates = [\n"
            "    Path.cwd(),\n"
            "    Path.cwd().parent,\n"
            "    Path('..').resolve(),\n"
            "    Path('.').resolve(),\n"
            "]\n"
            "REPO_ROOT = next(\n"
            "    (p for p in _candidates if (p / 'production' / 'hybrid' / 'models' / 'model.keras').exists()),\n"
            "    Path('../..').resolve(),\n"
            ")\n"
            "if str(REPO_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(REPO_ROOT))\n\n"
            "PROD = REPO_ROOT / 'production' / 'hybrid'\n"
            "if not PROD.exists():\n"
            "    raise FileNotFoundError(\n"
            "        'Production artifacts missing. Run: '\n"
            "        'tf_metal_env/bin/python production/scripts/train_production_hybrid.py'\n"
            "    )\n\n"
            "try:\n"
            "    plt.style.use('seaborn-v0_8-whitegrid')\n"
            "except OSError:\n"
            "    plt.style.use('ggplot')\n"
            "FIG_DPI = 120\n"
            "PALETTE = {'known': '#2563eb', 'unseen': '#f97316'}\n\n"
            "print('Repo root:', REPO_ROOT)\n"
            "print('Hybrid artifacts:', PROD)\n\n"
            "from production.utils.artifacts import (\n"
            "    load_production_scalers,\n"
            "    load_residual_stats,\n"
            ")\n"
            "from production.utils.inference import (\n"
            "    run_known_container_inference,\n"
            "    run_unseen_container_inference,\n"
            ")\n\n"
            "def pick_container(preferred, available, label):\n"
            "    \"\"\"Resolve a user-entered container ID against an allowed pool.\"\"\"\n"
            "    available = sorted(str(c) for c in available)\n"
            "    if not available:\n"
            "        raise ValueError(f'No {label} containers available.')\n"
            "    if preferred is None or str(preferred).strip() == '':\n"
            "        chosen = available[0]\n"
            "        print(f'No {label} ID set — using default: {chosen}')\n"
            "        return chosen\n"
            "    preferred = str(preferred).strip()\n"
            "    if preferred not in available:\n"
            "        partial = [c for c in available if preferred.lower() in c.lower()][:10]\n"
            "        msg = (\n"
            "            f\"Unknown {label} container: {preferred!r}\\n\"\n"
            "            f'Available: {len(available)} containers\\n'\n"
            "            f'Examples: {available[:10]}'\n"
            "        )\n"
            "        if partial:\n"
            "            msg += f'\\nPartial matches: {partial}'\n"
            "        raise ValueError(msg)\n"
            "    print(f'Selected {label} container: {preferred}')\n"
            "    return preferred\n\n"
            "def search_containers(query, available, limit=20):\n"
            "    \"\"\"Return container IDs containing the query substring.\"\"\"\n"
            "    q = str(query).strip().lower()\n"
            "    matches = [c for c in sorted(available) if q in str(c).lower()]\n"
            "    return matches[:limit]\n\n"
            "def plot_container_forecast(result, title_prefix='Container'):\n"
            "    \"\"\"Generate forecast, residual, and error plots for one container.\"\"\"\n"
            "    cid = result.container_id\n"
            "    ts = result.val_container['time_stamp'].values[:96]\n"
            "    actual = np.ravel(result.actual_day1_real)\n"
            "    hybrid = np.ravel(result.day1_final_real)\n"
            "    prophet = np.ravel(result.prophet_day1_real)\n"
            "    errors = hybrid - actual\n\n"
            "    fig, axes = plt.subplots(2, 2, figsize=(14, 8))\n"
            "    axes[0, 0].plot(ts, actual, label='Actual', color='#2563eb', lw=1.8)\n"
            "    axes[0, 0].plot(ts, hybrid, label='Hybrid', color='#ef4444', ls='--', lw=1.8)\n"
            "    axes[0, 0].plot(ts, prophet, label='Prophet', color='#16a34a', alpha=0.75)\n"
            "    axes[0, 0].set_title(f'{title_prefix} {cid} — Day-1 CPU forecast')\n"
            "    axes[0, 0].set_ylabel('CPU (%)')\n"
            "    axes[0, 0].legend()\n\n"
            "    axes[0, 1].plot(ts, errors, color='#9333ea', lw=1.5)\n"
            "    axes[0, 1].axhline(0, color='black', ls='--', alpha=0.4)\n"
            "    axes[0, 1].set_title('Forecast error (Hybrid − Actual)')\n"
            "    axes[0, 1].set_ylabel('Error (%)')\n\n"
            "    axes[1, 0].plot(ts, result.day1_residual, color='#7c3aed', lw=1.5)\n"
            "    axes[1, 0].set_title('GRU residual component')\n"
            "    axes[1, 0].set_ylabel('Scaled residual')\n"
            "    axes[1, 0].set_xlabel('Time')\n\n"
            "    axes[1, 1].scatter(actual, hybrid, alpha=0.65, s=18, color='#2563eb')\n"
            "    lim = max(actual.max(), hybrid.max())\n"
            "    axes[1, 1].plot([0, lim], [0, lim], 'r--', lw=1)\n"
            "    axes[1, 1].set_xlabel('Actual CPU (%)')\n"
            "    axes[1, 1].set_ylabel('Predicted CPU (%)')\n"
            "    axes[1, 1].set_title('Actual vs Hybrid scatter')\n\n"
            "    for ax in axes[0, :].flat:\n"
            "        ax.tick_params(axis='x', rotation=20)\n"
            "    plt.tight_layout()\n"
            "    plt.show()\n\n"
            "def container_metrics_table(eval_df, container_id):\n"
            "    cols = [\n"
            "        'container_id', 'split', 'train_steps', 'validation_steps',\n"
            "        'day1_mae', 'day1_rmse', 'day1_mape', 'day1_pearson_r', 'day1_r2',\n"
            "    ]\n"
            "    row = eval_df.loc[eval_df['container_id'] == container_id, cols]\n"
            "    if row.empty:\n"
            "        print(f'No saved metrics for {container_id!r}. Metrics below are computed live.')\n"
            "        return None\n"
            "    return row.reset_index(drop=True)\n"
        ),
        _md("## 1. Introduction"),
        _code(
            "with (PROD / 'config' / 'production_config.json').open() as fh:\n"
            "    prod_config = json.load(fh)\n"
            "with (PROD / 'config' / 'training_metadata.json').open() as fh:\n"
            "    train_meta = json.load(fh)\n"
            "with (PROD / 'metadata' / 'container_split_metadata.json').open() as fh:\n"
            "    split_meta = json.load(fh)\n\n"
            "print('Model type:', prod_config['model_type'])\n"
            "print('Input window:', prod_config['input_window'])\n"
            "print('Forecast horizon:', prod_config['forecast_horizon'])\n"
            "print('Train containers:', split_meta['n_train_containers'])\n"
            "print('Unseen containers:', split_meta['n_unseen_containers'])"
        ),
        _code(
            "fig, ax = plt.subplots(figsize=(12, 2.5))\n"
            "stages = [\n"
            "    'High-quality\\ncontainers',\n"
            "    '90% production\\ntraining',\n"
            "    '10% unseen\\nholdout',\n"
            "    'Per-container\\n80/20 temporal',\n"
            "    'Prophet + GRU\\ntraining',\n"
            "    'Production\\nevaluation',\n"
            "]\n"
            "x = np.arange(len(stages))\n"
            "ax.barh(0, len(stages), color='none', edgecolor='none')\n"
            "for i, label in enumerate(stages):\n"
            "    ax.text(i, 0, label, ha='center', va='center', fontsize=10,\n"
            "            bbox=dict(boxstyle='round', facecolor='#dbeafe', edgecolor='#3b82f6'))\n"
            "    if i < len(stages) - 1:\n"
            "        ax.annotate('', xy=(i + 0.45, 0), xytext=(i + 0.55, 0),\n"
            "                    arrowprops=dict(arrowstyle='->', color='#64748b'))\n"
            "ax.set_xlim(-0.5, len(stages) - 0.5)\n"
            "ax.set_ylim(-0.5, 0.5)\n"
            "ax.axis('off')\n"
            "ax.set_title('Production Hybrid Pipeline Overview', fontsize=13, fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("## 2. Production Dataset"),
        _code(
            "with (PROD / 'metadata' / 'train_container_ids.json').open() as fh:\n"
            "    train_ids = json.load(fh)['container_ids']\n"
            "with (PROD / 'metadata' / 'unseen_container_ids.json').open() as fh:\n"
            "    unseen_ids = json.load(fh)['container_ids']\n\n"
            "train_df = pd.read_parquet(PROD / 'cache' / 'train_df.parquet')\n"
            "val_df = pd.read_parquet(PROD / 'cache' / 'val_df.parquet')\n"
            "unseen_raw = pd.read_parquet(PROD / 'cache' / 'unseen_resampled.parquet')\n\n"
            "dataset_summary = pd.DataFrame([\n"
            "    {'split': 'Production train (80%)', 'containers': train_df['container_id'].nunique(),\n"
            "     'rows': len(train_df), 'period': 'historical 80% per container'},\n"
            "    {'split': 'Production val (20%)', 'containers': val_df['container_id'].nunique(),\n"
            "     'rows': len(val_df), 'period': 'temporal holdout 20%'},\n"
            "    {'split': 'Unseen holdout (raw)', 'containers': unseen_raw['container_id'].nunique(),\n"
            "     'rows': len(unseen_raw), 'period': 'full timeline (eval uses 80/20 locally)'},\n"
            "])\n"
            "display(dataset_summary)"
        ),
        _code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "split_counts = pd.Series({'Train containers': len(train_ids), 'Unseen containers': len(unseen_ids)})\n"
            "split_counts.plot(kind='bar', ax=axes[0], color=['#2563eb', '#f97316'], rot=0)\n"
            "axes[0].set_title('Container-Level Split (90/10)')\n"
            "axes[0].set_ylabel('Number of containers')\n\n"
            "row_counts = pd.Series({\n"
            "    'Train period': len(train_df),\n"
            "    'Val period': len(val_df),\n"
            "    'Unseen raw': len(unseen_raw),\n"
            "})\n"
            "row_counts.plot(kind='bar', ax=axes[1], color=['#16a34a', '#eab308', '#f97316'], rot=0)\n"
            "axes[1].set_title('Row Counts by Dataset Partition')\n"
            "axes[1].set_ylabel('Rows')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _code(
            "sample_cid = train_df['container_id'].iloc[0]\n"
            "g = pd.concat([\n"
            "    train_df[train_df['container_id'] == sample_cid].assign(partition='train'),\n"
            "    val_df[val_df['container_id'] == sample_cid].assign(partition='val'),\n"
            "])\n"
            "fig, ax = plt.subplots(figsize=(12, 3))\n"
            "for part, color in [('train', '#2563eb'), ('val', '#ef4444')]:\n"
            "    sub = g[g['partition'] == part]\n"
            "    ax.plot(sub['time_stamp'], sub['cpu'], label=part, color=color, alpha=0.8)\n"
            "ax.axvline(g[g['partition'] == 'train']['time_stamp'].max(), color='black', ls='--', alpha=0.5)\n"
            "ax.set_title(f'Example temporal split — container {sample_cid}')\n"
            "ax.set_xlabel('Time')\n"
            "ax.set_ylabel('CPU usage (%)')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md("## 3. Training Summary"),
        _code(
            "history = pd.read_csv(PROD / 'logs' / 'training_history.csv')\n"
            "display(history.tail())\n"
            "print(f\"Best epoch: {train_meta['best_epoch']}  |  \"\n"
            "      f\"Best val loss: {train_meta['best_val_loss']:.6f}\")"
        ),
        _code(
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "axes[0].plot(history['epoch'], history['loss'], label='Train loss')\n"
            "axes[0].plot(history['epoch'], history['val_loss'], label='Val loss')\n"
            "axes[0].axvline(train_meta['best_epoch'], color='gray', ls='--', label='Best epoch')\n"
            "axes[0].set_title('Training & Validation Loss (MSE)')\n"
            "axes[0].set_xlabel('Epoch')\n"
            "axes[0].legend()\n\n"
            "axes[1].plot(history['epoch'], history['mae'], label='Train MAE')\n"
            "axes[1].plot(history['epoch'], history['val_mae'], label='Val MAE')\n"
            "axes[1].axvline(train_meta['best_epoch'], color='gray', ls='--', label='Best epoch')\n"
            "axes[1].set_title('Training & Validation MAE')\n"
            "axes[1].set_xlabel('Epoch')\n"
            "axes[1].legend()\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _code(
            "model = tf.keras.models.load_model(str(PROD / 'models' / 'model.keras'))\n"
            "model.summary()\n"
            "n_params = model.count_params()\n"
            "print(f'Total parameters: {n_params:,}')"
        ),
        _md("## 4. Production Evaluation"),
        _code(
            "eval_df = pd.read_csv(PROD / 'metrics' / 'evaluation_results.csv')\n"
            "display(eval_df.groupby('split').agg(\n"
            "    containers=('container_id', 'count'),\n"
            "    mae_mean=('day1_mae', 'mean'),\n"
            "    rmse_mean=('day1_rmse', 'mean'),\n"
            "    mape_mean=('day1_mape', 'mean'),\n"
            "    pearson_mean=('day1_pearson_r', 'mean'),\n"
            "    r2_mean=('day1_r2', 'mean'),\n"
            ").round(4))"
        ),
        _code(
            "def _split_boxplot(ax, df, col, title):\n"
            "    groups = [df.loc[df['split'] == s, col].dropna().values for s in ('known', 'unseen')]\n"
            "    bp = ax.boxplot(groups, patch_artist=True)\n"
            "    ax.set_xticks([1, 2])\n"
            "    ax.set_xticklabels(['known', 'unseen'])\n"
            "    for patch, split in zip(bp['boxes'], ('known', 'unseen')):\n"
            "        patch.set_facecolor(PALETTE[split])\n"
            "        patch.set_alpha(0.55)\n"
            "    ax.set_title(title)\n"
            "    ax.set_xlabel('Split')\n\n"
            "fig, axes = plt.subplots(2, 2, figsize=(12, 8))\n"
            "for ax, col, title in zip(\n"
            "    axes.ravel(),\n"
            "    ['day1_mae', 'day1_rmse', 'day1_mape', 'day1_r2'],\n"
            "    ['Day-1 MAE', 'Day-1 RMSE', 'Day-1 MAPE (%)', 'Day-1 R²'],\n"
            "):\n"
            "    _split_boxplot(ax, eval_df, col, title)\n"
            "plt.suptitle('Production Evaluation by Split', fontweight='bold')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _code(
            "known_dir = PROD / 'predictions' / 'known'\n"
            "all_actual, all_pred = [], []\n"
            "for pkl in sorted(known_dir.glob('*.pkl'))[:50]:\n"
            "    with pkl.open('rb') as fh:\n"
            "        rec = pickle.load(fh)\n"
            "    all_actual.extend(rec['actual_day1_real'].ravel())\n"
            "    all_pred.extend(rec['day1_final_real'].ravel())\n"
            "all_actual = np.array(all_actual)\n"
            "all_pred = np.array(all_pred)\n"
            "errors = all_pred - all_actual\n\n"
            "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
            "axes[0].scatter(all_actual, all_pred, alpha=0.15, s=8)\n"
            "mx = max(all_actual.max(), all_pred.max())\n"
            "axes[0].plot([0, mx], [0, mx], 'r--', label='Perfect')\n"
            "axes[0].set_xlabel('Actual CPU (%)')\n"
            "axes[0].set_ylabel('Predicted CPU (%)')\n"
            "axes[0].set_title('Actual vs Predicted (known, sample)')\n"
            "axes[0].legend()\n\n"
            "axes[1].hist(errors, bins=40, color='#6366f1', edgecolor='white')\n"
            "axes[1].set_title('Prediction Error Distribution')\n"
            "axes[1].set_xlabel('Error (pred - actual)')\n\n"
            "axes[2].scatter(all_actual, errors, alpha=0.15, s=8)\n"
            "axes[2].axhline(0, color='r', ls='--')\n"
            "axes[2].set_xlabel('Actual CPU (%)')\n"
            "axes[2].set_ylabel('Residual error')\n"
            "axes[2].set_title('Residual Plot')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _md(
            "## 5. Known Container — pick your container\n\n"
            "Edit `PREFERRED_KNOWN_CONTAINER` below to any training-container ID "
            "(e.g. `\"c_1016\"`). Leave as `None` to use the first available container.\n\n"
            "Re-run **this cell and the next cell** after changing the ID."
        ),
        _code(
            "scalers = load_production_scalers()\n"
            "res_mean, res_std, input_window, _ = load_residual_stats()\n\n"
            "known_cids = eval_df.loc[eval_df['split'] == 'known', 'container_id'].unique()\n"
            "unseen_cids = eval_df.loc[eval_df['split'] == 'unseen', 'container_id'].unique()\n\n"
            "# ===== EDIT THIS =====\n"
            "PREFERRED_KNOWN_CONTAINER = None  # e.g. 'c_1016'\n"
            "# =====================\n\n"
            "SELECTED_KNOWN = pick_container(PREFERRED_KNOWN_CONTAINER, known_cids, 'known')\n"
            "print('Available known containers:', len(known_cids))\n"
            "print('Search tip: search_containers(\"1016\", known_cids)')"
        ),
        _code(
            "known_result = run_known_container_inference(\n"
            "    SELECTED_KNOWN, train_df, val_df, model, scalers, res_mean, res_std, input_window\n"
            ")\n"
            "plot_container_forecast(known_result, title_prefix='Known container')\n\n"
            "metrics = container_metrics_table(eval_df, SELECTED_KNOWN)\n"
            "if metrics is not None:\n"
            "    display(metrics)\n\n"
            "live = pd.DataFrame([{\n"
            "    'container_id': SELECTED_KNOWN,\n"
            "    'live_mae': float(np.mean(np.abs(\n"
            "        np.ravel(known_result.day1_final_real) - np.ravel(known_result.actual_day1_real)\n"
            "    ))),\n"
            "    'live_rmse': float(np.sqrt(np.mean((\n"
            "        np.ravel(known_result.day1_final_real) - np.ravel(known_result.actual_day1_real)\n"
            "    ) ** 2))),\n"
            "}])\n"
            "display(live)"
        ),
        _md(
            "## 6. Unseen Container — pick your container\n\n"
            "Edit `PREFERRED_UNSEEN_CONTAINER` to any holdout-container ID. "
            "Leave as `None` for the default first unseen container.\n\n"
            "Re-run **this cell and the next cell** after changing the ID."
        ),
        _code(
            "# ===== EDIT THIS =====\n"
            "PREFERRED_UNSEEN_CONTAINER = None  # e.g. 'c_23456'\n"
            "# =====================\n\n"
            "SELECTED_UNSEEN = pick_container(PREFERRED_UNSEEN_CONTAINER, unseen_cids, 'unseen')\n"
            "print('Available unseen containers:', len(unseen_cids))\n"
            "print('Search tip: search_containers(\"2345\", unseen_cids)')"
        ),
        _code(
            "unseen_result = run_unseen_container_inference(\n"
            "    SELECTED_UNSEEN, unseen_raw, model, res_mean, res_std, input_window\n"
            ")\n"
            "plot_container_forecast(unseen_result, title_prefix='Unseen container')\n\n"
            "metrics = container_metrics_table(eval_df, SELECTED_UNSEEN)\n"
            "if metrics is not None:\n"
            "    display(metrics)\n\n"
            "live = pd.DataFrame([{\n"
            "    'container_id': SELECTED_UNSEEN,\n"
            "    'live_mae': float(np.mean(np.abs(\n"
            "        np.ravel(unseen_result.day1_final_real) - np.ravel(unseen_result.actual_day1_real)\n"
            "    ))),\n"
            "    'live_rmse': float(np.sqrt(np.mean((\n"
            "        np.ravel(unseen_result.day1_final_real) - np.ravel(unseen_result.actual_day1_real)\n"
            "    ) ** 2))),\n"
            "}])\n"
            "display(live)"
        ),
        _md("## 7. Container ID lookup (optional)"),
        _code(
            "LOOKUP_QUERY = '1016'  # change substring to search known/unseen IDs\n\n"
            "lookup = pd.DataFrame({\n"
            "    'known_matches': pd.Series(search_containers(LOOKUP_QUERY, known_cids, limit=15)),\n"
            "    'unseen_matches': pd.Series(search_containers(LOOKUP_QUERY, unseen_cids, limit=15)),\n"
            "})\n"
            "display(lookup)"
        ),
        _md("## 8. Known vs Unseen Comparison"),
        _code(
            "def _split_violin(ax, df, col, title):\n"
            "    groups = [df.loc[df['split'] == s, col].dropna().values for s in ('known', 'unseen')]\n"
            "    parts = ax.violinplot(groups, showmeans=True, showmedians=True)\n"
            "    for idx, body in enumerate(parts['bodies']):\n"
            "        body.set_facecolor(list(PALETTE.values())[idx])\n"
            "        body.set_alpha(0.55)\n"
            "    ax.set_xticks([1, 2])\n"
            "    ax.set_xticklabels(['known', 'unseen'])\n"
            "    ax.set_title(title)\n"
            "    ax.set_xlabel('Split')\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "_split_violin(axes[0], eval_df, 'day1_mae', 'MAE Distribution: Known vs Unseen')\n"
            "_split_violin(axes[1], eval_df, 'day1_mape', 'MAPE Distribution: Known vs Unseen')\n"
            "plt.tight_layout()\n"
            "plt.show()"
        ),
        _code(
            "comparison = eval_df.groupby('split')[\n"
            "    ['day1_mae', 'day1_rmse', 'day1_mape', 'day1_pearson_r', 'day1_r2']\n"
            "].agg(['mean', 'std', 'median']).round(4)\n"
            "display(comparison)"
        ),
        _md("## 9. Model Summary"),
        _code(
            "summary_table = pd.DataFrame([\n"
            "    {'Property': 'Architecture', 'Value': 'Prophet + 3-layer GRU Hybrid'},\n"
            "    {'Property': 'Input shape', 'Value': f\"({prod_config['input_window']}, 1)\"},\n"
            "    {'Property': 'Output shape', 'Value': f\"({prod_config['forecast_horizon']},)\"},\n"
            "    {'Property': 'Parameters', 'Value': f'{n_params:,}'},\n"
            "    {'Property': 'Optimizer', 'Value': prod_config['optimizer']},\n"
            "    {'Property': 'Loss', 'Value': prod_config['loss']},\n"
            "    {'Property': 'Batch size', 'Value': train_meta['batch_size']},\n"
            "    {'Property': 'Best epoch', 'Value': train_meta['best_epoch']},\n"
            "    {'Property': 'Residual norm', 'Value': prod_config['residual_normalization']},\n"
            "])\n"
            "display(summary_table)"
        ),
        _md("## 10. Conclusions"),
        _code(
            "known_mae = eval_df.loc[eval_df['split'] == 'known', 'day1_mae'].mean()\n"
            "unseen_mae = eval_df.loc[eval_df['split'] == 'unseen', 'day1_mae'].mean()\n"
            "gap = unseen_mae - known_mae if pd.notna(unseen_mae) and pd.notna(known_mae) else np.nan\n\n"
            "print('Production readiness assessment')\n"
            "print('-' * 50)\n"
            "print(f'Mean Day-1 MAE (known):   {known_mae:.4f} %')\n"
            "print(f'Mean Day-1 MAE (unseen):  {unseen_mae:.4f} %')\n"
            "print(f'Generalization gap:       {gap:.4f} %')\n"
            "print()\n"
            "print('Deployment notes:')\n"
            "print('• Load model.keras + scalers.pkl + residual_stats.pkl for inference')\n"
            "print('• For new containers: fit Prophet + scaler on historical 80% only')\n"
            "print('• Apply global res_mean/res_std from residual_stats.pkl')\n"
            "print('• Primary forecast horizon: 96 steps (24 h at 15-min resolution)')"
        ),
        _md("## 11. Retrain Production Model (disabled by default)"),
        _code(
            "RUN_TRAINING = False  # Set True only to retrain from the notebook\n\n"
            "if RUN_TRAINING:\n"
            "    import subprocess\n"
            "    subprocess.run(\n"
            "        [sys.executable, str(REPO_ROOT / 'production' / 'scripts' / 'train_production_hybrid.py')],\n"
            "        check=True,\n"
            "        cwd=str(REPO_ROOT),\n"
            "    )\n"
            "    print('Retraining complete. Re-run notebook cells to refresh artifacts.')\n"
            "else:\n"
            "    print('Retraining disabled. Use production/scripts/train_production_hybrid.py to train.')"
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
