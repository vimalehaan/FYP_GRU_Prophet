#!/usr/bin/env python3
"""Generate notebooks/lfhe_visualization.ipynb from LFHE experiment artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat import v4 as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "lfhe_visualization.ipynb"


def md(text: str) -> dict:
    return nbf.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> dict:
    return nbf.new_code_cell(text.strip() + "\n")


cells: list[dict] = []

cells.append(md("""
# LFHE — Loss Function Hypothesis Experiment

**Read-only visualization** of MSE vs DA-MSE on CSRLE B0-LC.

Execute cells sequentially. Change **`SELECTED_CONTAINER`** in Section 6 to visualize any container, then re-run Sections 7–9.
"""))

cells.append(md("## 1 — Experiment overview"))
cells.append(code("""
import json
import sys
from pathlib import Path

%matplotlib inline

REPO_ROOT = Path('..').resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Update this path if re-running against a different LFHE timestamp
LFHE_EXP = REPO_ROOT / 'experiments' / 'loss_function_hypothesis_2026-07-26_130252'
MSE_DIR = LFHE_EXP / 'b0_lc_mse'
DA_DIR = LFHE_EXP / 'b0_lc_da_mse'
SAVE_FIGS = False
FIG_DIR = LFHE_EXP / 'plots'
print('LFHE experiment:', LFHE_EXP)
"""))

cells.append(md("## 2 — Load frozen artifacts"))
cells.append(code("""
import pandas as pd

with (LFHE_EXP / 'config' / 'lfhe_config_frozen.json').open() as fh:
    cfg = json.load(fh)
with (LFHE_EXP / 'reports' / 'final_report.json').open() as fh:
    final_report = json.load(fh)

CSRLE_EXP = REPO_ROOT / cfg['csrle_experiment_dir']

mse_ext = pd.read_csv(MSE_DIR / 'evaluation' / 'extended_metrics.csv')
da_ext = pd.read_csv(DA_DIR / 'evaluation' / 'extended_metrics.csv')
mse_hist = pd.read_csv(MSE_DIR / 'training' / 'training_history.csv')
da_hist = pd.read_csv(DA_DIR / 'training' / 'training_history.csv')
hv_cohort = pd.read_csv(LFHE_EXP / 'diagnostics' / 'horizon_variance_recovery_cohort.csv')
hv_pc = pd.read_csv(LFHE_EXP / 'diagnostics' / 'horizon_variance_recovery_per_container.csv')
ec_cohort = pd.read_csv(LFHE_EXP / 'diagnostics' / 'energy_recovery_cohort.csv')
ec_pc = pd.read_csv(LFHE_EXP / 'diagnostics' / 'energy_recovery_per_container.csv')
summary = pd.read_csv(LFHE_EXP / 'tables' / 'lfhe_summary.csv')
ALL_CONTAINERS = sorted(mse_ext['container_id'].astype(str).unique())
display(summary)
"""))

cells.append(md("## 3 — Verify frozen configuration"))
cells.append(code("""
print('Protocol:', cfg['protocol_version'])
print('λ (frozen):', cfg['lambda_disp'])
print('Reproduction gate:', final_report['reproduction_gate']['pass'])
print('Hypothesis verdict:', final_report['hypothesis_verdict']['verdict'])
"""))

cells.append(md("## 4 — Load MSE reference (Stage 2 via reproduction gate)"))
cells.append(code("""
repro = final_report['reproduction_gate']
print('Stage 2 B0 reference (frozen weights eval):')
for k, v in repro['cohort_gate']['checks'].items():
    if 'stage2_reference' in v:
        print(f'  {k}: {v[\"stage2_reference\"]:.4f}')
"""))

cells.append(md("## 5 — Load DA-MSE experiment"))
cells.append(code("""
with (DA_DIR / 'training' / 'training_metadata.json').open() as fh:
    da_meta = json.load(fh)
print('DA-MSE best epoch:', da_meta['best_epoch'], 'best val loss:', da_meta['best_val_loss'])
"""))

cells.append(md("""
## 6 — Container selector

Set **`SELECTED_CONTAINER`** to any `container_id` from the browse table, or leave **`None`** for the auto representative (median |r| on DA-MSE).

**Examples:**
```python
SELECTED_CONTAINER = None          # auto
SELECTED_CONTAINER = 'c_10733'     # specific container
```

After changing, re-run this cell and Sections 7–9.
"""))
cells.append(code("""
from utils.csrle.stage2_plots import select_representative_container

# ── Change this line to pick a container ──
SELECTED_CONTAINER = None  # e.g. 'c_10733'

metric_cols = [
    'residual_pearson_r', 'residual_std_ratio',
    'residual_mae_scaled', 'residual_r2',
]
browse = da_ext[['container_id'] + metric_cols].merge(
    mse_ext[['container_id'] + metric_cols],
    on='container_id', suffixes=('_da', '_mse'),
)
browse['delta_r'] = browse['residual_pearson_r_da'] - browse['residual_pearson_r_mse']
browse['delta_std_ratio'] = browse['residual_std_ratio_da'] - browse['residual_std_ratio_mse']
browse = browse.sort_values('container_id')

print(f'Available containers: {len(browse)}')
print('Top 10 by DA-MSE std ratio gain:')
display(browse.nlargest(10, 'delta_std_ratio')[
    ['container_id', 'residual_pearson_r_mse', 'residual_pearson_r_da',
     'residual_std_ratio_mse', 'residual_std_ratio_da', 'delta_std_ratio']
])

if SELECTED_CONTAINER is None:
    SELECTED_CONTAINER = select_representative_container(da_ext)
elif str(SELECTED_CONTAINER) not in ALL_CONTAINERS:
    raise ValueError(
        f'Unknown container {SELECTED_CONTAINER!r}. '
        f'Choose from ALL_CONTAINERS ({len(ALL_CONTAINERS)} ids).'
    )

SELECTED_CONTAINER = str(SELECTED_CONTAINER)
print('\\n>>> Visualizing container:', SELECTED_CONTAINER)
display(browse.loc[browse['container_id'] == SELECTED_CONTAINER].T)
"""))

cells.append(md("## 7 — Inference setup (cached models & data)"))
cells.append(code("""
from utils.csrle.stage2_metrics import actual_predicted_residual_arrays, horizon_band_metrics
from utils.csrle.stage2_training import build_condition_scalers, load_condition_frames
from utils.lfhe.artifacts import load_lfhe_arm_artifacts
from utils.hybrid_inference import run_hybrid_inference

_train_df, _val_df = load_condition_frames(CSRLE_EXP, 'b0_lc')
_scalers = build_condition_scalers(_train_df)

_mse_model, _mse_rm, _mse_rs, _ = load_lfhe_arm_artifacts(MSE_DIR)
_da_model, _da_rm, _da_rs, _ = load_lfhe_arm_artifacts(DA_DIR)

_inference_cache: dict[str, dict] = {}


def get_container_residuals(container_id: str) -> dict:
    \"\"\"Run frozen MSE + DA-MSE inference for one container (cached).\"\"\"
    cid = str(container_id)
    if cid not in _inference_cache:
        mse_result = run_hybrid_inference(
            cid, _train_df, _val_df, _mse_model, _scalers, _mse_rm, _mse_rs,
        )
        da_result = run_hybrid_inference(
            cid, _train_df, _val_df, _da_model, _scalers, _da_rm, _da_rs,
        )
        actual_m, pred_m = actual_predicted_residual_arrays(mse_result)
        actual_d, pred_d = actual_predicted_residual_arrays(da_result)
        _inference_cache[cid] = {
            'actual': actual_m,
            'pred_mse': pred_m,
            'pred_da': pred_d,
            'mse_result': mse_result,
            'da_result': da_result,
        }
    return _inference_cache[cid]

print('Inference pipeline ready. Call get_container_residuals(SELECTED_CONTAINER) in next cells.')
"""))

cells.append(md("## 8 — Per-container residual trajectory"))
cells.append(code("""
import matplotlib.pyplot as plt
import numpy as np

res = get_container_residuals(SELECTED_CONTAINER)
actual, pred_mse, pred_da = res['actual'], res['pred_mse'], res['pred_da']
steps = np.arange(1, len(actual) + 1)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(steps, actual, label='Actual residual', lw=1.5, color='#2ca02c')
ax.plot(steps, pred_mse, label='MSE pred', lw=1.2, alpha=0.85)
ax.plot(steps, pred_da, label='DA-MSE pred', lw=1.2, alpha=0.85)
ax.set_xlabel('Day-1 step (1–96)')
ax.set_ylabel('Scaled residual')
ax.set_title(f'Residual trajectory — {SELECTED_CONTAINER}')
ax.legend()
plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(FIG_DIR / f'trajectory_{SELECTED_CONTAINER}.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

cells.append(md("## 9 — Per-container distributions & horizon bands"))
cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(actual, bins=30, alpha=0.5, label='Actual', density=True, color='#2ca02c')
axes[0].hist(pred_mse, bins=30, alpha=0.5, label='MSE', density=True)
axes[0].hist(pred_da, bins=30, alpha=0.5, label='DA-MSE', density=True)
axes[0].set_xlabel('Scaled residual')
axes[0].set_title(f'Distribution — {SELECTED_CONTAINER}')
axes[0].legend()

bands_mse = horizon_band_metrics(actual, pred_mse)
bands_da = horizon_band_metrics(actual, pred_da)
x = np.arange(len(bands_mse))
w = 0.35
axes[1].bar(x - w/2, bands_mse['std_ratio'], width=w, label='MSE')
axes[1].bar(x + w/2, bands_da['std_ratio'], width=w, label='DA-MSE')
axes[1].set_xticks(x, bands_mse['band'], rotation=30, ha='right')
axes[1].axhline(0.22, color='k', ls=':', alpha=0.5, label='Ridge ref')
axes[1].set_ylabel('Std ratio')
axes[1].set_title(f'Horizon-band std ratio — {SELECTED_CONTAINER}')
axes[1].legend(fontsize=8)

plt.tight_layout()
if SAVE_FIGS:
    fig.savefig(FIG_DIR / f'container_detail_{SELECTED_CONTAINER}.png', dpi=150, bbox_inches='tight')
plt.show()

# Per-container diagnostics from frozen CSVs
hv_row = hv_pc[(hv_pc['container_id'] == SELECTED_CONTAINER)]
ec_row = ec_pc[ec_pc['container_id'] == SELECTED_CONTAINER]
print('Horizon variance recovery (frozen diagnostic CSV):')
display(hv_row.pivot(index='band', columns='arm', values='std_ratio'))
print('Energy recovery ratio:')
display(ec_row[['arm', 'energy_recovery_ratio']])
"""))

cells.append(md("## 10 — Training curves"))
cells.append(code("""
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(mse_hist['val_loss'], label='MSE val', ls='--')
ax.plot(da_hist['val_loss'], label='DA-MSE val', ls='--')
ax.plot(mse_hist['loss'], label='MSE train', alpha=0.7)
ax.plot(da_hist['loss'], label='DA-MSE train', alpha=0.7)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('Training Loss')
ax.legend(); plt.tight_layout(); plt.show()
"""))

cells.append(md("## 11 — Cohort std ratio distribution"))
cells.append(code("""
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(mse_ext['residual_std_ratio'], bins=25, alpha=0.6, label='MSE')
ax.hist(da_ext['residual_std_ratio'], bins=25, alpha=0.6, label='DA-MSE')
if SELECTED_CONTAINER:
    sel_std_m = float(mse_ext.set_index('container_id').loc[SELECTED_CONTAINER, 'residual_std_ratio'])
    sel_std_d = float(da_ext.set_index('container_id').loc[SELECTED_CONTAINER, 'residual_std_ratio'])
    ax.axvline(sel_std_m, color='#1f77b4', ls='--', lw=2, label=f'{SELECTED_CONTAINER} MSE')
    ax.axvline(sel_std_d, color='#ff7f0e', ls='--', lw=2, label=f'{SELECTED_CONTAINER} DA-MSE')
ax.axvline(0.22, color='k', ls=':', label='Ridge ref')
ax.set_xlabel('Std ratio'); ax.set_ylabel('Count'); ax.set_title('Std Ratio Distribution')
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
"""))

cells.append(md("## 12 — Horizon-wise variance recovery (cohort)"))
cells.append(code("""
fig, ax = plt.subplots(figsize=(9, 4))
for arm, color in [('mse', '#1f77b4'), ('da_mse', '#ff7f0e')]:
    sub = hv_cohort[hv_cohort['arm'] == arm]
    ax.errorbar(sub['band'], sub['mean_std_ratio'],
                yerr=[sub['mean_std_ratio']-sub['ci_low'], sub['ci_high']-sub['mean_std_ratio']],
                fmt='o-', capsize=4, label=arm, color=color)
ax.set_ylabel('Std ratio'); ax.set_title('Horizon-wise Variance Recovery (cohort)')
ax.legend(); plt.xticks(rotation=20); plt.tight_layout(); plt.show()
display(hv_cohort)
"""))

cells.append(md("## 13 — Residual energy recovery (cohort)"))
cells.append(code("""
fig, ax = plt.subplots(figsize=(8, 4))
for arm, color in [('mse', '#1f77b4'), ('da_mse', '#ff7f0e')]:
    vals = ec_pc.loc[ec_pc['arm'] == arm, 'energy_recovery_ratio']
    ax.hist(vals, bins=25, alpha=0.6, label=arm, color=color)
ax.set_xlabel('ERR = Σ(ŷ²)/Σ(y²)'); ax.set_ylabel('Count'); ax.set_title('Energy Recovery')
ax.legend(); plt.tight_layout(); plt.show()
display(ec_cohort)
"""))

cells.append(md("## 14 — Pre-exported experiment plots"))
cells.append(code("""
from utils.notebook_display import show_png_gallery

show_png_gallery(FIG_DIR)
"""))

cells.append(md("## 15 — Metric comparison"))
cells.append(code("""
metrics = ['residual_pearson_r', 'residual_std_ratio', 'residual_mae_scaled']
rows = []
for m in metrics:
    rows.append({'metric': m, 'mse': mse_ext[m].mean(), 'da_mse': da_ext[m].mean(),
                 'delta_da_minus_mse': da_ext[m].mean() - mse_ext[m].mean()})
display(pd.DataFrame(rows))
"""))

cells.append(md("## 16 — Hypothesis evaluation"))
cells.append(code("""
v = final_report['hypothesis_verdict']
print('Verdict:', v['verdict'])
print('S1 dispersion recovery:', v['S1_dispersion'])
print('S2 correlation improvement:', v['S2_correlation'])
print('S3 MAE guardrail:', v['S3_mae_guardrail'])
print()
for q, a in final_report['explicit_answers'].items():
    print(f'{q}: {a}')
"""))

cells.append(md("## 17 — Final conclusions"))
cells.append(code("""
interp = final_report['diagnostic_interpretation']
print('MSE horizon band std ratios:', interp['mse_horizon_band_means'])
print('DA-MSE horizon band std ratios:', interp['da_horizon_band_means'])
print('Progressive MSE collapse across bands?', not interp['mse_approximately_flat_across_bands'])
print()
print(open(LFHE_EXP / 'reports' / 'final_report.md').read())
"""))

nb = nbf.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.11.0"}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)
print("Wrote", OUT)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the notebook after generation (embeds plots/tables in .ipynb)",
    )
    args = parser.parse_args()
    if args.execute:
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "execute_notebook.py"), str(OUT)],
            check=True,
        )
