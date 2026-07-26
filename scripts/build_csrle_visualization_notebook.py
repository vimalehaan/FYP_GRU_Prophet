#!/usr/bin/env python3
"""Generate notebooks/csrle_visualization.ipynb (read-only CSRLE dashboard)."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat import v4 as nbf

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "csrle_visualization.ipynb"
CSRLE = "experiments/synthetic_residual_learnability_2026-07-25_164307"


def md(text: str) -> dict:
    return nbf.new_markdown_cell(text.strip() + "\n")


def code(text: str) -> dict:
    return nbf.new_code_cell(text.strip() + "\n")


cells: list[dict] = []

cells.append(md("""
# CSRLE — Controlled Synthetic Residual Learnability Experiment

**Read-only visualization notebook** (no training, no artifact overwrites).

Mirrors the structure of `hybrid_model.ipynb`: configuration → data → summary tables → matrices → exploratory plots → representative-container demo.

| Phase | Content |
|-------|---------|
| Pre-GRU | Synthetic data, α distribution, Prophet retention |
| Stage 1 | Condition A (real-data) Hybrid reproduction |
| Stage 2 | B0 / B1 / C0 / C1 independent GRU evaluation |

**Authoritative experiment:** `experiments/synthetic_residual_learnability_2026-07-25_164307/`

Execute cells sequentially. Most sections load precomputed CSV/JSON (fast). Optional Section 10 runs single-container inference.
"""))

cells.append(md("## 1 — Configuration"))
cells.append(code("""
import json
import os
import sys
from pathlib import Path

%matplotlib inline

REPO_ROOT = Path('..').resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

CSRLE_EXP = REPO_ROOT / 'experiments' / 'synthetic_residual_learnability_2026-07-25_164307'
STAGE1 = CSRLE_EXP / 'stage1_control_reproduction'
STAGE2 = CSRLE_EXP / 'stage2_synthetic_gru'
PREGRU = CSRLE_EXP / 'synthetic_validation'
DATA = CSRLE_EXP / 'data'

# Optional: save figures from notebook (new folder — does not overwrite Stage 1/2)
SAVE_FIGS = False
FIG_DIR = CSRLE_EXP / 'notebook_visualizations'
if SAVE_FIGS:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = {
    'A (control)': ('control', STAGE1),
    'B0': ('b0_lc', STAGE2 / 'b0_lc'),
    'B1': ('b1_snar', STAGE2 / 'b1_snar'),
    'C0': ('c0_iid', STAGE2 / 'c0_iid'),
    'C1': ('c1_iid', STAGE2 / 'c1_iid'),
}
STAGE2_SYN = ['b0_lc', 'b1_snar', 'c0_iid', 'c1_iid']

plt.rcParams.update({'figure.figsize': (10, 4), 'font.size': 11})


def plot_heatmap(matrix: pd.DataFrame, title: str, center: float = 0.0) -> None:
    \"\"\"Matplotlib heatmap (no seaborn dependency).\"\"\"
    data = matrix.values.astype(float)
    vmax = max(abs(data.min() - center), abs(data.max() - center), 1e-6)
    norm = TwoSlopeNorm(vmin=center - vmax, vcenter=center, vmax=center + vmax)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', norm=norm)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=30, ha='right')
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


print('CSRLE experiment:', CSRLE_EXP)
"""))

cells.append(md("## 2 — Load frozen summaries"))
cells.append(code("""
with (CSRLE_EXP / 'config' / 'synthetic_config_frozen.json').open() as fh:
    frozen_cfg = json.load(fh)
with (PREGRU / 'pre_gru_summary.json').open() as fh:
    pre_gru = json.load(fh)
with (STAGE2 / 'stage2_final_report.json').open() as fh:
    stage2_report = json.load(fh)

residual_summary = pd.read_csv(STAGE2 / 'comparisons' / 'all_conditions_residual_summary.csv')
paired_boot = json.loads((STAGE2 / 'comparisons' / 'paired_bootstrap_results.json').read_text())
alpha_b0 = pd.read_csv(PREGRU / 'alpha_map_b0_lc.csv')

print('Protocol:', frozen_cfg.get('protocol_version', 'v2'))
print('Pre-GRU generator gates:', pre_gru.get('generator_pass'))
print('Pre-GRU retention gates:', pre_gru.get('retention_pass'))
print('Stage 2 conditions:', stage2_report['verification']['conditions_trained_independently'])
display(residual_summary)
"""))

cells.append(md("## 3 — α distribution & cohort overview"))
cells.append(code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(alpha_b0['alpha'], bins=20, color='steelblue', edgecolor='white')
axes[0].axvline(1.0, color='k', ls='--', alpha=0.5, label='α=1')
axes[0].set_xlabel('Boundary-safe α (B0)'); axes[0].set_ylabel('Containers')
axes[0].set_title('α distribution (B0-LC)'); axes[0].legend()

alpha_zero = alpha_b0.loc[alpha_b0['alpha'] == 0, 'container_id'].tolist()
axes[1].bar(['α=1', '0<α<1', 'α=0'],
            [ (alpha_b0['alpha']==1).sum(), ((alpha_b0['alpha']>0)&(alpha_b0['alpha']<1)).sum(), (alpha_b0['alpha']==0).sum()],
            color=['#2ca02c', '#ff7f0e', '#d62728'])
axes[1].set_title('α cohort breakdown'); axes[1].set_ylabel('Count')
plt.tight_layout(); plt.show()
print('α=0 containers (8):', alpha_zero)
"""))

cells.append(md("## 4 — Synthetic data explorer (sample container)"))
cells.append(code("""
manifest = pd.read_parquet(DATA / 'ground_truth' / 'injection_manifest.parquet')
control_val = pd.read_parquet(DATA / 'control' / 'val_syn.parquet')
b0_val = pd.read_parquet(DATA / 'b0_lc' / 'val_syn.parquet')

# Representative: median α among active containers, or first α=1
sample_cid = alpha_b0.loc[alpha_b0['alpha']==1, 'container_id'].iloc[10]
print('Sample container:', sample_cid)

m = manifest[(manifest['container_id']==sample_cid) & (manifest['condition_id']=='b0_lc') & (manifest['split']=='val')].sort_values('time_stamp').head(96)
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(m['y_base'].values, label='y_base (control)', color='#1f77b4')
axes[0].plot(m['y_syn'].values, label='y_syn (B0)', color='#ff7f0e')
axes[0].set_ylabel('CPU %'); axes[0].set_title(f'Synthetic CPU — {sample_cid} (val Day-1)'); axes[0].legend()
axes[1].plot(m['N'].values, color='#2ca02c'); axes[1].set_ylabel('N'); axes[1].set_title('Effective injection N')
axes[2].plot(m['z'].values, color='#9467bd', alpha=0.7); axes[2].set_ylabel('z'); axes[2].set_xlabel('Step'); axes[2].set_title('Generator state z')
plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 5 — Cross-condition metric matrix (Stage 2 + Condition A)

Heatmaps of cohort-mean residual-learning and CPU metrics across A / B0 / B1 / C0 / C1.
"""))
cells.append(code("""
# Residual-learning matrix
res_cols = ['residual_pearson_r_mean', 'residual_r2_mean', 'residual_mae_mean', 'residual_std_ratio_mean']
res_mat = residual_summary.set_index('condition')[res_cols]
res_mat.index = ['A', 'B0', 'B1', 'C0', 'C1']

plot_heatmap(res_mat.T, 'Residual-learning metrics by condition (cohort mean)')
display(res_mat.round(4))
"""))

cells.append(md("## 6 — CPU forecasting: Prophet vs Hybrid by condition"))
cells.append(code("""
cpu_frames = []
labels = []
for label, (cid, path) in CONDITIONS.items():
    if label.startswith('A'):
        continue  # Stage 1 eval CSV
    cpu = pd.read_csv(path / 'evaluation' / 'cpu_metrics.csv')
    cpu['condition'] = label
    cpu_frames.append(cpu)
cpu_all = pd.concat(cpu_frames, ignore_index=True)

cpu_summary = cpu_all.groupby('condition').agg(
    prophet_mae=('prophet_day1_mae', 'mean'),
    hybrid_mae=('hybrid_day1_mae', 'mean'),
    delta_mae=('hybrid_minus_prophet_mae', 'mean'),
).round(4)
display(cpu_summary)

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(cpu_summary))
w = 0.35
ax.bar(x - w/2, cpu_summary['prophet_mae'], width=w, label='Prophet')
ax.bar(x + w/2, cpu_summary['hybrid_mae'], width=w, label='Hybrid')
ax.set_xticks(x, cpu_summary.index, rotation=15)
ax.set_ylabel('Day-1 MAE (CPU %)'); ax.set_title('Prophet vs Hybrid CPU MAE by synthetic condition')
ax.legend(); plt.tight_layout(); plt.show()
"""))

cells.append(md("## 7 — Stage 1 Condition A (real-data reference)"))
cells.append(code("""
a_eval = pd.read_csv(STAGE1 / 'evaluation' / 'csrle_new_model_evaluation_df.csv')
a_ext = pd.read_csv(STAGE1 / 'evaluation' / 'csrle_new_model_extended_metrics.csv')
a_summary = pd.read_csv(STAGE1 / 'evaluation' / 'csrle_new_model_summary.csv')

print('=== Stage 1 CPU (Condition A) ===')
display(a_summary)
print('=== Stage 1 residual learning (extended) ===')
display(a_ext[['residual_pearson_r','residual_r2','residual_std_ratio']].describe().round(4))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(a_eval['day1_mae'], bins=25, color='steelblue', edgecolor='white')
axes[0].set_xlabel('Day-1 MAE'); axes[0].set_title('Condition A — per-container MAE')
axes[1].hist(a_ext['residual_pearson_r'], bins=25, color='coral', edgecolor='white')
axes[1].set_xlabel('Residual Pearson r'); axes[1].set_title('Condition A — residual correlation')
plt.tight_layout(); plt.show()
"""))

cells.append(md("## 8 — Controlled comparisons: B0 vs C0, B1 vs C1"))
cells.append(code("""
b0_ext = pd.read_csv(STAGE2 / 'b0_lc' / 'evaluation' / 'extended_metrics.csv')
c0_ext = pd.read_csv(STAGE2 / 'c0_iid' / 'evaluation' / 'extended_metrics.csv')
b1_ext = pd.read_csv(STAGE2 / 'b1_snar' / 'evaluation' / 'extended_metrics.csv')
c1_ext = pd.read_csv(STAGE2 / 'c1_iid' / 'evaluation' / 'extended_metrics.csv')

def paired_plot(left, right, metric, title):
    m = left[['container_id', metric]].merge(right[['container_id', metric]], on='container_id', suffixes=('_b', '_c'))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(m[f'{metric}_c'], m[f'{metric}_b'], alpha=0.6, s=35)
    lim = max(m[f'{metric}_c'].abs().max(), m[f'{metric}_b'].abs().max(), 0.05)
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4)
    ax.set_xlabel(f'Control {metric}'); ax.set_ylabel(f'B {metric}'); ax.set_title(title)
    plt.show()

paired_plot(b0_ext, c0_ext, 'residual_pearson_r', 'B0 vs C0 — residual Pearson r')
paired_plot(b1_ext, c1_ext, 'residual_pearson_r', 'B1 vs C1 — residual Pearson r')

print('B0 vs C0 bootstrap (r):', paired_boot.get('b0_lc_vs_c0_iid_full_residual_pearson_r'))
print('B1 vs C1 bootstrap (r):', paired_boot.get('b1_snar_vs_c1_iid_full_residual_pearson_r'))
"""))

cells.append(md("## 9 — Baselines, horizons & training (B0)"))
cells.append(code("""
b0_base = pd.read_csv(STAGE2 / 'b0_lc' / 'evaluation' / 'baseline_comparison.csv')
display(b0_base.groupby('baseline')['residual_mae'].describe().round(4))

fig, ax = plt.subplots(figsize=(8, 4))
order = ['gru', 'ridge', 'zero', 'persistence']
data = [b0_base.loc[b0_base.baseline == b, 'residual_mae'].dropna().values for b in order]
ax.boxplot(data, tick_labels=order)
ax.set_title('B0 residual MAE by baseline'); ax.set_ylabel('MAE'); plt.show()

# Horizon cohort
hz = pd.read_csv(STAGE2 / 'b0_lc' / 'evaluation' / 'horizon_cohort_summary.csv')
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hz['horizon'], hz['mae'], marker='o')
ax.set_xlabel('Horizon h'); ax.set_ylabel('MAE'); ax.set_title('B0 residual MAE by horizon (cohort mean)'); plt.show()

# Training curves — all Stage 2 conditions
fig, ax = plt.subplots(figsize=(9, 4))
for cond in STAGE2_SYN:
    hist = pd.read_csv(STAGE2 / cond / 'training' / 'training_history.csv')
    ax.plot(hist['val_loss'], label=cond, ls='--')
ax.set_xlabel('Epoch'); ax.set_ylabel('Val MSE'); ax.set_title('Stage 2 GRU validation loss'); ax.legend(); plt.show()
"""))

cells.append(md("""
## 10 — Optional demo: single-container B0 Hybrid inference

Uses frozen Stage 2 B0 model. **Read-only** — does not retrain. Skip if slow.
Set `RUN_DEMO_INFERENCE = False` to disable.
"""))
cells.append(code("""
RUN_DEMO_INFERENCE = False  # set True for Prophet+GRU demo plots

if RUN_DEMO_INFERENCE:
    sys.path.insert(0, str(REPO_ROOT))
    from utils.hybrid_artifacts import load_hybrid_artifacts
    from utils.csrle.stage2_training import build_condition_scalers, load_condition_frames
    from utils.hybrid_inference import run_hybrid_inference

    train_df, val_df = load_condition_frames(CSRLE_EXP, 'b0_lc')
    model, res_mean, res_std, iw = load_hybrid_artifacts(
        STAGE2 / 'b0_lc' / 'models' / 'hybrid_gru.keras',
        STAGE2 / 'b0_lc' / 'models' / 'residual_stats.pkl',
    )
    scalers = build_condition_scalers(train_df)
    demo_cid = sample_cid
    result = run_hybrid_inference(demo_cid, train_df, val_df, model, scalers, res_mean, res_std)

    steps = np.arange(1, len(result.actual_day1_real)+1)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(steps, result.actual_day1_real, label='Actual', lw=1.5)
    axes[0].plot(steps, result.prophet_day1_real, label='Prophet', lw=1.5)
    axes[0].plot(steps, result.day1_final_real, label='Hybrid', lw=1.5)
    axes[0].set_ylabel('CPU %'); axes[0].set_title(f'B0 Hybrid demo — {demo_cid}'); axes[0].legend()
    actual_res = result.actual_day1_scaled - result.day1_prophet
    axes[1].plot(steps, actual_res, label='Actual residual')
    axes[1].plot(steps, result.day1_residual, label='GRU pred residual')
    axes[1].set_xlabel('Step'); axes[1].set_ylabel('Scaled residual'); axes[1].legend()
    plt.tight_layout(); plt.show()
else:
    print('Demo inference skipped. Set RUN_DEMO_INFERENCE=True to enable.')
"""))

cells.append(md("""
## 11 — Prophet retention (pre-GRU, frozen CSV)

Visualize ρ(N,I), variance separation VS, and retention metrics from pre-GRU validation.
"""))
cells.append(code("""
ret_b0 = pd.read_csv(PREGRU / 'prophet_retention_b0.csv')
print('B0 retention columns:', list(ret_b0.columns)[:12], '...')
display(ret_b0[['container_id', 'rho_n_i', 'vs', 'vr']].describe().round(4))

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, col, title in zip(axes, ['rho_n_i', 'vs', 'vr'], ['ρ(N,I)', 'VS=Var(ΔP)/Var(N)', 'VR=Var(I)/Var(N)']):
    ax.hist(ret_b0[col].dropna(), bins=20, color='teal', edgecolor='white')
    ax.set_title(title)
plt.suptitle('B0 Prophet retention (per container)'); plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 12 — Frozen Stage 2 plots (from experiment run)

Precomputed PNGs from `stage2_synthetic_gru/plots/` — same style as hybrid exploratory figures.
"""))
cells.append(code("""
from utils.notebook_display import show_png_gallery

show_png_gallery(STAGE2 / 'plots')
"""))

cells.append(md("""
## 13 — Summary dashboard

Quick reference tables loaded from frozen artifacts.
"""))
cells.append(code("""
print('='*60)
print('CSRLE SUMMARY DASHBOARD')
print('='*60)
print('Residual learning (cohort mean r):')
print(residual_summary[['condition','residual_pearson_r_mean','residual_std_ratio_mean']].to_string(index=False))
print()
print('Stage 2 CPU delta (Hybrid - Prophet MAE):')
print(cpu_summary['delta_mae'].to_string())
print()
print('Pre-GRU: all generator gates passed:', pre_gru['generator_pass'])
print('Stage 1 reproduction gate:', json.loads((STAGE1/'verification'/'reproduction_gate_result.json').read_text())['pass'])
print('='*60)
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
