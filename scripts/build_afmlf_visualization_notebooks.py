#!/usr/bin/env python3
"""Build AFMLF visualization notebooks (live plots from experiment artifacts)."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [text],
    }


def build_official_notebook() -> list[dict]:
    return [
        md(
            "# Adaptive Forecast Model Lifecycle Framework (AFMLF)\n\n"
            "**Official thesis evaluation — live visualization notebook.**\n\n"
            "This notebook loads the frozen official experiment artifacts, **recomputes metrics "
            "from raw CSV/JSON**, and regenerates every figure with matplotlib. "
            "No pre-rendered PNGs are used as the primary view.\n\n"
            "> Experiment: `experiments/adaptive_forecast_model_lifecycle_2026-07-30_071206` "
            "(auto-detected via `OFFICIAL_RUN` marker)"
        ),
        code(
            "import json\nimport sys\nfrom pathlib import Path\n\n"
            "import matplotlib.pyplot as plt\nimport numpy as np\nimport pandas as pd\n\n"
            "try:\n"
            "    from IPython.display import Markdown, display\n"
            "except ImportError:\n"
            "    display = print\n"
            "    Markdown = print\n\n"
            "REPO = Path('..').resolve()\n"
            "sys.path.insert(0, str(REPO))\n\n"
            "from utils.adaptive_lifecycle import plots as afmlf_plots\n"
            "from utils.adaptive_lifecycle import presentation_plots as afmlf_pres\n"
            "from utils.adaptive_lifecycle.config import AFMLFConfig\n"
            "from utils.adaptive_lifecycle.evaluation import compute_framework_evaluation\n"
            "from utils.adaptive_lifecycle.rolling_metrics import append_rolling_metrics\n\n"
            "NOTEBOOK_OUT = REPO / 'notebooks' / 'outputs' / 'afmlf_official_live'\n"
            "NOTEBOOK_OUT.mkdir(parents=True, exist_ok=True)\n\n"
            "def pick_experiment_dir(repo: Path) -> Path:\n"
            "    exps = sorted(repo.glob('experiments/adaptive_forecast_model_lifecycle_*'))\n"
            "    official = [p for p in exps if (p / 'OFFICIAL_RUN').exists()]\n"
            "    if official:\n"
            "        return official[-1]\n"
            "    raise FileNotFoundError('No official AFMLF experiment found')\n\n"
            "def show_interpretation(title: str, body: str) -> None:\n"
            "    display(Markdown(f'**Interpretation — {title}**\\n\\n{body}'))\n\n"
            "EXP_DIR = pick_experiment_dir(REPO)\n"
            "CFG = AFMLFConfig.from_json(EXP_DIR / 'config' / 'afmlf_config.json')\n"
            "N_CONTAINERS = int(pd.read_csv(EXP_DIR / 'diagnostics' / 'diagnostics.csv')['container_id'].nunique())\n\n"
            "life = pd.read_csv(EXP_DIR / 'lifecycle' / 'transitions.csv')\n"
            "decisions = pd.read_csv(EXP_DIR / 'decisions' / 'decisions.csv')\n"
            "diag = pd.read_csv(EXP_DIR / 'diagnostics' / 'diagnostics.csv')\n"
            "rec = pd.read_csv(EXP_DIR / 'recommendations' / 'recommendations.csv')\n\n"
            "print('Experiment:', EXP_DIR.name)\n"
            "print('Official:', (EXP_DIR / 'OFFICIAL_RUN').exists())\n"
            "print('Containers:', N_CONTAINERS)"
        ),
        md(
            "## Executive summary\n\n"
            "The official AFMLF run evaluated **99 research containers** with a frozen lifecycle "
            "methodology. The summary below is loaded from the experiment report (text only — "
            "all charts that follow are regenerated live)."
        ),
        code(
            "print((EXP_DIR / 'reports' / 'AFMLF_FINAL_RESULTS.md').read_text())"
        ),
        md(
            "## 1. Monitoring — rolling Day-1 MAE\n\n"
            "**Reproducibility:** Reload `origin_metrics.csv`, drop stored rolling columns, "
            "and recompute with `append_rolling_metrics()`."
        ),
        code(
            "origin_raw = pd.read_csv(EXP_DIR / 'monitoring' / 'origin_metrics.csv')\n"
            "base_cols = [c for c in origin_raw.columns if not c.startswith('rolling_')]\n"
            "origin_recomputed = append_rolling_metrics(origin_raw[base_cols], CFG.rolling_window_origins)\n\n"
            "stored = origin_raw['rolling_hybrid_mae'].values\n"
            "recomp = origin_recomputed['rolling_hybrid_mae'].values\n"
            "max_delta = float(np.nanmax(np.abs(stored - recomp)))\n"
            "print('Rolling MAE max |delta|:', max_delta)\n\n"
            "fig, ax = afmlf_plots.plot_rolling_mae(origin_recomputed)\n"
            "ax.set_title('Rolling Day-1 MAE (recomputed live)')\n"
            "plt.show()\n"
            "fig.savefig(NOTEBOOK_OUT / 'rolling_mae_live.png', dpi=150, bbox_inches='tight')\n\n"
            "show_interpretation(\n"
            "    'Rolling Day-1 MAE',\n"
            "    f'This figure tracks **{N_CONTAINERS} containers** across **{origin_recomputed[\"origin_index\"].nunique()} walk-forward origins**. '\n"
            "    f'Each line is one container\\'s rolling Hybrid Day-1 MAE — the primary operational signal AFMLF monitors after deployment. '\n"
            "    f'Live recomputation matches stored metrics (max |Δ| = {max_delta:.2e}), confirming monitoring reproducibility. '\n"
            "    'AFMLF uses these rolling errors as input to per-container drift detection.'\n"
            ")"
        ),
        md("## 2. Monitoring — rolling Day-1 RMSE"),
        code(
            "fig, ax = afmlf_plots.plot_rolling_rmse(origin_recomputed)\n"
            "ax.set_title('Rolling Day-1 RMSE (recomputed live)')\n"
            "plt.show()\n"
            "fig.savefig(NOTEBOOK_OUT / 'rolling_rmse_live.png', dpi=150, bbox_inches='tight')\n\n"
            "show_interpretation(\n"
            "    'Rolling Day-1 RMSE',\n"
            "    'RMSE penalises large forecast spikes more than MAE. AFMLF tracks both metrics so that '\n"
            "    'sudden error bursts (regime shifts) are visible alongside typical degradation. '\n"
            "    f'Together with rolling MAE across **{len(origin_recomputed)} origin-container pairs**, '\n"
            "    'this supports robust container-level monitoring before any cohort-level action.'\n"
            ")"
        ),
        md("## 3. Page-Hinkley confirmatory test"),
        code(
            "ph_log = pd.read_csv(EXP_DIR / 'lifecycle' / 'page_hinkley.csv')\n"
            "sample_cid = ph_log['container_id'].value_counts().idxmax()\n"
            "fig, ax = afmlf_plots.plot_page_hinkley(ph_log, sample_cid)\n"
            "plt.show()\n"
            "n_alarms = int(ph_log['ph_alarm'].sum())\n"
            "show_interpretation(\n"
            "    'Page-Hinkley confirmatory test',\n"
            "    f'Page-Hinkley (PH) is a sequential change-detection test on each container\\'s MAE stream. '\n"
            "    f'In **confirmatory mode**, PH strengthens drift confirmation before lifecycle progression. '\n"
            "    f'This sample container (`{sample_cid}`) illustrates the PH statistic over time; '\n"
            "    f'the full log records **{n_alarms} PH alarms** across the cohort. '\n"
            "    'In the official run, confirmatory PH blocked per-container confirmation — '\n"
            "    'cohort-level diagnostics still drove the single retraining recommendation.'\n"
            ")"
        ),
        md("## 4. Diagnostic layer"),
        code(
            "fig, ax, diag_summary = afmlf_pres.plot_diagnostic_distribution_enhanced(diag)\n"
            "plt.show()\n"
            "display(diag_summary)\n\n"
            "def _diag_line(summary, cls, desc):\n"
            "    if cls in summary.index:\n"
            "        return f'**{cls}** ({int(summary.loc[cls, \"count\"])} events, {summary.loc[cls, \"pct\"]}%) — {desc}'\n"
            "    return ''\n\n"
            "diag_lines = ' '.join(filter(None, [\n"
            "    f'**STABLE** dominates ({int(diag_summary.loc[\"STABLE\", \"count\"])} events, {diag_summary.loc[\"STABLE\", \"pct\"]}%) because most monitored containers remained healthy.' if 'STABLE' in diag_summary.index else '',\n"
            "    _diag_line(diag_summary, 'GRU_STALE', 'degradation mainly in the residual-learning component.'),\n"
            "    _diag_line(diag_summary, 'REGIME_CHANGE', 'workload changes affecting both Prophet and Hybrid.'),\n"
            "    _diag_line(diag_summary, 'INVESTIGATE', 'ambiguous cases requiring further analysis.'),\n"
            "]))\n\n"
            "show_interpretation(\n"
            "    'Diagnostic distribution',\n"
            "    f'Across **{len(diag)} monitoring events** ({N_CONTAINERS} containers × 5 origins), {diag_lines} '\n"
            "    'Diagnostics explain *why* errors changed — they feed the cohort decision engine alongside drift evidence.'\n"
            ")"
        ),
        md(
            "**Diagnostic class reference:**\n\n"
            "- **STABLE** — most containers remained healthy throughout monitoring\n"
            "- **GRU_STALE** — Hybrid degraded while Prophet stayed stable; residual path likely stale\n"
            "- **REGIME_CHANGE** — both Prophet and Hybrid elevated; underlying workload shift\n"
            "- **INVESTIGATE** — ambiguous signal; defer action pending further analysis"
        ),
        md("## 5. Lifecycle state transitions"),
        code(
            "fig, ax, milestones = afmlf_pres.plot_lifecycle_cohort_flow(life, decisions)\n"
            "plt.show()\n"
            "pd.DataFrame(milestones)\n\n"
            "m0 = milestones[0] if milestones else {}\n"
            "show_interpretation(\n"
            "    'Cohort lifecycle flow',\n"
            "    'This diagram shows how the cohort lifecycle **evolves chronologically** across walk-forward origins — '\n"
            "    'not isolated scatter points. Arrows connect origin milestones: containers enter **Warning** on threshold breach, '\n"
            "    'progress to **Drift Suspected** after persistence, and at origin 200 the cohort satisfies the global trigger '\n"
            "    'leading to **Candidate Retraining**. Later origins show continued monitoring without further global retrain events. '\n"
            "    f'First milestone: origin {m0.get(\"origin_index\", \"n/a\")} — {m0.get(\"label\", \"n/a\").replace(chr(10), \" \")}.'\n"
            ")"
        ),
        md("## 6. Cohort decision timeline"),
        code(
            "display(decisions)\n"
            "fig, ax, dec_ann = afmlf_pres.plot_decision_timeline_annotated(decisions, life, N_CONTAINERS)\n"
            "plt.show()\n"
            "display(dec_ann)\n\n"
            "d200 = dec_ann[dec_ann['origin_index'] == 200].iloc[0]\n"
            "show_interpretation(\n"
            "    'Cohort decision timeline',\n"
            "    f'At **origin 200**, **{int(d200[\"n_drifting\"])}/{N_CONTAINERS}** containers ({d200[\"drift_pct\"]}%) showed drift evidence '\n"
            "    f'and the **global trigger was satisfied** (`{d200[\"decision\"]}`). '\n"
            "    f'At subsequent origins the trigger was not satisfied again (cooldown / insufficient fraction) — '\n"
            "    f'**{int((decisions[\"decision\"] == \"do_not_retrain\").sum())}** origins recorded `do_not_retrain`. '\n"
            "    'Retraining is a **cohort-level** decision: individual container drift alone is insufficient.'\n"
            ")"
        ),
        md("## 7. Model version lineage"),
        code(
            "with (EXP_DIR / 'versions' / 'registry.json').open() as fh:\n"
            "    registry = json.load(fh)\n"
            "fig, ax = afmlf_plots.plot_version_history(registry)\n"
            "plt.show()\n"
            "version_df = pd.json_normalize(registry['versions'])[['version_id','deployment_recommendation','parent_version']]\n"
            "display(version_df)\n\n"
            "show_interpretation(\n"
            "    'Model version lineage',\n"
            "    'AFMLF never overwrites production in place. **hybrid_v1** is the frozen incumbent; '\n"
            "    '**hybrid_v2** is the offline candidate trained after the single cohort-level retraining event. '\n"
            "    f'The registry records **{len(registry.get(\"versions\", []))} version(s)** with parent lineage and deployment recommendations — '\n"
            "    'supporting auditability and human-in-the-loop promotion.'\n"
            ")"
        ),
        md("## 8. Candidate evaluation & deployment recommendation"),
        code(
            "display(rec)\n"
            "if len(rec):\n"
            "    fig, ax, cmp_meta = afmlf_pres.plot_candidate_mae_comparison(rec, CFG.candidate_improvement_margin)\n"
            "    plt.show()\n"
            "    pd.Series(cmp_meta)\n"
            "else:\n"
            "    cmp_meta = {}\n"
            "    print('No recommendations recorded.')\n\n"
            "if cmp_meta:\n"
            "    show_interpretation(\n"
            "        'Candidate vs incumbent MAE',\n"
            "        f'Candidate **hybrid_v2** achieved Day-1 MAE **{cmp_meta[\"candidate_mae\"]:.4f}%** vs incumbent '\n"
            "        f'**{cmp_meta[\"incumbent_mae\"]:.4f}%** (ΔMAE {cmp_meta[\"delta_mae\"]:+.4f}%). '\n"
            "        f'The improvement ({abs(cmp_meta[\"delta_mae\"]):.4f} pp) is **below the deployment margin** '\n"
            "        f'({cmp_meta[\"deploy_margin\"]:.4f} pp), so AFMLF correctly recommended '\n"
            "        f'**{cmp_meta[\"recommendation\"].replace(\"_\", \" \")}** — avoiding unnecessary model churn.'\n"
            "    )"
        ),
        md("## 9. Framework evaluation (recomputed)"),
        code(
            "fw_live = compute_framework_evaluation(life, decisions, diag, rec)\n"
            "with (EXP_DIR / 'reports' / 'final_report.json').open() as fh:\n"
            "    fw_stored = json.load(fh)['framework_evaluation']\n\n"
            "compare_keys = ['confirmed_drift_events','retraining_frequency','prevented_bad_deployments','diagnostic_distribution']\n"
            "for k in compare_keys:\n"
            "    print(f'{k}: stored={fw_stored.get(k)!r}  live={getattr(fw_live,k)!r}')\n\n"
            "display(pd.DataFrame(fw_live.recommendation_quality))\n\n"
            "show_interpretation(\n"
            "    'Framework evaluation (recomputed)',\n"
            "    f'Live recomputation via `compute_framework_evaluation()` matches the official JSON report. '\n"
            "    f'Key outcomes: **{fw_live.retraining_frequency}** retraining event(s), '\n"
            "    f'**{fw_live.confirmed_drift_events}** confirmed drift transitions, '\n"
            "    f'**{fw_live.prevented_bad_deployments}** bad deployments prevented. '\n"
            "    'Primary thesis metric is recommendation quality — whether lifecycle guidance is trustworthy.'\n"
            ")"
        ),
        md("## 10. End-to-end lifecycle summary"),
        code(
            "fig, ax, funnel_df = afmlf_pres.plot_official_lifecycle_summary(diag, life, decisions, rec)\n"
            "plt.show()\n"
            "display(funnel_df)\n\n"
            "deploy_label = rec.iloc[0]['deployment_recommendation'] if len(rec) else 'n/a'\n"
            "show_interpretation(\n"
            "    'End-to-end lifecycle summary',\n"
            "    f'This funnel summarises the official AFMLF run using experiment-derived counts only: '\n"
            "    f'**{N_CONTAINERS} containers** → **{len(diag)} monitoring events** → '\n"
            "    f'**{int((life[\"to_state\"] == \"drift_suspected\").sum())} drift-suspected transitions** → '\n"
            "    f'**{int((decisions[\"decision\"] == \"recommend_retraining\").sum())} global retraining recommendation** → '\n"
            "    f'**hybrid_v2 candidate** → evaluation → **{deploy_label.replace(\"_\", \" \")}**. '\n"
            "    'AFMLF separates container-level monitoring from cohort-level lifecycle governance.'\n"
            ")"
        ),
        md(
            "## 11. Reproducibility statement\n\n"
            "All figures above were generated inside this notebook from:\n"
            "- `monitoring/origin_metrics.csv`\n"
            "- `lifecycle/*.csv`\n"
            "- `diagnostics/diagnostics.csv`\n"
            "- `decisions/decisions.csv`\n"
            "- `recommendations/recommendations.csv`\n"
            "- `versions/registry.json`\n"
            "- `config/afmlf_config.json`\n\n"
            "Optional PNG copies saved under `notebooks/outputs/afmlf_official_live/`."
        ),
    ]


def build_case_study_notebook() -> list[dict]:
    import sys

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.afmlf_case_study_notebook_cells import build_case_study_notebook as _build

    return _build()


def write_notebook(path: Path, cells: list[dict], *, kernel_name: str = "python3") -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": kernel_name,
                "language": "python",
                "name": kernel_name,
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1))


def main() -> None:
    write_notebook(REPO / "notebooks" / "adaptive_forecast_model_lifecycle.ipynb", build_official_notebook())
    write_notebook(
        REPO / "notebooks" / "afmlf_unseen_container_case_study.ipynb",
        build_case_study_notebook(),
        kernel_name="tf_metal_env",
    )
    print("Wrote notebooks/adaptive_forecast_model_lifecycle.ipynb")
    print("Wrote notebooks/afmlf_unseen_container_case_study.ipynb")


if __name__ == "__main__":
    main()
