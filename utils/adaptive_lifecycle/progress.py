"""Terminal progress reporting for AFMLF experiments (execution only)."""

from __future__ import annotations

import sys
import time
from typing import Any


def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds > 86400 * 7:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class AFMLFProgress:
    """Structured stdout progress for long AFMLF runs."""

    def __init__(self, official: bool = False) -> None:
        self.official = official
        self._monitor_t0 = time.perf_counter()
        self._phase = "init"

    def _print(self, msg: str) -> None:
        print(msg, flush=True)

    def banner(self, n_containers: int, exp_dir: str) -> None:
        tag = "OFFICIAL THESIS EVALUATION" if self.official else "AFMLF RUN"
        self._print("=" * 72)
        self._print(f"  {tag}")
        self._print(f"  Experiment: {exp_dir}")
        self._print(f"  Cohort size: {n_containers} containers")
        self._print("=" * 72)

    def phase(self, name: str) -> None:
        self._phase = name
        self._print(f"\n>>> Phase: {name}")

    def monitoring(
        self,
        container_idx: int,
        n_containers: int,
        container_id: str,
        origin_index: int,
        done: int,
        total: int,
    ) -> None:
        elapsed = time.perf_counter() - self._monitor_t0
        rate = done / elapsed if elapsed > 0 and done else 0
        remaining = (total - done) / rate if rate > 0 else 0
        pct = 100.0 * done / total if total else 0
        self._print(
            f"[monitoring {pct:5.1f}%] container {container_idx}/{n_containers} "
            f"{container_id} origin={origin_index} | {done}/{total} origins | "
            f"elapsed {_fmt_eta(elapsed)} | ETA {_fmt_eta(remaining)}"
        )

    def monitoring_failure(self, container_id: str, origin_index: int, error: str) -> None:
        self._print(
            f"  !! SKIP {container_id} origin={origin_index}: {error[:120]}"
        )

    def lifecycle_origin(
        self,
        origin_index: int,
        origin_idx: int,
        n_origins: int,
        n_breaches: int,
        n_ph_alarms: int,
        top_diag: str,
    ) -> None:
        self._print(
            f"[lifecycle {origin_idx}/{n_origins}] origin={origin_index} | "
            f"drift_breaches={n_breaches} ph_alarms={n_ph_alarms} | top_diag={top_diag}"
        )

    def decision(self, origin_index: int, decision: str, reason: str) -> None:
        self._print(f"[decision] origin={origin_index} → {decision} ({reason})")

    def retraining_start(self, version_id: str, origin_index: int, n_rows: int) -> None:
        self._print(
            f"[retrain] START {version_id} at origin={origin_index} | train_rows={n_rows}"
        )

    def retraining_done(self, version_id: str, duration_s: float, n_sequences: int) -> None:
        self._print(
            f"[retrain] DONE  {version_id} | {duration_s:.1f}s | GRU sequences={n_sequences}"
        )

    def evaluation(
        self,
        version_id: str,
        n_eval: int,
        incumbent_mae: float,
        candidate_mae: float,
        recommendation: str,
    ) -> None:
        delta = candidate_mae - incumbent_mae
        self._print(
            f"[eval] {version_id} | n_origins={n_eval} | "
            f"incumbent={incumbent_mae:.4f} candidate={candidate_mae:.4f} "
            f"delta={delta:+.4f} → {recommendation}"
        )

    def failures_summary(self, failures: list[dict[str, Any]]) -> None:
        self._print(f"\n>>> Failures: {len(failures)} container-origin pairs skipped")
        for item in failures[:20]:
            self._print(
                f"  - {item['container_id']} origin={item['origin_index']}: {item['error'][:100]}"
            )
        if len(failures) > 20:
            self._print(f"  ... and {len(failures) - 20} more (see tables/monitoring_failures.csv)")

    def complete(self, exp_dir: str, elapsed_s: float) -> None:
        self._print("\n" + "=" * 72)
        self._print(f"  AFMLF COMPLETE in {_fmt_eta(elapsed_s)}")
        self._print(f"  Outputs: {exp_dir}")
        self._print("=" * 72)
