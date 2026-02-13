from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from software.calibration.optimization_2_planes.types import ProtocolEval


# ----------------------------
# Exports
# ----------------------------


def export_protocols_csv(
    scored: Sequence[tuple[float, ProtocolEval]], path: str
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "score",
        "name",
        "n_total",
        "n_train",
        "n_test",
        "rmse_total",
        "r2_total",
        "A_std_mean",
        "A_std_max",
        "A_cv_mean",
        "A_cv_max",
        "influence_top_score",
        "influence_top_idx",
        "influence_top_file",
        "rmse_per_axis",
        "r2_per_axis",
        "notes",
    ]

    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k, (s, ev) in enumerate(scored, start=1):
            w.writerow(
                {
                    "rank": k,
                    "score": float(s),
                    "name": ev.name,
                    "n_total": ev.n_total,
                    "n_train": ev.n_train,
                    "n_test": ev.n_test,
                    "rmse_total": ev.rmse_total,
                    "r2_total": ev.r2_total,
                    "A_std_mean": ev.A_std_mean,
                    "A_std_max": ev.A_std_max,
                    "A_cv_mean": ev.A_cv_mean,
                    "A_cv_max": ev.A_cv_max,
                    "influence_top_score": ev.influence_top_score,
                    "influence_top_idx": ev.influence_top_idx,
                    "influence_top_file": ev.influence_top_file,
                    "rmse_per_axis": json.dumps(ev.rmse_per_axis),
                    "r2_per_axis": json.dumps(ev.r2_per_axis),
                    "notes": ev.notes,
                }
            )


def export_protocols_json(
    scored: Sequence[tuple[float, ProtocolEval]], path: str
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = []
    for k, (s, ev) in enumerate(scored, start=1):
        payload.append({"rank": k, "score": float(s), **asdict(ev)})

    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)