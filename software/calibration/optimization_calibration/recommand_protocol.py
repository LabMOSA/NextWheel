from pathlib import Path
from typing import Sequence

import numpy as np

from software.calibration.optimization_calibration import FitConfig, MonteCarloConfig
from software.calibration.optimization_calibration.types import ProtocolEval, ProtocolSpec


# # ----------------------------
# # One-call helper
# # ----------------------------



# def recommend_protocol(
#     trials: Sequence[dict],
#     acc_bias: np.ndarray,
#     base: np.ndarray,
#     fit_cfg: FitConfig,
#     mc_cfg: MonteCarloConfig,
#     out_dir: str | Path | None = None,
#     seed: int = 0,
#     min_trials: int = 20,
#     top_n: int = 5,
#     alpha_rmse: float = 1.0,
#     beta_cv: float = 1.0,
#     gamma_dom: float = 0.0,
# ) -> tuple[list[tuple[float, ProtocolEval]], list[ProtocolSpec]]:
#     """
#     Generate realistic candidate protocols, evaluate them, and return the best ones.
#     Optionally export CSV/JSON.
#
#     Defaults are set to avoid "ubuesque" scenarios:
#       - min_trials=20
#       - invalid protocols are discarded automatically
#     """
#     cands = generate_protocol_candidates(trials, min_trials=min_trials)
#
#     best = search_best_protocols(
#         trials=trials,
#         acc_bias=acc_bias,
#         base=base,
#         fit_cfg=fit_cfg,
#         mc_cfg=mc_cfg,
#         top_n=top_n,
#         alpha_rmse=alpha_rmse,
#         beta_cv=beta_cv,
#         gamma_dom=gamma_dom,
#         candidates=cands,
#         seed=seed,
#     )
#
#     if out_dir is not None:
#         out_dir = Path(out_dir)
#         export_protocols_csv(best, str(out_dir / "protocol_ranking.csv"))
#         export_protocols_json(best, str(out_dir / "protocol_ranking.json"))
#
#     return best, cands