import numpy as np
import matplotlib.pyplot as plt

AXES_6 = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis_names: list[str] = AXES_6,
    title_prefix: str = "Test",
    split_forces_moments: bool = True,
):
    """
    Scatter Vrai vs Prédit pour chaque axe.
    - Ligne y=x pour référence.
    - Option: séparer Forces (0..2) et Moments (3..5) en 2 figures (mieux car unités différentes).
    """
    assert y_true.shape == y_pred.shape
    assert y_true.shape[1] == len(axis_names)

    def _one_figure(indices: list[int], fig_title: str):
        n = len(indices)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
        for j, k in enumerate(indices):
            ax = axes[0, j]
            yt = y_true[:, k]
            yp = y_pred[:, k]

            ax.scatter(yt, yp, alpha=0.7)

            # diagonale y=x sur l'étendue des données
            lo = float(min(np.min(yt), np.min(yp)))
            hi = float(max(np.max(yt), np.max(yp)))
            ax.plot([lo, hi], [lo, hi], linewidth=1)

            ax.set_title(axis_names[k])
            ax.set_xlabel("Vrai")
            ax.set_ylabel("Prédit")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        fig.suptitle(fig_title)
        fig.tight_layout()
        return fig

    if split_forces_moments:
        fig1 = _one_figure([0, 1, 2], f"{title_prefix} — Forces (Vrai vs Prédit)")
        fig2 = _one_figure([3, 4, 5], f"{title_prefix} — Moments (Vrai vs Prédit)")
        return fig1, fig2
    else:
        fig = _one_figure(list(range(6)), f"{title_prefix} — Vrai vs Prédit")
        return (fig,)

def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis_names: list[str] = AXES_6,
    title_prefix: str = "Test",
    split_forces_moments: bool = True,
):
    """
    Résidus (Prédit - Vrai) vs Vrai, par axe.
    """
    assert y_true.shape == y_pred.shape
    err = y_pred - y_true

    def _one_figure(indices: list[int], fig_title: str):
        n = len(indices)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4), squeeze=False)
        for j, k in enumerate(indices):
            ax = axes[0, j]
            ax.scatter(y_true[:, k], err[:, k], alpha=0.7)
            ax.axhline(0.0, linewidth=1)
            ax.set_title(axis_names[k])
            ax.set_xlabel("Vrai")
            ax.set_ylabel("Résidu (Prédit - Vrai)")
        fig.suptitle(fig_title)
        fig.tight_layout()
        return fig

    if split_forces_moments:
        fig1 = _one_figure([0, 1, 2], f"{title_prefix} — Résidus (Forces)")
        fig2 = _one_figure([3, 4, 5], f"{title_prefix} — Résidus (Moments)")
        return fig1, fig2
    else:
        fig = _one_figure(list(range(6)), f"{title_prefix} — Résidus")
        return (fig,)
