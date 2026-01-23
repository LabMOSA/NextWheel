from __future__ import annotations

from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

AXES_6: tuple[str, ...] = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


def _make_axis_figures(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis_names: Sequence[str],
    title_prefix: str,
    split_forces_moments: bool,
    panel_title: str,
    plot_one_axis,
) -> tuple[plt.Figure, ...]:
    """
    Generic helper to produce 1-row figures for selected axes.
    Parameters
    ----------
    y_true : np.ndarray
        True values, shape (n_samples, n_axes).
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_axes).
    axis_names : Sequence[str]
        Names of the axes.
    title_prefix : str
        Prefix for the figure titles.
    split_forces_moments : bool
        Whether to create separate figures for forces and moments.
    panel_title : str
        Title for each panel.
    plot_one_axis : Callable[[plt.Axes, np.ndarray, np.ndarray, str],
        None]
        Function to plot one axis on a given Axes.
    Returns
    -------
    tuple[plt.Figure, ...]
        Tuple of matplotlib Figures.
    """

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.ndim != 2:
        raise ValueError(f"Expected 2D arrays (n_samples, n_axes), got y_true.ndim={y_true.ndim}")
    if y_true.shape[1] != len(axis_names):
        raise ValueError(f"Expected {len(axis_names)} axes, got {y_true.shape[1]}")

    if split_forces_moments:
        groups: list[tuple[list[int], str]] = [
            ([0, 1, 2], f"{title_prefix} — Forces ({panel_title})"),
            ([3, 4, 5], f"{title_prefix} — Moments ({panel_title})"),
        ]
    else:
        groups = [(list(range(len(axis_names))), f"{title_prefix} — {panel_title}")]

    figs: list[plt.Figure] = []
    for indices, fig_title in groups:
        n = len(indices)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)

        for j, k in enumerate(indices):
            ax = axes[0, j]
            yt = y_true[:, k]
            yp = y_pred[:, k]
            plot_one_axis(ax, yt, yp, axis_names[k])

        fig.suptitle(fig_title)
        fig.tight_layout()
        figs.append(fig)

    return tuple(figs)


def plot_pred_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis_names: Sequence[str] = AXES_6,
    title_prefix: str = "Test",
    split_forces_moments: bool = True,
) -> tuple[plt.Figure, ...]:
    """
    Scatter plot of predicted vs true values, per axis, with y=x reference line.

    Parameters
    ----------
    y_true : np.ndarray
        True values, shape (n_samples, n_axes).
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_axes).
    axis_names : Sequence[str], optional
        Names of the axes, by default AXES_6.
    title_prefix : str, optional
        Prefix for the figure titles, by default "Test".
    split_forces_moments : bool, optional
        Whether to create separate figures for forces and moments, by default True.
    Returns
    -------
    tuple[plt.Figure, ...]
        Tuple of matplotlib Figures.
    """

    def _plot_one(ax: plt.Axes, yt: np.ndarray, yp: np.ndarray, name: str) -> None:
        ax.scatter(yt, yp, alpha=0.7)

        lo = float(min(np.min(yt), np.min(yp)))
        hi = float(max(np.max(yt), np.max(yp)))
        ax.plot([lo, hi], [lo, hi], linewidth=1)

        ax.set_title(name)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    return _make_axis_figures(
        y_true=y_true,
        y_pred=y_pred,
        axis_names=axis_names,
        title_prefix=title_prefix,
        split_forces_moments=split_forces_moments,
        panel_title="True vs Predicted",
        plot_one_axis=_plot_one,
    )


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    axis_names: Sequence[str] = AXES_6,
    title_prefix: str = "Test",
    split_forces_moments: bool = True,
) -> tuple[plt.Figure, ...]:
    """
    Scatter plot of residuals (predicted - true) vs true values, per axis, with y=0 reference line.
    Parameters
    ----------
    y_true : np.ndarray
        True values, shape (n_samples, n_axes).
    y_pred : np.ndarray
        Predicted values, shape (n_samples, n_axes).
    axis_names : Sequence[str], optional
        Names of the axes, by default AXES_6.
    title_prefix : str, optional
        Prefix for the figure titles, by default "Test".
    split_forces_moments : bool, optional
        Whether to create separate figures for forces and moments, by default True.
    Returns
    -------
    tuple[plt.Figure, ...]
        Tuple of matplotlib Figures.
    """

    def _plot_one(ax: plt.Axes, yt: np.ndarray, yp: np.ndarray, name: str) -> None:
        err = yp - yt
        ax.scatter(yt, err, alpha=0.7)
        ax.axhline(0.0, linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("True")
        ax.set_ylabel("Residual (Predicted - True)")

    return _make_axis_figures(
        y_true=y_true,
        y_pred=y_pred,
        axis_names=axis_names,
        title_prefix=title_prefix,
        split_forces_moments=split_forces_moments,
        panel_title="Residuals",
        plot_one_axis=_plot_one,
    )
