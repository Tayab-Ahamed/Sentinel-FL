"""
ai/attacks/visualizer.py — Poisoned sample visualisation.

``PoisonedSampleVisualizer`` generates matplotlib figures showing:

  1. **Sample grid**: Side-by-side clean vs. poisoned images.
     Poisoned images have a red border.

  2. **Trigger pattern**: The trigger stamp in isolation.

  3. **ASR over rounds**: Line chart of ASR and C-Acc vs. round number.

All figures are returned as ``matplotlib.figure.Figure`` objects so the caller
can decide whether to display, save, or embed them in a report.  Use
``save_figure(fig, path)`` for disk persistence.

Supports both MNIST (1-channel, greyscale) and CIFAR-10 (3-channel, RGB).

Optional dependency: ``matplotlib``.  An ``ImportError`` with a clear
install message is raised if it is not available.

Usage::

    viz = PoisonedSampleVisualizer()
    fig = viz.plot_sample_grid(X_clean, X_poisoned, mask, n_cols=8)
    viz.save_figure(fig, "experiments/exp_001/poisoned_grid.png")
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend — safe in all environments
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    Figure = object  # type: ignore[misc,assignment]


def _require_matplotlib() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualisation. Install it with: pip install matplotlib"
        )


class PoisonedSampleVisualizer:
    """Generates publication-quality visualisations of poisoned images.

    Args:
        dpi: Dots per inch for saved figures.
        cmap_gray: Colormap for greyscale (1-channel) images.
    """

    def __init__(
        self,
        dpi: int = 120,
        cmap_gray: str = "gray",
    ) -> None:
        _require_matplotlib()
        self._dpi = dpi
        self._cmap = cmap_gray

    # ------------------------------------------------------------------
    # Public figures
    # ------------------------------------------------------------------

    def plot_sample_grid(
        self,
        X_clean: np.ndarray,
        X_poisoned: np.ndarray,
        mask: np.ndarray,
        n_cols: int = 8,
        n_samples: int = 16,
        title: str = "Clean (top) vs. Poisoned (bottom) — red border = poisoned",
    ) -> Figure:
        """Grid of clean images (top row) paired with their poisoned versions.

        Only the first ``n_samples`` masked (poisoned) samples are shown.

        Args:
            X_clean: Clean image batch ``(N, C, H, W)`` float32.
            X_poisoned: Poisoned batch of same shape.
            mask: Boolean mask of shape ``(N,)``; True = poisoned sample.
            n_cols: Number of columns in the grid.
            n_samples: Maximum number of poisoned samples to display.
            title: Figure super-title.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        poisoned_idx = np.where(mask)[0][:n_samples]
        if len(poisoned_idx) == 0:
            logger.warning("plot_sample_grid: no poisoned samples in mask.")
            poisoned_idx = np.arange(min(n_samples, len(X_clean)))

        n = len(poisoned_idx)
        n_cols = min(n_cols, n)
        n_rows = 2 * ((n + n_cols - 1) // n_cols)  # 2 rows per row-pair

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(n_cols * 1.5, n_rows * 1.5),
            dpi=self._dpi,
        )
        axes = np.array(axes).reshape(n_rows, n_cols)

        # Turn off all axes first
        for ax in axes.flat:
            ax.axis("off")

        for col, idx in enumerate(poisoned_idx):
            c_in_row = col % n_cols
            pair_row = (col // n_cols) * 2

            # Top: clean
            self._imshow(axes[pair_row, c_in_row], X_clean[idx])
            axes[pair_row, c_in_row].set_title("clean", fontsize=6, pad=1)

            # Bottom: poisoned (red border)
            self._imshow(axes[pair_row + 1, c_in_row], X_poisoned[idx])
            axes[pair_row + 1, c_in_row].set_title("poisoned", fontsize=6, pad=1, color="red")
            for spine in axes[pair_row + 1, c_in_row].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(2)
                spine.set_visible(True)

        fig.suptitle(title, fontsize=9, y=1.01)
        fig.tight_layout()
        return fig

    def plot_trigger_pattern(
        self,
        pattern: object,  # TriggerPattern — avoid circular import at module level
        input_shape: tuple[int, int, int] = (1, 28, 28),
        title: str = "Trigger Pattern",
    ) -> Figure:
        """Render the trigger stamp in isolation on a blank image.

        Args:
            pattern: ``TriggerPattern`` instance.
            input_shape: ``(C, H, W)`` of the target image.
            title: Figure title.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        from ai.attacks.triggers import apply_trigger  # local import

        C, H, W = input_shape
        blank = np.zeros((C, H, W), dtype=np.float32)
        triggered = apply_trigger(blank, pattern)

        fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=self._dpi)
        self._imshow(axes[0], blank)
        axes[0].set_title("Blank (no trigger)", fontsize=8)
        axes[0].axis("off")

        self._imshow(axes[1], triggered)
        axes[1].set_title("With trigger", fontsize=8)
        axes[1].axis("off")

        fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        return fig

    def plot_asr_curve(
        self,
        round_numbers: list[int],
        asr_values: list[float],
        clean_acc_values: list[float],
        title: str = "Attack Success Rate & Clean Accuracy over Rounds",
    ) -> Figure:
        """Line chart of ASR and C-Acc vs. FL round.

        Args:
            round_numbers: List of round indices.
            asr_values: ASR per round (floats in [0, 1]).
            clean_acc_values: Clean accuracy per round.
            title: Figure title.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        fig, ax = plt.subplots(figsize=(7, 3.5), dpi=self._dpi)

        ax.plot(
            round_numbers,
            asr_values,
            marker="o",
            color="#e74c3c",
            linewidth=2,
            label="ASR (Attack Success Rate)",
        )
        ax.plot(
            round_numbers,
            clean_acc_values,
            marker="s",
            color="#2ecc71",
            linewidth=2,
            label="C-Acc (Clean Accuracy)",
        )

        ax.set_xlabel("FL Round", fontsize=10)
        ax.set_ylabel("Rate", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0, decimals=0))

        fig.tight_layout()
        return fig

    def plot_poison_distribution(
        self,
        mask: np.ndarray,
        y_original: np.ndarray,
        y_poisoned: np.ndarray,
        n_classes: int,
        title: str = "Label Distribution: Clean vs. Poisoned",
    ) -> Figure:
        """Bar chart comparing class distributions before/after poisoning.

        Args:
            mask: Boolean mask; True = poisoned sample.
            y_original: Original label array before poisoning.
            y_poisoned: Label array after poisoning.
            n_classes: Total number of classes.
            title: Figure title.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        classes = np.arange(n_classes)
        orig_counts = np.bincount(y_original, minlength=n_classes)
        pois_counts = np.bincount(y_poisoned, minlength=n_classes)

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=self._dpi)
        axes[0].bar(classes, orig_counts, color="#3498db")
        axes[0].set_title("Original", fontsize=9)
        axes[0].set_xlabel("Class")
        axes[0].set_ylabel("Count")

        axes[1].bar(classes, pois_counts, color="#e67e22")
        axes[1].set_title("After Poisoning", fontsize=9)
        axes[1].set_xlabel("Class")

        n_poisoned = int(mask.sum())
        fig.suptitle(f"{title}\n({n_poisoned}/{len(mask)} samples poisoned)", fontsize=9)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_figure(
        self,
        fig: Figure,
        path: str | Path,
        dpi: int | None = None,
    ) -> Path:
        """Save a figure to disk, creating parent directories as needed.

        Args:
            fig: The matplotlib figure to save.
            path: Output file path (PNG, PDF, SVG supported).
            dpi: Override DPI (defaults to ``self._dpi``).

        Returns:
            Resolved ``Path`` of the saved file.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=dpi or self._dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("PoisonedSampleVisualizer: saved figure to %s", out)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _imshow(self, ax: plt.Axes, img: np.ndarray) -> None:
        """Display a ``(C, H, W)`` image on a matplotlib Axes.

        Handles greyscale (C=1) and RGB (C=3).  Clips to [0, 1].
        """
        img_clipped = np.clip(img, 0.0, 1.0)
        if img_clipped.shape[0] == 1:
            ax.imshow(img_clipped[0], cmap=self._cmap, vmin=0, vmax=1)
        elif img_clipped.shape[0] == 3:
            ax.imshow(img_clipped.transpose(1, 2, 0))  # (C,H,W) → (H,W,C)
        else:
            # Multi-channel: show first channel
            ax.imshow(img_clipped[0], cmap=self._cmap, vmin=0, vmax=1)
