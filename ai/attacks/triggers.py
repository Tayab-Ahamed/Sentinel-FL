"""
ai/attacks/triggers.py — Trigger pattern generation for BadNets image attacks.

A ``TriggerPattern`` describes the visual stamp that is applied to poisoned
images.  Four shapes are supported:

  ``square``       — solid filled rectangle (the canonical BadNets patch)
  ``cross``        — plus-sign overlay
  ``checkerboard`` — alternating pixels (like a chess board) within the patch
  ``random_noise`` — per-pixel random values, seeded for reproducibility

Locations:
  ``bottom_right`` | ``top_left`` | ``top_right`` | ``bottom_left`` | ``center``

Opacity: 1.0 = fully opaque replacement; <1.0 = alpha-blend with original pixel.

Usage::

    pattern = TriggerFactory.make_square(size=4, location="bottom_right",
                                         color=1.0, opacity=1.0)
    X_triggered = apply_trigger(X_batch, pattern)   # (N, C, H, W) → same shape
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

TriggerShape = Literal["square", "cross", "checkerboard", "random_noise"]
TriggerLocation = Literal["bottom_right", "top_left", "top_right", "bottom_left", "center"]


# ---------------------------------------------------------------------------
# TriggerPattern dataclass
# ---------------------------------------------------------------------------


@dataclass
class TriggerPattern:
    """Fully describes a BadNets-style trigger stamp.

    Args:
        shape: Visual shape of the trigger patch.
        size: Side length (pixels) of the bounding box containing the trigger.
        location: Where on the image the trigger is placed.
        color: Trigger pixel value(s).  Scalar → applied to all channels.
            Tuple of length C → per-channel values.
        opacity: Blend ratio in [0, 1].  1.0 = fully opaque replacement;
            0.5 = average of trigger and original.
        seed: Random seed for ``random_noise`` shape (ignored otherwise).
    """

    shape: TriggerShape = "square"
    size: int = 4
    location: TriggerLocation = "bottom_right"
    color: float | tuple[float, ...] = 1.0
    opacity: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.size < 1:
            raise ValueError(f"TriggerPattern.size must be >= 1, got {self.size}")
        if not (0.0 <= self.opacity <= 1.0):
            raise ValueError(f"TriggerPattern.opacity must be in [0, 1], got {self.opacity}")

    def stamp(self, n_channels: int, h_img: int, w_img: int) -> np.ndarray:
        """Return a ``(C, H, W)`` binary/valued stamp array and a boolean mask.

        Returns:
            ``stamp_pixels`` — float32 array, same shape as the patch region.
            Shape: ``(n_channels, size, size)``.
        """
        s = min(self.size, h_img, w_img)  # clamp to image dims
        rng = np.random.default_rng(self.seed)

        # ── Build per-pixel pattern within (s, s) bounding box ──────────────
        if self.shape == "square":
            alpha_mask = np.ones((s, s), dtype=np.float32)

        elif self.shape == "cross":
            alpha_mask = np.zeros((s, s), dtype=np.float32)
            mid = s // 2
            alpha_mask[mid, :] = 1.0  # horizontal bar
            alpha_mask[:, mid] = 1.0  # vertical bar

        elif self.shape == "checkerboard":
            rows, cols = np.indices((s, s))
            alpha_mask = ((rows + cols) % 2).astype(np.float32)

        elif self.shape == "random_noise":
            alpha_mask = rng.uniform(0.0, 1.0, size=(s, s)).astype(np.float32)

        else:
            raise ValueError(f"Unknown trigger shape: {self.shape!r}")

        # ── Build per-channel color ──────────────────────────────────────────
        color: tuple[float, ...]
        if isinstance(self.color, (int, float)):
            color = tuple(float(self.color) for _ in range(n_channels))
        else:
            color = tuple(self.color)
            if len(color) != n_channels:
                raise ValueError(f"color tuple length {len(color)} != n_channels {n_channels}")

        # stamp shape: (C, s, s)
        stamp = np.stack([alpha_mask * c for c in color], axis=0).astype(np.float32)
        return stamp  # (C, s, s)

    def get_patch_slice(self, h_img: int, w_img: int) -> tuple[slice, slice]:
        """Return the ``(row_slice, col_slice)`` covering the trigger region.

        Args:
            h_img: Image height in pixels.
            w_img: Image width in pixels.

        Returns:
            ``(row_slice, col_slice)`` selecting the trigger region.
        """
        s = min(self.size, h_img, w_img)

        if self.location == "bottom_right":
            r0, c0 = h_img - s, w_img - s
        elif self.location == "top_left":
            r0, c0 = 0, 0
        elif self.location == "top_right":
            r0, c0 = 0, w_img - s
        elif self.location == "bottom_left":
            r0, c0 = h_img - s, 0
        elif self.location == "center":
            r0 = (h_img - s) // 2
            c0 = (w_img - s) // 2
        else:
            raise ValueError(f"Unknown trigger location: {self.location!r}")

        return slice(r0, r0 + s), slice(c0, c0 + s)


# ---------------------------------------------------------------------------
# apply_trigger — works on single image or batch
# ---------------------------------------------------------------------------


def apply_trigger(
    X: np.ndarray,
    pattern: TriggerPattern,
) -> np.ndarray:
    """Apply a trigger pattern to an image or batch of images.

    Args:
        X: Input image array.
            Single image: ``(C, H, W)`` float32.
            Batch: ``(N, C, H, W)`` float32.
        pattern: ``TriggerPattern`` to apply.

    Returns:
        New array of same shape/dtype as ``X`` with trigger applied.
        The original ``X`` is never mutated.

    Raises:
        ValueError: If ``X`` is not 3-D or 4-D.
    """
    if X.ndim == 3:
        return _apply_single(X, pattern)
    if X.ndim == 4:
        out = X.copy()
        for i in range(len(out)):
            out[i] = _apply_single(out[i], pattern)
        return out
    raise ValueError(f"apply_trigger: expected 3-D (C,H,W) or 4-D (N,C,H,W) array, got {X.shape}")


def _apply_single(img: np.ndarray, pattern: TriggerPattern) -> np.ndarray:
    """Apply trigger to a single ``(C, H, W)`` image."""
    C, H, W = img.shape
    out = img.copy()
    row_sl, col_sl = pattern.get_patch_slice(H, W)
    stamp = pattern.stamp(C, H, W)  # (C, s, s)
    # Blend: out = opacity * stamp + (1 - opacity) * original_patch
    original_patch = out[:, row_sl, col_sl]
    out[:, row_sl, col_sl] = pattern.opacity * stamp + (1.0 - pattern.opacity) * original_patch
    return out


# ---------------------------------------------------------------------------
# TriggerFactory — convenience constructors
# ---------------------------------------------------------------------------


class TriggerFactory:
    """Convenience factory for common trigger patterns."""

    @staticmethod
    def make_square(
        size: int = 4,
        location: TriggerLocation = "bottom_right",
        color: float | tuple[float, ...] = 1.0,
        opacity: float = 1.0,
    ) -> TriggerPattern:
        """Create a solid filled square trigger."""
        return TriggerPattern(
            shape="square",
            size=size,
            location=location,
            color=color,
            opacity=opacity,
        )

    @staticmethod
    def make_cross(
        size: int = 5,
        location: TriggerLocation = "bottom_right",
        color: float | tuple[float, ...] = 1.0,
        opacity: float = 1.0,
    ) -> TriggerPattern:
        """Create a plus-sign cross trigger."""
        return TriggerPattern(
            shape="cross",
            size=size,
            location=location,
            color=color,
            opacity=opacity,
        )

    @staticmethod
    def make_checkerboard(
        size: int = 4,
        location: TriggerLocation = "bottom_right",
        color: float | tuple[float, ...] = 1.0,
        opacity: float = 1.0,
    ) -> TriggerPattern:
        """Create a checkerboard trigger."""
        return TriggerPattern(
            shape="checkerboard",
            size=size,
            location=location,
            color=color,
            opacity=opacity,
        )

    @staticmethod
    def make_random_noise(
        size: int = 4,
        location: TriggerLocation = "bottom_right",
        seed: int = 0,
    ) -> TriggerPattern:
        """Create a random-noise trigger (reproducible via seed)."""
        return TriggerPattern(
            shape="random_noise",
            size=size,
            location=location,
            color=1.0,
            opacity=1.0,
            seed=seed,
        )

    @staticmethod
    def from_config(cfg: object) -> TriggerPattern:
        """Build a TriggerPattern from a config object (duck-typed).

        Reads ``cfg.trigger.*`` fields with sensible defaults.
        """
        trigger = getattr(cfg, "trigger", None)
        if trigger is None:
            logger.warning(
                "TriggerFactory.from_config: no trigger sub-config found; "
                "using defaults (4×4 square, bottom_right)."
            )
            return TriggerFactory.make_square()

        return TriggerPattern(
            shape=getattr(trigger, "shape", "square"),
            size=int(getattr(trigger, "size", 4)),
            location=getattr(trigger, "location", "bottom_right"),
            color=float(getattr(trigger, "color", 1.0)),
            opacity=float(getattr(trigger, "opacity", 1.0)),
            seed=int(getattr(trigger, "seed", 0)),
        )
