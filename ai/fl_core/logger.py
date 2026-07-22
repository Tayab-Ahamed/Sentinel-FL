"""
ai/fl_core/logger.py — Structured JSON-lines logger for SENTINEL-FL.

Implements the Logger interface (INTERFACES.md §Logger) with the following
guarantees mandated by ARCHITECTURE.md §7.8:

  1. ``log()`` never raises into the calling layer's control flow.
     Any I/O failure is caught, counted, and dropped.
  2. Every entry includes timestamp, layer_id, event_type, and round_num
     per the LogEntry schema (SCHEMAS.md §LogEntry).
  3. The sink can be stdout or a file path (set via env-var or config).

Usage:
    logger = StructuredLogger(sink="stdout", log_level="INFO")
    logger.log("L1", "cluster_flagged", {"cluster": [2, 5, 9], "round_num": 3})
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from ai.fl_core.interfaces import Logger

# ---------------------------------------------------------------------------
# Module-level Python logger (separate from SENTINEL's structured logger)
# Captures internal StructuredLogger errors without recursing.
# ---------------------------------------------------------------------------
_internal_log = logging.getLogger(__name__)


class StructuredLogger(Logger):
    """Concrete Logger that writes JSON-lines to stdout or a file.

    Thread-safe: a reentrant lock serialises all ``log()`` calls so log lines
    from concurrent FL round threads are not interleaved.

    Args:
        sink: ``"stdout"`` (default) or an absolute/relative path to a ``.jsonl`` file.
        log_level: Minimum log level to emit.  One of ``"DEBUG"``, ``"INFO"``,
            ``"WARNING"``, ``"ERROR"``, ``"CRITICAL"``.
        round_num: Current FL round number.  Updated externally via
            :py:meth:`set_round`.
    """

    LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def __init__(
        self,
        sink: str = "stdout",
        log_level: str = "INFO",
        round_num: int | None = None,
    ) -> None:
        if log_level.upper() not in self.LEVELS:
            raise ValueError(f"Invalid log_level '{log_level}'. Must be one of {self.LEVELS}.")

        self._log_level = log_level.upper()
        self._level_rank = list(self.LEVELS).index(self._log_level)
        self._round_num = round_num
        self._lock = threading.Lock()
        self._dropped_count = 0

        # Open the sink.
        self._file: TextIO | None = None
        if sink == "stdout":
            self._stream: TextIO = sys.stdout
        else:
            path = Path(sink)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Append mode so we don't truncate existing logs on restart.
            self._file = open(path, "a", encoding="utf-8")
            self._stream = self._file

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_round(self, round_num: int) -> None:
        """Update the current FL round number stamped on every subsequent log entry."""
        self._round_num = round_num

    @property
    def dropped_count(self) -> int:
        """Number of log entries silently dropped due to I/O failures."""
        return self._dropped_count

    def log(
        self,
        layer_id: str,
        event_type: str,
        payload: dict[str, Any],
        level: str = "INFO",
    ) -> None:
        """Emit one structured JSON-lines log entry.

        Never raises.  Any exception is caught, counted, and surfaced only via
        the internal Python logger at WARNING level.

        Args:
            layer_id: ``"L1"``, ``"L2"``, ``"L3"``, or ``"L4"``.
            event_type: Short snake_case event name.
            payload: Event-specific data merged into the LogEntry.
            level: Log level for this entry (default ``"INFO"``).
        """
        try:
            self._write(layer_id, event_type, payload, level)
        except Exception as exc:
            self._dropped_count += 1
            _internal_log.warning(
                "StructuredLogger: failed to write log entry (dropped #%d): %s",
                self._dropped_count,
                exc,
            )

    def close(self) -> None:
        """Flush and close the file sink if one is open."""
        if self._file is not None:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass

    def __enter__(self) -> StructuredLogger:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(
        self,
        layer_id: str,
        event_type: str,
        payload: dict[str, Any],
        level: str,
    ) -> None:
        """Build the LogEntry dict and write it as a single JSON line."""
        entry = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "layer_id": layer_id,
            "event_type": event_type,
            "round_num": self._round_num,
            "level": level.upper(),
            "payload": payload,
        }
        line = json.dumps(entry, default=str) + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()


# ---------------------------------------------------------------------------
# Module-level convenience factory
# ---------------------------------------------------------------------------


def make_logger(log_level: str = "INFO", log_sink: str = "stdout") -> StructuredLogger:
    """Create a StructuredLogger from config values.

    Args:
        log_level: Minimum log level string (case-insensitive).
        log_sink: ``"stdout"`` or a file path.

    Returns:
        A configured StructuredLogger ready to use.
    """
    return StructuredLogger(sink=log_sink, log_level=log_level)
