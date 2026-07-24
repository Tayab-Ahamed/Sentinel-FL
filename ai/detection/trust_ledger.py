"""
ai/detection/trust_ledger.py — L4: Trust Ledger (full production implementation).

Implements an append-only JSON-lines store for ``TrustLedgerEntry`` records
(SCHEMAS.md §TrustLedgerEntry) plus an in-memory reputation cache, query
API, and all visualization-data helpers needed by the dashboard.

Design contracts (ARCHITECTURE.md §7.5):
  - ``add_entry()`` NEVER raises into the caller's control flow.
  - I/O failure → in-memory buffer with retry on next call.
  - Ledger write failure does NOT gate a training round or detection decision.
  - Append-only on disk; no record is ever modified or deleted.

Public surface:
  FileTrustLedger
    add_entry(entry) → None
    flush_buffer() → int           # returns #entries flushed
    get_score(subject_id) → TrustScore | None
    get_all_scores() → list[TrustScore]
    decay_scores(current_round) → None
    load_entries() → list[TrustLedgerEntry]
    query(q: TrustLedgerQuery) → list[TrustLedgerEntry]
    get_client_history(client_id, max_rounds) → list[TrustLedgerEntry]
    reputation_heatmap(round_range) → dict
    top_k_suspicious(k) → list[TrustScore]
    suspicious_above(threshold) → list[TrustScore]
    round_summary(round_num) → dict
    export_snapshot(round_num, threshold) → list[ReputationSnapshot]
    get_stats() → dict
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from ai.fl_core.schemas import (
    ReputationSnapshot,
    TrustLedgerEntry,
    TrustLedgerQuery,
    TrustScore,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FileTrustLedger
# ---------------------------------------------------------------------------


class FileTrustLedger:
    """Filesystem-backed Trust Ledger with full query and analytics API.

    Args:
        ledger_path: Path to the append-only ``.jsonl`` ledger file.
        decay_rate: Exponential decay applied to ``TrustScore.score`` per
            round.  0.0 = no decay; 0.1 = 10% reduction per round.
        suspicious_threshold: Score at or above which a client is flagged
            as suspicious.  Used by ``suspicious_above()`` and
            ``top_k_suspicious()``.
        max_history_per_client: Maximum in-memory ``TrustLedgerEntry``
            objects cached per ``subject_id``.  Older entries are evicted.
        write_max_retries: Number of write attempts before buffering.
        write_retry_backoff_ms: Sleep between retry attempts (ms).
        sentinel_logger: Optional ``StructuredLogger``; if provided, emits
            ``L4 ledger_entry_added`` events.
        warm_start: If True, replay existing ledger on init to rebuild the
            in-memory score cache.  Set False in tests for a blank slate.
    """

    def __init__(
        self,
        ledger_path: str | Path,
        decay_rate: float = 0.1,
        suspicious_threshold: float = 0.5,
        max_history_per_client: int = 200,
        write_max_retries: int = 3,
        write_retry_backoff_ms: int = 50,
        sentinel_logger: Any | None = None,
        warm_start: bool = True,
    ) -> None:
        self._path = Path(ledger_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._decay_rate = decay_rate
        self._suspicious_threshold = suspicious_threshold
        self._max_history = max_history_per_client
        self._max_retries = write_max_retries
        self._backoff_s = write_retry_backoff_ms / 1000.0
        self._sentinel_logger = sentinel_logger

        # In-memory caches
        self._scores: dict[str, TrustScore] = {}
        # subject_id → list of entries (most-recent last, capped at max_history)
        self._subject_index: dict[str, list[TrustLedgerEntry]] = defaultdict(list)
        # round_num → list of entry_ids
        self._round_index: dict[int, list[str]] = defaultdict(list)
        # Total entries written
        self._total_entries: int = 0
        # Buffered entries pending retry
        self._buffer: list[TrustLedgerEntry] = []

        if warm_start and self._path.exists():
            self._replay_ledger()

    # ------------------------------------------------------------------
    # Core write API
    # ------------------------------------------------------------------

    def add_entry(self, entry: TrustLedgerEntry) -> None:
        """Append a ``TrustLedgerEntry`` and update all in-memory indices.

        Never raises.  I/O failures are buffered and retried on the next call.

        Args:
            entry: The flag event to record.
        """
        # First try to flush any buffered entries
        if self._buffer:
            self.flush_buffer()

        try:
            self._write_entry_with_retry(entry)
            self._ingest_entry(entry)
        except Exception as exc:
            logger.warning(
                "TrustLedger.add_entry: write failed (buffering entry %s): %s",
                entry.entry_id,
                exc,
            )
            self._buffer.append(entry)
            # Still ingest into memory even if disk write failed
            self._ingest_entry(entry)

    def flush_buffer(self) -> int:
        """Write all buffered entries to disk.

        Returns:
            Number of entries successfully flushed.
        """
        if not self._buffer:
            return 0
        flushed = 0
        still_buffered: list[TrustLedgerEntry] = []
        for entry in self._buffer:
            try:
                self._write_entry_with_retry(entry)
                flushed += 1
            except Exception as exc:
                logger.warning(
                    "TrustLedger.flush_buffer: entry %s still failing: %s",
                    entry.entry_id,
                    exc,
                )
                still_buffered.append(entry)
        self._buffer = still_buffered
        if flushed > 0:
            logger.debug("TrustLedger.flush_buffer: flushed %d buffered entries.", flushed)
        return flushed

    # ------------------------------------------------------------------
    # Score management (existing API — unchanged signatures)
    # ------------------------------------------------------------------

    def get_score(self, subject_id: str) -> TrustScore | None:
        """Return the current ``TrustScore`` for a client/label, or ``None``."""
        return self._scores.get(subject_id)

    def get_all_scores(self) -> list[TrustScore]:
        """Return all current ``TrustScore`` objects (clients and labels)."""
        return list(self._scores.values())

    def decay_scores(self, current_round: int) -> None:
        """Apply exponential decay to all cached ``TrustScore`` objects.

        Must be called once per FL round after L1 processing completes.

        Args:
            current_round: The round number to stamp on decayed scores.
        """
        for score in self._scores.values():
            score.score = max(0.0, score.score * (1.0 - self._decay_rate))
            score.last_updated_round = current_round
        logger.debug(
            "TrustLedger.decay_scores: decayed %d scores at round %d.",
            len(self._scores),
            current_round,
        )

    # ------------------------------------------------------------------
    # Disk read
    # ------------------------------------------------------------------

    def load_entries(self) -> list[TrustLedgerEntry]:
        """Read all entries from the ledger file.

        Returns an empty list if the file does not exist yet.

        Returns:
            List of ``TrustLedgerEntry`` objects in append order.
        """
        if not self._path.exists():
            return []
        entries: list[TrustLedgerEntry] = []
        with open(self._path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(TrustLedgerEntry(**json.loads(line)))
                except Exception as exc:
                    logger.warning("TrustLedger: skipping malformed line: %s", exc)
        return entries

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def query(self, q: TrustLedgerQuery) -> list[TrustLedgerEntry]:
        """Return entries matching all filters in ``q`` (AND semantics).

        Args:
            q: ``TrustLedgerQuery`` with optional subject/layer/round/score
               filters.

        Returns:
            Matching entries, newest first.  Respects ``q.limit``.
        """
        # Collect candidates from the appropriate index
        if q.subject_ids is not None:
            candidates: list[TrustLedgerEntry] = []
            for sid in q.subject_ids:
                candidates.extend(self._subject_index.get(sid, []))
        else:
            # Flatten all subject histories
            candidates = [e for entries in self._subject_index.values() for e in entries]

        # Apply all filters
        results: list[TrustLedgerEntry] = []
        for e in candidates:
            if q.subject_types is not None and e.subject_type not in q.subject_types:
                continue
            if q.layers is not None and e.layer_id not in q.layers:
                continue
            if q.round_min is not None and (e.round_num is None or e.round_num < q.round_min):
                continue
            if q.round_max is not None and (e.round_num is None or e.round_num > q.round_max):
                continue
            if q.min_score is not None and e.score < q.min_score:
                continue
            if q.max_score is not None and e.score > q.max_score:
                continue
            results.append(e)

        # Deduplicate (a client may appear in both subject and flat iteration)
        seen: set[str] = set()
        unique: list[TrustLedgerEntry] = []
        for e in results:
            if e.entry_id not in seen:
                seen.add(e.entry_id)
                unique.append(e)

        # Sort newest first (by round_num, then insertion order is preserved)
        unique.sort(key=lambda e: e.round_num if e.round_num is not None else -1, reverse=True)

        if q.limit is not None:
            unique = unique[: q.limit]
        return unique

    def get_client_history(
        self,
        client_id: str,
        max_rounds: int | None = None,
    ) -> list[TrustLedgerEntry]:
        """Return all entries for one client, newest first.

        Args:
            client_id: Subject ID to look up.
            max_rounds: If set, only return entries from the last
                ``max_rounds`` rounds.

        Returns:
            List of ``TrustLedgerEntry`` objects, newest first.
        """
        entries = list(reversed(self._subject_index.get(client_id, [])))
        if max_rounds is not None and entries:
            max_round = (
                max(e.round_num for e in entries if e.round_num is not None)
                if any(e.round_num is not None for e in entries)
                else 0
            )
            cutoff = max_round - max_rounds
            entries = [e for e in entries if e.round_num is None or e.round_num >= cutoff]
        return entries

    # ------------------------------------------------------------------
    # Visualization data
    # ------------------------------------------------------------------

    def reputation_heatmap(
        self,
        round_range: tuple[int | None, int | None] = (None, None),
    ) -> dict[str, Any]:
        """Return a client × round trust-score matrix for the dashboard heatmap.

        Corresponds to API.md §4 ``reputation_heatmap`` endpoint format.

        Args:
            round_range: ``(round_min, round_max)`` inclusive.  ``None``
                means no bound.

        Returns:
            Dict with keys:
            - ``"client_ids"``: ordered list of client IDs
            - ``"rounds"``: ordered list of round numbers
            - ``"matrix"``: 2-D list ``[clients][rounds]`` of trust scores
              (``null`` where no data for that round)
            - ``"current_scores"``: ``{client_id: score}`` of latest values
        """
        rmin, rmax = round_range

        # Determine which rounds to include
        all_rounds: set[int] = set()
        for entries in self._subject_index.values():
            for e in entries:
                if e.round_num is not None:
                    if (rmin is None or e.round_num >= rmin) and (
                        rmax is None or e.round_num <= rmax
                    ):
                        all_rounds.add(e.round_num)

        rounds_sorted = sorted(all_rounds)
        client_ids = sorted(self._subject_index.keys())

        # Build per-client, per-round score map
        # Use the max anomaly score for that client+round as the heatmap value
        score_map: dict[str, dict[int, float]] = {cid: {} for cid in client_ids}
        for cid in client_ids:
            for e in self._subject_index[cid]:
                if e.round_num is not None and e.round_num in all_rounds:
                    existing = score_map[cid].get(e.round_num, 0.0)
                    score_map[cid][e.round_num] = max(existing, e.score)

        matrix = [[score_map[cid].get(rnd) for rnd in rounds_sorted] for cid in client_ids]

        return {
            "client_ids": client_ids,
            "rounds": rounds_sorted,
            "matrix": matrix,
            "current_scores": {
                cid: round(self._scores[cid].score, 4) for cid in client_ids if cid in self._scores
            },
        }

    def top_k_suspicious(self, k: int = 5) -> list[TrustScore]:
        """Return the top-k most suspicious clients by current trust score.

        Args:
            k: Number of clients to return.

        Returns:
            List of ``TrustScore`` objects sorted by score descending
            (highest = most suspicious first).
        """
        return sorted(self._scores.values(), key=lambda ts: ts.score, reverse=True)[:k]

    def suspicious_above(self, threshold: float | None = None) -> list[TrustScore]:
        """Return clients whose trust score >= threshold.

        Args:
            threshold: Override. If None, uses ``self._suspicious_threshold``.

        Returns:
            List of ``TrustScore`` objects for suspicious clients.
        """
        thr = threshold if threshold is not None else self._suspicious_threshold
        return [ts for ts in self._scores.values() if ts.score >= thr]

    def round_summary(self, round_num: int) -> dict[str, Any]:
        """Return per-round statistics for a given FL round.

        Args:
            round_num: The round to summarise.

        Returns:
            Dict with keys: ``round_num``, ``n_entries``, ``n_flagged_clients``,
            ``mean_score``, ``max_score``, ``layers_active``, ``flagged_client_ids``.
        """
        entry_ids = self._round_index.get(round_num, [])
        # Look up entries from subject index by entry_id
        id_set = set(entry_ids)
        entries: list[TrustLedgerEntry] = []
        for subject_entries in self._subject_index.values():
            for e in subject_entries:
                if e.entry_id in id_set:
                    entries.append(e)

        client_ids = sorted({e.subject_id for e in entries if e.subject_type == "client"})
        scores = [e.score for e in entries]
        layers = sorted({e.layer_id for e in entries})
        return {
            "round_num": round_num,
            "n_entries": len(entries),
            "n_flagged_clients": len(client_ids),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "max_score": round(max(scores), 4) if scores else 0.0,
            "layers_active": layers,
            "flagged_client_ids": client_ids,
        }

    def export_snapshot(
        self,
        round_num: int,
        threshold: float | None = None,
    ) -> list[ReputationSnapshot]:
        """Return a ``ReputationSnapshot`` for every known client.

        Produces the data structure the dashboard uses for a round-level
        reputation report.

        Args:
            round_num: The round to snapshot (used as the ``round_num``
                field; does not filter entries to only that round).
            threshold: Suspicious threshold override.

        Returns:
            List of ``ReputationSnapshot`` objects, one per client.
        """
        thr = threshold if threshold is not None else self._suspicious_threshold
        snapshots: list[ReputationSnapshot] = []

        for cid, entries in self._subject_index.items():
            ts = self._scores.get(cid)
            current_score = ts.score if ts else 0.0
            contributing = ts.contributing_events if ts else []

            # Collect layer IDs that have flagged this client
            flagged_layers = sorted({e.layer_id for e in entries})

            # Extract anomaly_score and norm history from evidence dicts
            # (populated by UpdateGuard via TrustScoreManager)
            rounds_seen: dict[int, dict[str, float]] = {}
            for e in entries:
                if e.round_num is not None:
                    rnd = e.round_num
                    ev = e.evidence or {}
                    if rnd not in rounds_seen:
                        rounds_seen[rnd] = {}
                    # Take the max within the same round
                    if "anomaly_score" in ev:
                        rounds_seen[rnd]["anomaly"] = max(
                            rounds_seen[rnd].get("anomaly", 0.0),
                            float(ev["anomaly_score"]),
                        )
                    if "norm" in ev:
                        rounds_seen[rnd]["norm"] = max(
                            rounds_seen[rnd].get("norm", 0.0),
                            float(ev["norm"]),
                        )

            sorted_rounds = sorted(rounds_seen.keys())
            anomaly_history = [round(rounds_seen[r].get("anomaly", 0.0), 4) for r in sorted_rounds]
            norm_history = [round(rounds_seen[r].get("norm", 0.0), 6) for r in sorted_rounds]

            snapshots.append(
                ReputationSnapshot(
                    client_id=cid,
                    round_num=round_num,
                    trust_score=round(current_score, 4),
                    contributing_entry_count=len(contributing),
                    flagged_by_layers=flagged_layers,
                    anomaly_score_history=anomaly_history,
                    norm_history=norm_history,
                    is_suspicious=current_score >= thr,
                )
            )
        return sorted(snapshots, key=lambda s: s.trust_score, reverse=True)

    def get_stats(self) -> dict[str, Any]:
        """Return global ledger statistics.

        Returns:
            Dict with keys: ``total_entries``, ``n_clients``, ``n_buffered``,
            ``ledger_path``, ``suspicious_count``, ``mean_score``.
        """
        scores = [ts.score for ts in self._scores.values()]
        return {
            "total_entries": self._total_entries,
            "n_clients": len(self._scores),
            "n_buffered": len(self._buffer),
            "ledger_path": str(self._path),
            "suspicious_count": sum(1 for s in scores if s >= self._suspicious_threshold),
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_entry_with_retry(self, entry: TrustLedgerEntry) -> None:
        """Write one entry to disk, retrying up to ``_max_retries`` times."""
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(entry.model_dump_json() + "\n")
                return  # success
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    time.sleep(self._backoff_s)
        raise OSError(
            f"TrustLedger: failed to write entry after {self._max_retries} attempts: {last_exc}"
        )

    def _ingest_entry(self, entry: TrustLedgerEntry) -> None:
        """Update all in-memory indices from a new entry."""
        self._total_entries += 1

        # Subject index (per-client history)
        history = self._subject_index[entry.subject_id]
        history.append(entry)
        # Evict oldest if over cap
        if len(history) > self._max_history:
            del history[: len(history) - self._max_history]

        # Round index
        if entry.round_num is not None:
            self._round_index[entry.round_num].append(entry.entry_id)

        # Score cache
        self._update_score(entry)

        # Structured logging
        self._emit_l4_event(entry)

    def _update_score(self, entry: TrustLedgerEntry) -> None:
        """Update the in-memory TrustScore cache from a new entry."""
        sid = entry.subject_id
        existing = self._scores.get(sid)
        prev_score = existing.score if existing else 0.0
        prev_events = existing.contributing_events if existing else []

        # Score update formula: accumulate + weight, cap at 1.0
        new_score = min(1.0, prev_score + entry.score * 0.5)

        subject_type = entry.subject_type
        if subject_type == "input":
            subject_type = "client"

        self._scores[sid] = TrustScore(
            subject_type=subject_type,  # type: ignore[arg-type]
            subject_id=sid,
            score=new_score,
            last_updated_round=entry.round_num or 0,
            contributing_events=[*prev_events, entry.entry_id][-50:],
        )

    def _emit_l4_event(self, entry: TrustLedgerEntry) -> None:
        """Emit a structured log event to the sentinel logger."""
        if self._sentinel_logger is None:
            return
        try:
            self._sentinel_logger.log(
                "L4",
                "ledger_entry_added",
                {
                    "entry_id": entry.entry_id,
                    "subject_id": entry.subject_id,
                    "subject_type": entry.subject_type,
                    "layer_id": entry.layer_id,
                    "round_num": entry.round_num,
                    "score": entry.score,
                    "reason": entry.reason,
                },
            )
        except Exception as exc:
            logger.debug("TrustLedger: L4 event log failed (non-fatal): %s", exc)

    def _replay_ledger(self) -> None:
        """Rebuild in-memory caches by replaying the existing ledger file."""
        entries = self.load_entries()
        for entry in entries:
            self._ingest_entry(entry)
        logger.info(
            "TrustLedger: warm-start replayed %d entries from %s",
            len(entries),
            self._path,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ledger_path(self) -> Path:
        """Absolute path to the ``.jsonl`` ledger file."""
        return self._path

    @property
    def suspicious_threshold(self) -> float:
        """Current suspicious score threshold."""
        return self._suspicious_threshold

    @suspicious_threshold.setter
    def suspicious_threshold(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"suspicious_threshold must be in [0, 1], got {value}")
        self._suspicious_threshold = value

    @property
    def buffered_count(self) -> int:
        """Number of entries currently in the retry buffer."""
        return len(self._buffer)

    @property
    def total_entries(self) -> int:
        """Total entries ingested (written + buffered)."""
        return self._total_entries
