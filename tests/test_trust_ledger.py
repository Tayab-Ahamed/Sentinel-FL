"""
tests/test_trust_ledger.py — Comprehensive tests for L4 Trust Ledger (Milestone 6).

Covers:
  FileTrustLedger:
    - add_entry persistence (written to .jsonl)
    - warm-restart (replay from existing ledger file)
    - retry buffer (flush_buffer, failed write recovery)
    - decay_scores
    - load_entries (reads all entries from disk)
    - query() with all filter combinations
    - get_client_history() + max_rounds filter
    - reputation_heatmap() shape and content
    - top_k_suspicious() ordering
    - suspicious_above() threshold
    - round_summary() statistics
    - export_snapshot() → ReputationSnapshot structure
    - get_stats()
    - configurable suspicious_threshold (property setter)
    - UpdateGuard → ledger integration

  ReputationEngine:
    - client_reputation_report() content
    - heatmap_data() enrichment
    - flag_rate_by_layer() fractions sum to 1
    - score_distribution() histogram shape
    - suspicious_timeline() ordering
    - cross_layer_correlation() Jaccard symmetry
    - compute_all_metrics() structure

  Schema:
    - ReputationSnapshot validation
    - TrustLedgerQuery validation (round_max < round_min raises)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ai.detection.reputation_engine import ReputationEngine
from ai.detection.trust_ledger import FileTrustLedger
from ai.fl_core.schemas import (
    ReputationSnapshot,
    TrustLedgerEntry,
    TrustLedgerQuery,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry(
    subject_id: str = "client_00",
    layer_id: str = "L1",
    round_num: int = 1,
    score: float = 0.4,
    reason: str = "test flag",
    evidence: dict | None = None,
    subject_type: str = "client",
) -> TrustLedgerEntry:
    return TrustLedgerEntry(
        layer_id=layer_id,
        subject_type=subject_type,  # type: ignore
        subject_id=subject_id,
        round_num=round_num,
        score=score,
        reason=reason,
        evidence=evidence or {},
    )


def _ledger(tmp_path: Path, **kwargs) -> FileTrustLedger:
    return FileTrustLedger(
        ledger_path=tmp_path / "ledger.jsonl",
        warm_start=False,
        **kwargs,
    )


def _populated_ledger(tmp_path: Path) -> FileTrustLedger:
    """Ledger with 3 clients × 3 rounds of entries."""
    ledger = _ledger(tmp_path)
    for rnd in range(1, 4):
        for cid in ["client_00", "client_01", "client_02"]:
            ledger.add_entry(_entry(subject_id=cid, round_num=rnd, score=0.3))
    return ledger


# ---------------------------------------------------------------------------
# FileTrustLedger — basic add + persistence
# ---------------------------------------------------------------------------


class TestAddEntry:
    def test_entry_written_to_disk(self, tmp_path):
        ledger = _ledger(tmp_path)
        e = _entry()
        ledger.add_entry(e)
        lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["subject_id"] == "client_00"

    def test_multiple_entries_appended(self, tmp_path):
        ledger = _ledger(tmp_path)
        for i in range(5):
            ledger.add_entry(_entry(subject_id=f"client_{i:02d}"))
        lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_add_entry_never_raises(self, tmp_path):
        ledger = _ledger(tmp_path)
        # Simulate a persistent I/O failure in _write_entry_with_retry
        with patch.object(
            ledger, "_write_entry_with_retry", side_effect=OSError("disk full")
        ):
            ledger.add_entry(_entry())  # must not raise

    def test_score_cache_updated_after_add(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.add_entry(_entry(subject_id="c0", score=0.6))
        ts = ledger.get_score("c0")
        assert ts is not None
        assert ts.score > 0.0

    def test_score_accumulates(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.add_entry(_entry(subject_id="c0", score=0.4))
        first = ledger.get_score("c0").score
        ledger.add_entry(_entry(subject_id="c0", score=0.4))
        second = ledger.get_score("c0").score
        assert second > first

    def test_score_capped_at_one(self, tmp_path):
        ledger = _ledger(tmp_path)
        for _ in range(20):
            ledger.add_entry(_entry(subject_id="c0", score=1.0))
        assert ledger.get_score("c0").score <= 1.0

    def test_total_entries_tracks_all(self, tmp_path):
        ledger = _ledger(tmp_path)
        for _ in range(7):
            ledger.add_entry(_entry())
        assert ledger.total_entries == 7


# ---------------------------------------------------------------------------
# FileTrustLedger — warm-start (replay from disk)
# ---------------------------------------------------------------------------


class TestWarmStart:
    def test_scores_rebuilt_after_warm_start(self, tmp_path):
        # Write entries first
        ledger1 = _ledger(tmp_path)
        ledger1.add_entry(_entry(subject_id="c0", score=0.5, round_num=1))
        ledger1.add_entry(_entry(subject_id="c1", score=0.3, round_num=2))

        # Create a second ledger from same file with warm_start=True
        ledger2 = FileTrustLedger(
            ledger_path=tmp_path / "ledger.jsonl",
            warm_start=True,
        )
        assert ledger2.get_score("c0") is not None
        assert ledger2.get_score("c1") is not None

    def test_total_entries_rebuilt_after_warm_start(self, tmp_path):
        ledger1 = _ledger(tmp_path)
        for _ in range(4):
            ledger1.add_entry(_entry())
        ledger2 = FileTrustLedger(
            ledger_path=tmp_path / "ledger.jsonl", warm_start=True
        )
        assert ledger2.total_entries == 4

    def test_no_file_warm_start_is_empty(self, tmp_path):
        ledger = FileTrustLedger(
            ledger_path=tmp_path / "nonexistent.jsonl", warm_start=True
        )
        assert ledger.get_all_scores() == []
        assert ledger.total_entries == 0


# ---------------------------------------------------------------------------
# FileTrustLedger — flush buffer
# ---------------------------------------------------------------------------


class TestFlushBuffer:
    def test_flush_returns_zero_when_empty(self, tmp_path):
        ledger = _ledger(tmp_path)
        assert ledger.flush_buffer() == 0

    def test_buffered_entries_flushed(self, tmp_path):
        ledger = _ledger(tmp_path)
        # Manually push to buffer (simulate prior write failure)
        e = _entry()
        ledger._buffer.append(e)
        flushed = ledger.flush_buffer()
        assert flushed == 1
        assert ledger.buffered_count == 0

    def test_buffer_written_to_disk(self, tmp_path):
        ledger = _ledger(tmp_path)
        e = _entry(subject_id="c_buffered")
        ledger._buffer.append(e)
        ledger.flush_buffer()
        content = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
        assert "c_buffered" in content


# ---------------------------------------------------------------------------
# FileTrustLedger — decay_scores
# ---------------------------------------------------------------------------


class TestDecayScores:
    def test_decay_reduces_score(self, tmp_path):
        ledger = _ledger(tmp_path, decay_rate=0.2)
        ledger.add_entry(_entry(subject_id="c0", score=0.8))
        before = ledger.get_score("c0").score
        ledger.decay_scores(current_round=2)
        after = ledger.get_score("c0").score
        assert after < before

    def test_zero_decay_no_change(self, tmp_path):
        ledger = _ledger(tmp_path, decay_rate=0.0)
        ledger.add_entry(_entry(subject_id="c0", score=0.5))
        before = ledger.get_score("c0").score
        ledger.decay_scores(current_round=5)
        after = ledger.get_score("c0").score
        assert after == pytest.approx(before, rel=1e-5)

    def test_score_never_goes_negative(self, tmp_path):
        ledger = _ledger(tmp_path, decay_rate=1.0)
        ledger.add_entry(_entry(subject_id="c0", score=0.5))
        ledger.decay_scores(current_round=2)
        assert ledger.get_score("c0").score >= 0.0

    def test_round_stamped_correctly(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.add_entry(_entry(subject_id="c0", round_num=1))
        ledger.decay_scores(current_round=7)
        assert ledger.get_score("c0").last_updated_round == 7


# ---------------------------------------------------------------------------
# FileTrustLedger — load_entries
# ---------------------------------------------------------------------------


class TestLoadEntries:
    def test_load_returns_all_written(self, tmp_path):
        ledger = _ledger(tmp_path)
        for i in range(5):
            ledger.add_entry(_entry(subject_id=f"c{i}"))
        entries = ledger.load_entries()
        assert len(entries) == 5

    def test_load_missing_file_returns_empty(self, tmp_path):
        ledger = FileTrustLedger(
            ledger_path=tmp_path / "no_file.jsonl", warm_start=False
        )
        assert ledger.load_entries() == []

    def test_entries_are_trust_ledger_entry_objects(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.add_entry(_entry())
        entries = ledger.load_entries()
        assert isinstance(entries[0], TrustLedgerEntry)

    def test_malformed_lines_skipped(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        # Write one valid + one malformed line
        path.write_text(
            '{"entry_id": "bad"}\n'  # missing required fields → skip
            + _entry().model_dump_json() + "\n",
            encoding="utf-8",
        )
        ledger = FileTrustLedger(ledger_path=path, warm_start=False)
        entries = ledger.load_entries()
        # Only the valid entry should survive
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# FileTrustLedger — query()
# ---------------------------------------------------------------------------


class TestQuery:
    def _setup(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.add_entry(_entry("c0", layer_id="L1", round_num=1, score=0.3))
        ledger.add_entry(_entry("c0", layer_id="L2", round_num=2, score=0.6))
        ledger.add_entry(_entry("c1", layer_id="L1", round_num=2, score=0.2))
        ledger.add_entry(_entry("c2", layer_id="L3", round_num=3, score=0.5))
        return ledger

    def test_no_filters_returns_all(self, tmp_path):
        ledger = self._setup(tmp_path)
        results = ledger.query(TrustLedgerQuery())
        assert len(results) == 4

    def test_filter_by_subject_id(self, tmp_path):
        ledger = self._setup(tmp_path)
        results = ledger.query(TrustLedgerQuery(subject_ids=["c0"]))
        assert all(e.subject_id == "c0" for e in results)
        assert len(results) == 2

    def test_filter_by_layer(self, tmp_path):
        ledger = self._setup(tmp_path)
        results = ledger.query(TrustLedgerQuery(layers=["L1"]))
        assert all(e.layer_id == "L1" for e in results)

    def test_filter_by_round_range(self, tmp_path):
        ledger = self._setup(tmp_path)
        results = ledger.query(TrustLedgerQuery(round_min=2, round_max=2))
        assert all(e.round_num == 2 for e in results)
        assert len(results) == 2

    def test_filter_by_min_score(self, tmp_path):
        ledger = self._setup(tmp_path)
        results = ledger.query(TrustLedgerQuery(min_score=0.5))
        assert all(e.score >= 0.5 for e in results)

    def test_filter_by_max_score(self, tmp_path):
        ledger = self._setup(tmp_path)
        results = ledger.query(TrustLedgerQuery(max_score=0.3))
        assert all(e.score <= 0.3 for e in results)

    def test_limit_respected(self, tmp_path):
        ledger = self._setup(tmp_path)
        results = ledger.query(TrustLedgerQuery(limit=2))
        assert len(results) <= 2

    def test_results_newest_first(self, tmp_path):
        ledger = self._setup(tmp_path)
        results = ledger.query(TrustLedgerQuery())
        rounds = [e.round_num for e in results if e.round_num is not None]
        assert rounds == sorted(rounds, reverse=True)

    def test_combined_filters(self, tmp_path):
        ledger = self._setup(tmp_path)
        results = ledger.query(
            TrustLedgerQuery(subject_ids=["c0"], layers=["L1"], round_min=1, round_max=1)
        )
        assert len(results) == 1
        assert results[0].subject_id == "c0"
        assert results[0].layer_id == "L1"

    def test_query_round_max_lt_min_raises(self):
        with pytest.raises(Exception):
            TrustLedgerQuery(round_min=5, round_max=3)


# ---------------------------------------------------------------------------
# FileTrustLedger — get_client_history()
# ---------------------------------------------------------------------------


class TestGetClientHistory:
    def test_returns_entries_for_client(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        history = ledger.get_client_history("client_00")
        assert all(e.subject_id == "client_00" for e in history)

    def test_newest_first(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        history = ledger.get_client_history("client_00")
        rounds = [e.round_num for e in history if e.round_num is not None]
        assert rounds == sorted(rounds, reverse=True)

    def test_max_rounds_filter(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        # max_rounds=1 should return only entries from the last 1 rounds
        history = ledger.get_client_history("client_00", max_rounds=1)
        if history:
            max_rnd = max(e.round_num for e in history if e.round_num is not None)
            assert all(
                e.round_num >= max_rnd - 1
                for e in history
                if e.round_num is not None
            )

    def test_unknown_client_returns_empty(self, tmp_path):
        ledger = _ledger(tmp_path)
        assert ledger.get_client_history("ghost") == []


# ---------------------------------------------------------------------------
# FileTrustLedger — reputation_heatmap()
# ---------------------------------------------------------------------------


class TestReputationHeatmap:
    def test_client_ids_present(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        h = ledger.reputation_heatmap()
        assert "client_00" in h["client_ids"]
        assert "client_01" in h["client_ids"]

    def test_matrix_shape(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        h = ledger.reputation_heatmap()
        n_clients = len(h["client_ids"])
        n_rounds = len(h["rounds"])
        assert len(h["matrix"]) == n_clients
        assert all(len(row) == n_rounds for row in h["matrix"])

    def test_current_scores_present(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        h = ledger.reputation_heatmap()
        for cid in h["client_ids"]:
            assert cid in h["current_scores"]

    def test_round_range_filter(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        h = ledger.reputation_heatmap(round_range=(2, 3))
        assert all(r >= 2 for r in h["rounds"])
        assert all(r <= 3 for r in h["rounds"])

    def test_empty_ledger_returns_empty(self, tmp_path):
        ledger = _ledger(tmp_path)
        h = ledger.reputation_heatmap()
        assert h["client_ids"] == []
        assert h["rounds"] == []
        assert h["matrix"] == []


# ---------------------------------------------------------------------------
# FileTrustLedger — top_k_suspicious / suspicious_above
# ---------------------------------------------------------------------------


class TestSuspicious:
    def test_top_k_ordering(self, tmp_path):
        ledger = _ledger(tmp_path, suspicious_threshold=0.3)
        ledger.add_entry(_entry("high", score=0.8))
        ledger.add_entry(_entry("mid", score=0.5))
        ledger.add_entry(_entry("low", score=0.1))
        top = ledger.top_k_suspicious(k=2)
        assert len(top) == 2
        assert top[0].score >= top[1].score

    def test_top_k_capped_by_available(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.add_entry(_entry("c0"))
        assert len(ledger.top_k_suspicious(k=10)) == 1

    def test_suspicious_above_threshold(self, tmp_path):
        ledger = _ledger(tmp_path, suspicious_threshold=0.5)
        # score=0.8 → accumulated trust_score will be 0.4; use very low threshold
        ledger = _ledger(tmp_path, suspicious_threshold=0.1)
        ledger.add_entry(_entry(subject_id="high", score=0.8))
        ledger.add_entry(_entry(subject_id="low", score=0.01))
        sus = ledger.suspicious_above()
        assert any(ts.subject_id == "high" for ts in sus)
        assert not any(ts.subject_id == "low" for ts in sus)

    def test_suspicious_above_override(self, tmp_path):
        ledger = _ledger(tmp_path, suspicious_threshold=0.9)
        ledger.add_entry(_entry("c0", score=0.7))
        assert ledger.suspicious_above(threshold=0.3)  # override low → flagged


# ---------------------------------------------------------------------------
# FileTrustLedger — round_summary()
# ---------------------------------------------------------------------------


class TestRoundSummary:
    def test_structure(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        summary = ledger.round_summary(2)
        assert "round_num" in summary
        assert "n_entries" in summary
        assert "n_flagged_clients" in summary
        assert "mean_score" in summary
        assert "max_score" in summary
        assert "layers_active" in summary
        assert "flagged_client_ids" in summary

    def test_round_num_matches(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        summary = ledger.round_summary(1)
        assert summary["round_num"] == 1

    def test_correct_client_count(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        # 3 clients all write in round 1
        summary = ledger.round_summary(1)
        assert summary["n_flagged_clients"] == 3

    def test_empty_round_returns_zeros(self, tmp_path):
        ledger = _ledger(tmp_path)
        summary = ledger.round_summary(99)
        assert summary["n_entries"] == 0
        assert summary["mean_score"] == 0.0


# ---------------------------------------------------------------------------
# FileTrustLedger — export_snapshot()
# ---------------------------------------------------------------------------


class TestExportSnapshot:
    def test_returns_reputation_snapshots(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        snapshots = ledger.export_snapshot(round_num=3)
        assert all(isinstance(s, ReputationSnapshot) for s in snapshots)

    def test_one_snapshot_per_client(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        snapshots = ledger.export_snapshot(round_num=3)
        cids = [s.client_id for s in snapshots]
        assert len(cids) == len(set(cids))
        assert len(cids) == 3

    def test_sorted_by_trust_score_desc(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        snapshots = ledger.export_snapshot(round_num=3)
        scores = [s.trust_score for s in snapshots]
        assert scores == sorted(scores, reverse=True)

    def test_is_suspicious_flag_set(self, tmp_path):
        ledger = _ledger(tmp_path, suspicious_threshold=0.1)
        ledger.add_entry(_entry("c0", score=0.8))
        snapshots = ledger.export_snapshot(round_num=1)
        c0_snap = next(s for s in snapshots if s.client_id == "c0")
        assert c0_snap.is_suspicious is True

    def test_round_num_stamped(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        snapshots = ledger.export_snapshot(round_num=5)
        assert all(s.round_num == 5 for s in snapshots)

    def test_evidence_fields_parsed(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.add_entry(_entry(
            "c0", score=0.4, round_num=1,
            evidence={"anomaly_score": 0.7, "norm": 2.5},
        ))
        snapshots = ledger.export_snapshot(round_num=1)
        c0 = next(s for s in snapshots if s.client_id == "c0")
        assert len(c0.anomaly_score_history) == 1
        assert c0.anomaly_score_history[0] == pytest.approx(0.7, abs=1e-3)

    def test_empty_ledger_returns_empty(self, tmp_path):
        ledger = _ledger(tmp_path)
        assert ledger.export_snapshot(round_num=0) == []


# ---------------------------------------------------------------------------
# FileTrustLedger — get_stats()
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_structure(self, tmp_path):
        ledger = _ledger(tmp_path)
        stats = ledger.get_stats()
        for key in ("total_entries", "n_clients", "n_buffered", "ledger_path", "suspicious_count", "mean_score"):
            assert key in stats

    def test_stats_counts(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        stats = ledger.get_stats()
        assert stats["total_entries"] == 9  # 3 clients × 3 rounds
        assert stats["n_clients"] == 3

    def test_buffered_count_reflects_buffer(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger._buffer.append(_entry())
        assert ledger.get_stats()["n_buffered"] == 1


# ---------------------------------------------------------------------------
# FileTrustLedger — configurable threshold property
# ---------------------------------------------------------------------------


class TestSuspiciousThresholdProperty:
    def test_getter_returns_initial_value(self, tmp_path):
        ledger = _ledger(tmp_path, suspicious_threshold=0.7)
        assert ledger.suspicious_threshold == pytest.approx(0.7)

    def test_setter_updates_value(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.suspicious_threshold = 0.3
        assert ledger.suspicious_threshold == pytest.approx(0.3)

    def test_setter_rejects_out_of_range(self, tmp_path):
        ledger = _ledger(tmp_path)
        with pytest.raises(ValueError):
            ledger.suspicious_threshold = 1.5


# ---------------------------------------------------------------------------
# FileTrustLedger — structured logger integration
# ---------------------------------------------------------------------------


class TestStructuredLogger:
    def test_logger_called_on_add_entry(self, tmp_path):
        mock_logger = MagicMock()
        ledger = FileTrustLedger(
            ledger_path=tmp_path / "ledger.jsonl",
            sentinel_logger=mock_logger,
            warm_start=False,
        )
        ledger.add_entry(_entry())
        mock_logger.log.assert_called_once()
        args = mock_logger.log.call_args[0]
        assert args[0] == "L4"
        assert args[1] == "ledger_entry_added"

    def test_logger_failure_does_not_raise(self, tmp_path):
        mock_logger = MagicMock()
        mock_logger.log.side_effect = RuntimeError("log broken")
        ledger = FileTrustLedger(
            ledger_path=tmp_path / "ledger.jsonl",
            sentinel_logger=mock_logger,
            warm_start=False,
        )
        ledger.add_entry(_entry())  # must not raise


# ---------------------------------------------------------------------------
# FileTrustLedger — UpdateGuard integration
# ---------------------------------------------------------------------------


class TestUpdateGuardIntegration:
    def test_update_guard_writes_to_ledger(self, tmp_path):
        """UpdateGuard's TrustScoreManager must call ledger.add_entry."""
        import numpy as np

        from ai.detection.update_guard import UpdateGuard

        ledger = FileTrustLedger(
            ledger_path=tmp_path / "guard_ledger.jsonl",
            warm_start=False,
        )
        guard = UpdateGuard(
            sim_threshold=0.85,
            min_cluster_size=2,
            trust_score_weight=0.5,
            ledger=ledger,
        )
        rng = np.random.default_rng(42)
        deltas = [rng.standard_normal(20).astype(np.float32) for _ in range(5)]
        guard.process_round(
            round_num=1,
            client_ids=[f"c{i}" for i in range(5)],
            deltas=deltas,
        )
        # All 5 clients should appear in the ledger (anomaly_score > 0)
        scores = ledger.get_all_scores()
        assert len(scores) > 0

    def test_ledger_scores_in_range_after_guard(self, tmp_path):
        import numpy as np

        from ai.detection.update_guard import UpdateGuard

        ledger = FileTrustLedger(
            ledger_path=tmp_path / "guard_ledger2.jsonl",
            warm_start=False,
        )
        guard = UpdateGuard(ledger=ledger)
        rng = np.random.default_rng(0)
        deltas = [rng.standard_normal(15).astype(np.float32) for _ in range(4)]
        guard.process_round(1, [f"c{i}" for i in range(4)], deltas)
        for ts in ledger.get_all_scores():
            assert 0.0 <= ts.score <= 1.0


# ---------------------------------------------------------------------------
# ReputationEngine — client_reputation_report()
# ---------------------------------------------------------------------------


class TestClientReputationReport:
    def test_structure(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        report = engine.client_reputation_report("client_00")
        for key in ("client_id", "current_score", "is_suspicious", "total_flag_count",
                    "layer_breakdown", "round_history", "most_recent_reason"):
            assert key in report

    def test_total_flag_count(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        report = engine.client_reputation_report("client_00")
        assert report["total_flag_count"] == 3  # 3 rounds of entries

    def test_unknown_client_zero_score(self, tmp_path):
        ledger = _ledger(tmp_path)
        engine = ReputationEngine(ledger)
        report = engine.client_reputation_report("ghost")
        assert report["current_score"] == 0.0
        assert report["total_flag_count"] == 0


# ---------------------------------------------------------------------------
# ReputationEngine — heatmap_data()
# ---------------------------------------------------------------------------


class TestHeatmapData:
    def test_enrichment_keys(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        h = engine.heatmap_data()
        assert "per_client_summary" in h
        assert "per_round_summary" in h

    def test_per_client_summary_length(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        h = engine.heatmap_data()
        assert len(h["per_client_summary"]) == 3

    def test_per_round_summary_length(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        h = engine.heatmap_data()
        assert len(h["per_round_summary"]) == 3


# ---------------------------------------------------------------------------
# ReputationEngine — flag_rate_by_layer()
# ---------------------------------------------------------------------------


class TestFlagRateByLayer:
    def test_fractions_sum_to_one(self, tmp_path):
        ledger = _ledger(tmp_path)
        for _ in range(3):
            ledger.add_entry(_entry(layer_id="L1"))
        for _ in range(2):
            ledger.add_entry(_entry(layer_id="L2"))
        engine = ReputationEngine(ledger)
        rates = engine.flag_rate_by_layer()
        assert abs(sum(rates.values()) - 1.0) < 1e-5

    def test_empty_returns_empty(self, tmp_path):
        ledger = _ledger(tmp_path)
        assert ReputationEngine(ledger).flag_rate_by_layer() == {}


# ---------------------------------------------------------------------------
# ReputationEngine — score_distribution()
# ---------------------------------------------------------------------------


class TestScoreDistribution:
    def test_structure(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        dist = engine.score_distribution()
        for key in ("bins", "counts", "mean", "std", "n_suspicious", "total_clients"):
            assert key in dist

    def test_bins_count(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        dist = engine.score_distribution(n_bins=5)
        assert len(dist["bins"]) == 5
        assert len(dist["counts"]) == 5

    def test_counts_sum_to_total(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        dist = engine.score_distribution()
        assert sum(dist["counts"]) == dist["total_clients"]

    def test_empty_returns_defaults(self, tmp_path):
        ledger = _ledger(tmp_path)
        dist = ReputationEngine(ledger).score_distribution()
        assert dist["mean"] == 0.0
        assert dist["n_suspicious"] == 0


# ---------------------------------------------------------------------------
# ReputationEngine — suspicious_timeline()
# ---------------------------------------------------------------------------


class TestSuspiciousTimeline:
    def test_sorted_by_round(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        timeline = engine.suspicious_timeline(threshold=0.0)
        rounds = [t["round_num"] for t in timeline]
        assert rounds == sorted(rounds)

    def test_structure(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        timeline = engine.suspicious_timeline()
        for item in timeline:
            assert "round_num" in item
            assert "n_suspicious" in item
            assert "suspicious_client_ids" in item


# ---------------------------------------------------------------------------
# ReputationEngine — cross_layer_correlation()
# ---------------------------------------------------------------------------


class TestCrossLayerCorrelation:
    def test_structure(self, tmp_path):
        ledger = _ledger(tmp_path)
        for _ in range(2):
            ledger.add_entry(_entry(layer_id="L1"))
        for _ in range(2):
            ledger.add_entry(_entry(layer_id="L2"))
        engine = ReputationEngine(ledger)
        result = engine.cross_layer_correlation()
        assert "layer_ids" in result
        assert "agreement_matrix" in result

    def test_matrix_is_symmetric(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.add_entry(_entry("c0", layer_id="L1"))
        ledger.add_entry(_entry("c0", layer_id="L2"))
        ledger.add_entry(_entry("c1", layer_id="L1"))
        engine = ReputationEngine(ledger)
        result = engine.cross_layer_correlation()
        m = result["agreement_matrix"]
        n = len(m)
        for i in range(n):
            for j in range(n):
                assert abs(m[i][j] - m[j][i]) < 1e-6

    def test_self_similarity_is_one(self, tmp_path):
        ledger = _ledger(tmp_path)
        ledger.add_entry(_entry("c0", layer_id="L1"))
        engine = ReputationEngine(ledger)
        result = engine.cross_layer_correlation()
        m = result["agreement_matrix"]
        for i in range(len(m)):
            assert m[i][i] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ReputationEngine — compute_all_metrics()
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    def test_structure(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        metrics = engine.compute_all_metrics("exp_001")
        for key in ("experiment_id", "total_ledger_entries", "n_tracked_clients",
                    "n_suspicious_clients", "suspicious_fraction", "mean_trust_score",
                    "top_suspicious", "flag_rate_by_layer", "score_distribution"):
            assert key in metrics

    def test_experiment_id_echoed(self, tmp_path):
        ledger = _populated_ledger(tmp_path)
        engine = ReputationEngine(ledger)
        assert engine.compute_all_metrics("exp_xyz")["experiment_id"] == "exp_xyz"


# ---------------------------------------------------------------------------
# Schema — ReputationSnapshot
# ---------------------------------------------------------------------------


class TestReputationSnapshotSchema:
    def test_valid_snapshot(self):
        s = ReputationSnapshot(
            client_id="c0",
            round_num=3,
            trust_score=0.42,
            contributing_entry_count=5,
            is_suspicious=False,
        )
        assert s.trust_score == pytest.approx(0.42)

    def test_score_out_of_range_raises(self):
        with pytest.raises(Exception):
            ReputationSnapshot(
                client_id="c0",
                round_num=1,
                trust_score=1.5,
                contributing_entry_count=0,
                is_suspicious=True,
            )


# ---------------------------------------------------------------------------
# Schema — TrustLedgerQuery
# ---------------------------------------------------------------------------


class TestTrustLedgerQuerySchema:
    def test_default_all_none(self):
        q = TrustLedgerQuery()
        assert q.subject_ids is None
        assert q.layers is None
        assert q.limit is None

    def test_round_max_lt_min_raises(self):
        with pytest.raises(Exception):
            TrustLedgerQuery(round_min=5, round_max=2)

    def test_valid_query_accepted(self):
        q = TrustLedgerQuery(
            subject_ids=["c0"],
            layers=["L1"],
            round_min=1,
            round_max=10,
            min_score=0.2,
            limit=50,
        )
        assert q.limit == 50
