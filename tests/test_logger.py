"""
tests/test_logger.py — Unit tests for StructuredLogger (TESTING.md §2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai.fl_core.logger import StructuredLogger, make_logger


class TestStructuredLogger:
    def test_stdout_logger_does_not_raise(self, capsys: pytest.CaptureFixture):
        logger = StructuredLogger(sink="stdout", log_level="INFO")
        logger.log("L1", "test_event", {"key": "value"})
        captured = capsys.readouterr()
        assert "test_event" in captured.out

    def test_output_is_valid_json(self, capsys: pytest.CaptureFixture):
        logger = StructuredLogger(sink="stdout")
        logger.log("L2", "cluster_flagged", {"cluster": [0, 1]})
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["layer_id"] == "L2"
        assert parsed["event_type"] == "cluster_flagged"
        assert "timestamp" in parsed

    def test_log_entry_includes_round_num(self, capsys: pytest.CaptureFixture):
        logger = StructuredLogger(sink="stdout")
        logger.set_round(7)
        logger.log("L3", "input_flagged", {})
        out = capsys.readouterr().out.strip()
        parsed = json.loads(out)
        assert parsed["round_num"] == 7

    def test_file_sink_writes_jsonl(self, tmp_path: Path):
        log_file = tmp_path / "test.jsonl"
        with StructuredLogger(sink=str(log_file)) as logger:
            logger.log("L1", "event_one", {"a": 1})
            logger.log("L1", "event_two", {"b": 2})

        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["event_type"] == "event_one"
        assert json.loads(lines[1])["event_type"] == "event_two"

    def test_log_never_raises_on_bad_payload(self, capsys: pytest.CaptureFixture):
        """log() must never raise even with unserializable payload values."""
        logger = StructuredLogger(sink="stdout")
        # Pass an object that isn't JSON-serializable via default=str
        logger.log("L4", "edge_case", {"obj": object()})  # should not raise

    def test_invalid_log_level_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid log_level"):
            StructuredLogger(sink="stdout", log_level="VERBOSE")

    def test_make_logger_factory(self):
        logger = make_logger(log_level="WARNING", log_sink="stdout")
        assert isinstance(logger, StructuredLogger)


class TestDroppedCount:
    def test_dropped_count_starts_at_zero(self):
        logger = StructuredLogger(sink="stdout")
        assert logger.dropped_count == 0
