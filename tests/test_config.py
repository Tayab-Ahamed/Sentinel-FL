"""
tests/test_config.py — Unit tests for the config loader (TESTING.md §2).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.fl_core.config import load_config, load_config_from_dict
from ai.fl_core.exceptions import ConfigValidationError
from ai.fl_core.schemas import Configuration

# ---------------------------------------------------------------------------
# load_config_from_dict
# ---------------------------------------------------------------------------


class TestLoadConfigFromDict:
    def test_minimal_valid(self):
        cfg = load_config_from_dict({"n_clients": 6, "n_rounds": 5, "krum_select": 4})
        assert cfg.n_clients == 6
        assert cfg.n_rounds == 5

    def test_defaults_applied(self):
        cfg = load_config_from_dict({})
        assert cfg.n_clients == 12  # default from schema
        assert cfg.seed == 42

    def test_invalid_field_raises_config_validation_error(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config_from_dict({"n_clients": -1})
        assert "n_clients" in exc_info.value.field_errors

    def test_krum_select_exceeds_n_clients_raises(self):
        with pytest.raises(ConfigValidationError) as exc_info:
            load_config_from_dict({"n_clients": 4, "krum_select": 10})
        assert "krum_select" in exc_info.value.field_errors


# ---------------------------------------------------------------------------
# load_config (from YAML file)
# ---------------------------------------------------------------------------


class TestLoadConfigFromFile:
    def test_load_default_yaml(self):
        """The committed default.yaml must load and validate without error."""
        default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        if not default_path.exists():
            pytest.skip("configs/default.yaml not found")
        cfg = load_config(default_path)
        assert isinstance(cfg, Configuration)
        assert cfg.n_clients > 0

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_malformed_yaml_raises_config_validation_error(self, tmp_path: Path):
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("just_a_scalar_not_a_mapping\n", encoding="utf-8")
        with pytest.raises(ConfigValidationError):
            load_config(bad_file)


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvVarOverrides:
    def test_sentinel_n_clients_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SENTINEL_N_CLIENTS", "30")
        cfg = load_config_from_dict({"krum_select": 25})
        assert cfg.n_clients == 30

    def test_sentinel_seed_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SENTINEL_SEED", "99")
        cfg = load_config_from_dict({})
        assert cfg.seed == 99

    def test_invalid_env_var_type_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SENTINEL_N_CLIENTS", "not_an_int")
        with pytest.raises(ConfigValidationError):
            load_config_from_dict({})

    def test_env_var_cleaned_up_after_test(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("SENTINEL_N_CLIENTS", "7")
        cfg = load_config_from_dict({"krum_select": 5})
        assert cfg.n_clients == 7
        # After monkeypatch teardown (automatic), original value is restored
