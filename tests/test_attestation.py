"""
tests/test_attestation.py — Tests for the Remediation Attestation Certificate chain.

Pure numpy + stdlib (no torch/flwr), so this module runs in a Phase-0 install.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ai.remediation.attestation import (
    GENESIS_HASH,
    AttestationLedger,
    issue_certificate,
    model_fingerprint,
    verify_certificate,
)


def _report(**overrides):
    base = dict(
        remediation_id="rem-001",
        round_num=8,
        strategy_succeeded="unlearning",
        success=True,
        manual_review_required=False,
        asr_before=1.0,
        asr_after=0.25,
        clean_accuracy_before=1.0,
        clean_accuracy_after=0.99,
        asr_threshold=0.2,
        strategies_attempted=["rollback", "unlearning"],
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def params():
    rng = np.random.default_rng(0)
    return rng.normal(size=200), rng.normal(size=200)


class TestFingerprint:
    def test_fingerprint_is_deterministic(self, params):
        before, _ = params
        assert model_fingerprint(before) == model_fingerprint(before.copy())

    def test_fingerprint_changes_with_weights(self, params):
        before, after = params
        assert model_fingerprint(before) != model_fingerprint(after)

    def test_fingerprint_is_hex_sha256(self, params):
        before, _ = params
        fp = model_fingerprint(before)
        assert len(fp) == 64
        int(fp, 16)  # raises if not valid hex


class TestCertificate:
    def test_issue_and_verify(self, params):
        before, after = params
        cert = issue_certificate(_report(), before, after, secret_key="k")
        assert cert.success is True
        assert cert.asr_reduction == pytest.approx(0.75)
        assert cert.prev_hash == GENESIS_HASH
        assert verify_certificate(cert, secret_key="k") is True

    def test_tamper_breaks_hash(self, params):
        before, after = params
        cert = issue_certificate(_report(), before, after, secret_key="k")
        cert.asr_after = 0.0
        assert verify_certificate(cert, secret_key="k") is False

    def test_wrong_key_fails_signature(self, params):
        before, after = params
        cert = issue_certificate(_report(), before, after, secret_key="k")
        assert verify_certificate(cert, secret_key="wrong") is False

    def test_unsigned_certificate_verifies_hash_only(self, params):
        before, after = params
        cert = issue_certificate(_report(), before, after)
        assert cert.signature is None
        assert verify_certificate(cert) is True

    def test_dict_roundtrip_verifies(self, params):
        before, after = params
        cert = issue_certificate(_report(), before, after, secret_key="k")
        assert verify_certificate(cert.to_dict(), secret_key="k") is True


class TestLedger:
    def test_append_and_verify_chain(self, tmp_path, params):
        before, after = params
        ledger = AttestationLedger(tmp_path / "chain.jsonl", secret_key="k")
        c1 = ledger.append(_report(remediation_id="a"), before, after)
        c2 = ledger.append(_report(remediation_id="b"), after, before)
        assert len(ledger) == 2
        assert c1.prev_hash == GENESIS_HASH
        assert c2.prev_hash == c1.content_hash
        assert ledger.verify_chain() is True

    def test_tampered_file_breaks_chain(self, tmp_path, params):
        before, after = params
        path = tmp_path / "chain.jsonl"
        ledger = AttestationLedger(path, secret_key="k")
        ledger.append(_report(remediation_id="a"), before, after)
        ledger.append(_report(remediation_id="b"), after, before)
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = lines[0].replace("unlearning", "rollback")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        assert ledger.verify_chain() is False

    def test_empty_ledger_is_valid(self, tmp_path):
        ledger = AttestationLedger(tmp_path / "chain.jsonl")
        assert len(ledger) == 0
        assert ledger.verify_chain() is True
        assert ledger.head_hash() == GENESIS_HASH
