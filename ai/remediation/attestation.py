"""
ai/remediation/attestation.py — Tamper-evident Remediation Attestation Certificates.

Detection and even repair are not enough for a *trustworthy* federated system: an
operator (or a competition judge, or an auditor) needs cryptographic proof that a model
that was found to be backdoored was actually remediated, by which strategy, and with what
measured effect — and that the record has not been tampered with after the fact.

This module issues **Remediation Attestation Certificates**: signed, hash-chained records
that bind together

  * the fingerprints (SHA-256) of the model *before* and *after* remediation,
  * the measured attack-success-rate and clean-accuracy before/after,
  * the winning strategy and the acceptance decision,
  * the hash of the previous certificate (forming an append-only chain à la a mini
    transparency log), and
  * an optional HMAC-SHA-256 signature under an operator secret.

Everything here is pure standard library (``hashlib`` / ``hmac`` / ``json``) plus NumPy for
model fingerprints, so it adds zero dependencies and runs in any Phase-0 install.

The chain is verifiable end-to-end with :func:`verify_certificate` and
:meth:`AttestationLedger.verify_chain`, so anyone can independently confirm that the
sequence of remediations is authentic and unbroken.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Genesis hash for the head of an attestation chain (all-zero, like a blockchain genesis).
GENESIS_HASH = "0" * 64


def model_fingerprint(params: np.ndarray) -> str:
    """Return a stable SHA-256 fingerprint of a model parameter vector.

    Uses the raw little-endian float64 bytes of a C-contiguous copy so the digest is
    reproducible across machines for the same weights.
    """
    arr = np.ascontiguousarray(np.asarray(params, dtype=np.float64))
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic JSON encoding (sorted keys, compact separators) for hashing."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass
class RemediationCertificate:
    """A single tamper-evident attestation of one remediation event.

    The ``content_hash`` is the SHA-256 of the canonical JSON of every field except
    ``content_hash`` and ``signature``. ``signature`` is an optional HMAC-SHA-256 of the
    ``content_hash`` under an operator secret. ``prev_hash`` links to the prior
    certificate's ``content_hash`` to form an append-only chain.
    """

    certificate_id: str
    issued_at: str
    remediation_id: str
    round_num: int | None
    strategy_succeeded: str | None
    success: bool
    manual_review_required: bool
    asr_before: float
    asr_after: float
    clean_accuracy_before: float
    clean_accuracy_after: float
    asr_threshold: float
    model_before_sha256: str
    model_after_sha256: str
    strategies_attempted: list[str] = field(default_factory=list)
    prev_hash: str = GENESIS_HASH
    content_hash: str = ""
    signature: str | None = None

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    @property
    def asr_reduction(self) -> float:
        """Absolute drop in attack-success-rate achieved by the remediation."""
        return round(self.asr_before - self.asr_after, 6)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def _hashable_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("content_hash", None)
        payload.pop("signature", None)
        return payload

    def compute_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self._hashable_payload()).encode("utf-8")).hexdigest()


def issue_certificate(
    report: Any,
    params_before: np.ndarray,
    params_after: np.ndarray,
    *,
    prev_hash: str = GENESIS_HASH,
    secret_key: bytes | str | None = None,
) -> RemediationCertificate:
    """Mint a :class:`RemediationCertificate` from a ``RemediationReport``.

    Args:
        report: A ``RemediationReport`` (or any object exposing the same attributes).
        params_before: Model parameter vector prior to remediation.
        params_after: Repaired model parameter vector.
        prev_hash: ``content_hash`` of the previous certificate in the chain.
        secret_key: Optional operator secret; when provided the certificate is signed
            with HMAC-SHA-256 over its ``content_hash``.

    Returns:
        A fully populated, hashed, and (optionally) signed certificate.
    """
    remediation_id = str(getattr(report, "remediation_id", "unknown"))
    cert = RemediationCertificate(
        certificate_id=f"cert-{remediation_id}",
        issued_at=datetime.now(UTC).isoformat(),
        remediation_id=remediation_id,
        round_num=getattr(report, "round_num", None),
        strategy_succeeded=getattr(report, "strategy_succeeded", None),
        success=bool(getattr(report, "success", False)),
        manual_review_required=bool(getattr(report, "manual_review_required", False)),
        asr_before=float(getattr(report, "asr_before", 0.0)),
        asr_after=float(getattr(report, "asr_after", 0.0)),
        clean_accuracy_before=float(getattr(report, "clean_accuracy_before", 0.0)),
        clean_accuracy_after=float(getattr(report, "clean_accuracy_after", 0.0)),
        asr_threshold=float(getattr(report, "asr_threshold", 0.0)),
        model_before_sha256=model_fingerprint(params_before),
        model_after_sha256=model_fingerprint(params_after),
        strategies_attempted=list(getattr(report, "strategies_attempted", []) or []),
        prev_hash=prev_hash,
    )
    cert.content_hash = cert.compute_hash()
    if secret_key is not None:
        cert.signature = _sign(cert.content_hash, secret_key)
    return cert


def _sign(content_hash: str, secret_key: bytes | str) -> str:
    key = secret_key.encode("utf-8") if isinstance(secret_key, str) else secret_key
    return hmac.new(key, content_hash.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_certificate(
    cert: RemediationCertificate | dict[str, Any],
    *,
    secret_key: bytes | str | None = None,
) -> bool:
    """Verify a certificate's content hash and (if a key is given) its signature.

    Returns ``True`` only if the recomputed content hash matches the stored one and,
    when ``secret_key`` is provided, the HMAC signature validates in constant time.
    """
    if isinstance(cert, dict):
        stored_hash = cert.get("content_hash", "")
        stored_sig = cert.get("signature")
        cert = RemediationCertificate(**cert)
    else:
        stored_hash = cert.content_hash
        stored_sig = cert.signature

    if cert.compute_hash() != stored_hash:
        return False
    if secret_key is not None:
        expected = _sign(stored_hash, secret_key)
        if stored_sig is None or not hmac.compare_digest(expected, stored_sig):
            return False
    return True


class AttestationLedger:
    """Append-only, hash-chained ledger of remediation certificates (JSONL on disk).

    Each appended certificate's ``prev_hash`` is set to the ``content_hash`` of the
    previous one, so any post-hoc edit or deletion breaks :meth:`verify_chain`.
    """

    def __init__(self, path: str | Path, secret_key: bytes | str | None = None) -> None:
        self._path = Path(path)
        self._secret_key = secret_key

    def _read_raw(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def head_hash(self) -> str:
        """Return the ``content_hash`` of the most recent certificate, or genesis."""
        records = self._read_raw()
        return records[-1]["content_hash"] if records else GENESIS_HASH

    def append(
        self,
        report: Any,
        params_before: np.ndarray,
        params_after: np.ndarray,
    ) -> RemediationCertificate:
        """Issue a certificate chained onto the current head and persist it."""
        cert = issue_certificate(
            report,
            params_before,
            params_after,
            prev_hash=self.head_hash(),
            secret_key=self._secret_key,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(_canonical_json(cert.to_dict()) + "\n")
        return cert

    def verify_chain(self) -> bool:
        """Verify every certificate and the integrity of the hash chain linking them."""
        prev = GENESIS_HASH
        for raw in self._read_raw():
            if raw.get("prev_hash") != prev:
                return False
            if not verify_certificate(raw, secret_key=self._secret_key):
                return False
            prev = raw["content_hash"]
        return True

    def __len__(self) -> int:
        return len(self._read_raw())
