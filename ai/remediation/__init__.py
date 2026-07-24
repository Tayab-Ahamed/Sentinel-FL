"""
ai/remediation/ — the Remediation Engine (ARCHITECTURE.md §7.4).

Status: IMPLEMENTED. Turns a confirmed backdoor detection into an auditable,
escalating recovery: rollback → targeted unlearning → fine-pruning, with a
``manual_review_required`` escalation when every path is exhausted.

Public surface::

    from ai.remediation import (
        RemediationEngine,
        RollbackRemediator,
        TriggerUnlearner,
        FinePruner,
        LinearSoftmaxAdapter,
    )
"""

from ai.remediation.adapters import LinearSoftmaxAdapter, ModelAdapter
from ai.remediation.attestation import (
    AttestationLedger,
    RemediationCertificate,
    issue_certificate,
    model_fingerprint,
    verify_certificate,
)
from ai.remediation.pruning import FinePruner
from ai.remediation.remediation_engine import RemediationEngine
from ai.remediation.rollback import RollbackFailed, RollbackRemediator
from ai.remediation.triggers import (
    as_trigger_vector,
    stamp_trigger,
    trigger_from_block,
    trigger_mask,
)
from ai.remediation.unlearning import TriggerUnlearner

__all__ = [
    "RemediationEngine",
    "RollbackRemediator",
    "RollbackFailed",
    "TriggerUnlearner",
    "FinePruner",
    "LinearSoftmaxAdapter",
    "ModelAdapter",
    "AttestationLedger",
    "RemediationCertificate",
    "issue_certificate",
    "verify_certificate",
    "model_fingerprint",
    "as_trigger_vector",
    "stamp_trigger",
    "trigger_from_block",
    "trigger_mask",
]
