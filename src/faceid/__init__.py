"""Verificación facial con consentimiento para personal autorizado (issue #10)."""

from __future__ import annotations

from .verify import (
    EnrollmentStore,
    VerificationResult,
    Verifier,
    VerifyPolicy,
    VerifyStatus,
)

__all__ = [
    "EnrollmentStore",
    "VerificationResult",
    "Verifier",
    "VerifyPolicy",
    "VerifyStatus",
]
