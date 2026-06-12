"""
config/security.py
==================
Centralized security primitives for the trading platform.

Responsibilities
----------------
1. Password hashing & verification (bcrypt, per-password random salt)
2. Cryptographically signed session tokens (HMAC-SHA256, constant-time compare)
3. Strict input sanitization for user-supplied tickers (anti-injection)
4. Secrets loading + masking so raw keys can never reach logs or the UI

Design rules
------------
- No secret value is ever returned to a caller that renders output.
- All comparisons of secret material use hmac.compare_digest (timing-safe).
- Validation is allowlist-based (define what IS valid), never blocklist-based.
"""

from __future__ import annotations

import hmac
import hashlib
import json
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass
from typing import Final, Optional

import bcrypt
from dotenv import load_dotenv

logger = logging.getLogger("security")

# ---------------------------------------------------------------------------
# 0. Environment bootstrap
# ---------------------------------------------------------------------------

REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "SESSION_SIGNING_KEY",   # 64+ hex chars; generate with: python -c "import secrets;print(secrets.token_hex(32))"
)
# Optional: features degrade gracefully when absent (committee abstains without
# ANTHROPIC_API_KEY; macro agent uses demo series without FRED_API_KEY).
OPTIONAL_KEYS: Final[tuple[str, ...]] = (
    "ANTHROPIC_API_KEY", "ALPHAVANTAGE_API_KEY", "FRED_API_KEY",
    "BINANCE_API_KEY", "BINANCE_API_SECRET",
)


class SecurityConfigError(RuntimeError):
    """Raised when the environment is unsafe to start (missing/weak secrets)."""


def load_environment(dotenv_path: str = ".env") -> None:
    """Load .env (never committed; see .gitignore) and fail fast if misconfigured."""
    load_dotenv(dotenv_path=dotenv_path, override=False)

    missing = [k for k in REQUIRED_KEYS if not os.getenv(k)]
    if missing:
        # Log key NAMES only — never values.
        raise SecurityConfigError(
            f"Refusing to start. Missing required secrets: {', '.join(missing)}. "
            "Provide them via environment variables or a .env file."
        )

    signing_key = os.getenv("SESSION_SIGNING_KEY", "")
    if len(signing_key) < 32:
        raise SecurityConfigError(
            "SESSION_SIGNING_KEY is too short (<32 chars). "
            'Generate one with: python -c "import secrets; print(secrets.token_hex(32))"'
        )


def get_secret(name: str) -> str:
    """Fetch a secret for internal use. Callers must never log/render the result."""
    value = os.getenv(name)
    if not value:
        raise SecurityConfigError(f"Secret '{name}' requested but not configured.")
    return value


# ---------------------------------------------------------------------------
# 1. Secret masking — the only representation allowed in logs / UI
# ---------------------------------------------------------------------------

def mask_secret(value: str, visible: int = 4) -> str:
    """Return e.g. 'sk-a***************f3c1'. Safe for logs and dashboards."""
    if not value:
        return "<empty>"
    if len(value) <= visible * 2:
        return "*" * len(value)
    return f"{value[:visible]}{'*' * (len(value) - visible * 2)}{value[-visible:]}"


_SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9\-_]{8,}|[A-Fa-f0-9]{32,}|[A-Za-z0-9+/=]{40,})"
)


def scrub_for_logging(text: str) -> str:
    """Defense-in-depth: redact anything that *looks* like a credential
    before a string is allowed into a log record or the UI."""
    return _SECRET_PATTERN.sub(lambda m: mask_secret(m.group(0)), text)


class RedactingFilter(logging.Filter):
    """Attach to every handler so no raw secret can leak through logging."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = scrub_for_logging(str(record.msg))
        if record.args:
            record.args = tuple(scrub_for_logging(str(a)) for a in record.args)
        return True


def assert_logs_are_masked() -> dict[str, str]:
    """Validation layer required by ops: prove every configured key renders masked.
    Returns {KEY_NAME: masked_preview} — safe to display on an admin panel."""
    report: dict[str, str] = {}
    for key in (*REQUIRED_KEYS, *OPTIONAL_KEYS):
        raw = os.getenv(key)
        if raw is None:
            continue
        masked = mask_secret(raw)
        if raw in masked and len(raw) > 8:  # masking failed somehow
            raise SecurityConfigError(f"Masking invariant violated for {key}.")
        report[key] = masked
    return report


# ---------------------------------------------------------------------------
# 2. Password hashing (bcrypt — per-hash random salt, work factor 12)
# ---------------------------------------------------------------------------

BCRYPT_ROUNDS: Final[int] = 12


def hash_password(plain: str) -> str:
    if len(plain) < 10:
        raise ValueError("Password must be at least 10 characters.")
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False  # malformed hash — treat as failed auth, never raise to user


# ---------------------------------------------------------------------------
# 3. Signed session tokens (stateless HMAC; ready to swap for Redis sessions)
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS: Final[int] = 60 * 60 * 8  # one trading day


@dataclass(frozen=True)
class SessionClaims:
    username: str
    role: str          # 'admin' | 'trader' | 'viewer'  (RBAC)
    issued_at: float
    nonce: str


def _sign(payload: bytes) -> str:
    key = get_secret("SESSION_SIGNING_KEY").encode()
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def issue_session_token(username: str, role: str) -> str:
    claims = SessionClaims(username=username, role=role,
                           issued_at=time.time(), nonce=secrets.token_hex(8))
    payload = json.dumps(claims.__dict__, separators=(",", ":")).encode()
    return payload.hex() + "." + _sign(payload)


def verify_session_token(token: str) -> Optional[SessionClaims]:
    """Return claims if the token is authentic and unexpired, else None."""
    try:
        payload_hex, signature = token.split(".", 1)
        payload = bytes.fromhex(payload_hex)
    except (ValueError, AttributeError):
        return None
    if not hmac.compare_digest(_sign(payload), signature):
        logger.warning("Session token signature mismatch (possible tampering).")
        return None
    claims = SessionClaims(**json.loads(payload))
    if time.time() - claims.issued_at > SESSION_TTL_SECONDS:
        return None
    return claims


# ---------------------------------------------------------------------------
# 4. RBAC
# ---------------------------------------------------------------------------

ROLE_PERMISSIONS: Final[dict[str, frozenset[str]]] = {
    "admin":  frozenset({"view_dashboard", "add_ticker", "run_committee", "view_secrets_report", "manage_users"}),
    "trader": frozenset({"view_dashboard", "add_ticker", "run_committee"}),
    "viewer": frozenset({"view_dashboard"}),
}


def authorize(claims: Optional[SessionClaims], permission: str) -> bool:
    if claims is None:
        return False
    return permission in ROLE_PERMISSIONS.get(claims.role, frozenset())


# ---------------------------------------------------------------------------
# 5. Input sanitization — allowlist ticker validation (anti-injection)
# ---------------------------------------------------------------------------

# Accepts: AAPL, TSLA, BRK.B, BTC-USD, ES=F, ^GSPC, EURUSD=X, CL=F
_TICKER_RE: Final[re.Pattern[str]] = re.compile(
    r"^\^?[A-Z0-9]{1,8}([.\-=][A-Z0-9]{1,6})?$"
)
MAX_WATCHLIST_SIZE: Final[int] = 50


class InvalidTickerError(ValueError):
    pass


def sanitize_ticker(raw: str) -> str:
    """Normalize and validate a user-supplied ticker.
    Anything that doesn't match the strict allowlist pattern is rejected —
    this is what neutralizes SQL/command/path-traversal injection attempts."""
    if not isinstance(raw, str):
        raise InvalidTickerError("Ticker must be a string.")
    candidate = raw.strip().upper()
    if len(candidate) == 0 or len(candidate) > 15:
        raise InvalidTickerError("Ticker length out of bounds.")
    if not _TICKER_RE.fullmatch(candidate):
        # Never echo raw attacker input back into HTML/logs unescaped.
        logger.warning("Rejected invalid ticker input: %r", candidate[:32])
        raise InvalidTickerError(
            "Invalid ticker format. Allowed: letters/digits with optional "
            "single '.', '-', '=' separator (e.g. AAPL, BTC-USD, BRK.B, CL=F)."
        )
    return candidate
