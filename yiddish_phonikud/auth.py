"""Hugging Face OAuth identity for the ABE101-only lexicon editor.

On a Space, ``hf_oauth: true`` in the README YAML provisions OAuth and
``attach_huggingface_oauth`` adds login/logout/callback routes. Identity is the
logged-in Hugging Face username — not a shared password, not UI hiding.

The allowed editors are ``LEXICON_EDITOR_USER`` (default ``ABE101``), which
takes one username or a comma-separated list so a reviewer can be added without
a code change. Set it on the Space; do not fork the check into a client-side
flag.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

EDITOR_USER_ENV = "LEXICON_EDITOR_USER"
DEFAULT_EDITOR_USER = "ABE101"


def editor_usernames() -> tuple[str, ...]:
    """Every Hugging Face username allowed to write lexicon edits.

    Comma-separated so a native reviewer can be added from Space settings.
    Comparison is case-insensitive because HF usernames are.
    """
    raw = os.environ.get(EDITOR_USER_ENV) or DEFAULT_EDITOR_USER
    names = tuple(n.strip() for n in raw.split(",") if n.strip())
    return names or (DEFAULT_EDITOR_USER,)


def editor_username() -> str:
    """The first allowed editor. Kept for messages and the ``/me`` payload."""
    return editor_usernames()[0]


def _is_allowed(name: str | None) -> bool:
    if name is None:
        return False
    return name.casefold() in {n.casefold() for n in editor_usernames()}


def attach_space_oauth(app: FastAPI) -> bool:
    """Install HF OAuth routes. Returns False when local mock cannot run.

    ``attach_huggingface_oauth`` raises locally unless a valid HF token is on
    the machine (it fakes login as that user). The Space itself always has
    ``SPACE_ID`` and the OAuth client env vars, so production is unaffected.
    A laptop selftest must still be able to ``create_app()`` without a token.
    """
    try:
        from huggingface_hub import attach_huggingface_oauth
    except ImportError as exc:
        log.warning("huggingface_hub OAuth helpers missing: %s", exc)
        return False
    try:
        attach_huggingface_oauth(app)
    except Exception as exc:  # noqa: BLE001 - local mock needs a live HF login
        log.warning("HF OAuth not attached: %s", exc)
        return False
    return True


def logged_in_username(request: Request) -> str | None:
    """HF username from the OAuth session cookie, or None if signed out."""
    try:
        from huggingface_hub import parse_huggingface_oauth
    except ImportError:
        return None
    try:
        info = parse_huggingface_oauth(request)
    except Exception:  # noqa: BLE001 - parse is documented as lax, stay lax
        return None
    if info is None or info.user_info is None:
        return None
    name = (info.user_info.preferred_username or "").strip()
    return name or None


def is_editor(request: Request) -> bool:
    return _is_allowed(logged_in_username(request))


def _deny(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": "forbidden", "message": message}},
    )


def require_editor(request: Request) -> str | Any:
    """Return the username, or an error JSONResponse the handler should return.

    Unauthenticated → 401. Signed in as anyone else → 403. The UI may hide the
    form; this is the real gate, including for ``gradio_client`` / curl POSTs.
    """
    name = logged_in_username(request)
    if name is None:
        return _deny(401, "Sign in with Hugging Face to edit the lexicon.")
    if not _is_allowed(name):
        allowed = ", ".join(editor_usernames())
        return _deny(
            403,
            f"Lexicon edits are restricted to Hugging Face user(s) {allowed}.",
        )
    return name
