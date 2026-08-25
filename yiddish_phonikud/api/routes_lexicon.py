"""ABE101-only lexicon editor: lookup, update, and add gold / וי-class readings."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .. import auth, engine, lexicon_edits
from .routes_v1 import write_error

router = APIRouter(prefix="/v1/lexicon", tags=["lexicon"])


class LexiconUpdateBody(BaseModel):
    word: str = Field(..., max_length=40, description="Single Hebrew-script type.")
    ipa_primary: str = Field(..., max_length=80, description="Closed-inventory IPA.")
    variants: list[str] | None = Field(
        None, description="Optional alternate IPA readings; primary is always included."
    )
    layer: str = Field("G", max_length=1, description="G/L/E/A/N/X")
    note: str = Field("", max_length=240)
    vav_yud_class: str | None = Field(
        None,
        description="If set to `oʊ` or `ɔj`, rewrite the וי nucleus in the IPA.",
    )


def _gate(request: Request) -> str | JSONResponse:
    return auth.require_editor(request)


@router.get(
    "/me",
    summary="Who is signed in, and whether they may edit the lexicon",
)
async def lexicon_me(request: Request) -> dict[str, Any]:
    """Public: identity only. `can_edit` is true solely for the allowed username."""
    name = auth.logged_in_username(request)
    allowed = auth.editor_username()
    return {
        "username": name,
        "editor": allowed,
        "can_edit": name is not None and name == allowed,
        "login_url": "/oauth/huggingface/login",
        "logout_url": "/oauth/huggingface/logout",
        "persist": lexicon_edits.persist_status(),
        "engine_loaded": engine.is_loaded(),
    }


@router.get(
    "/lookup",
    summary="Look up a type in the live lexicon (ABE101 only)",
    response_model=None,
)
async def lexicon_lookup(request: Request, word: str = "") -> dict[str, Any] | JSONResponse:
    gated = _gate(request)
    if not isinstance(gated, str):
        return gated
    if not engine.is_loaded():
        return write_error(503, "not_available", "engine is still loading")
    try:
        return await run_in_threadpool(lexicon_edits.lookup, engine._g2p, word)
    except ValueError as exc:
        return write_error(400, "invalid_request", str(exc))


@router.post(
    "/update",
    summary="Update a lexicon reading (ABE101 only)",
    response_model=None,
)
async def lexicon_update(
    request: Request, body: LexiconUpdateBody,
) -> dict[str, Any] | JSONResponse:
    gated = _gate(request)
    if not isinstance(gated, str):
        return gated
    if not engine.is_loaded():
        return write_error(503, "not_available", "engine is still loading")
    try:
        result = await run_in_threadpool(
            lexicon_edits.save_edit,
            engine._g2p,
            word=body.word,
            ipa_primary=body.ipa_primary,
            variants=body.variants,
            layer=body.layer,
            note=body.note,
            vav_yud_class=body.vav_yud_class,
            username=gated,
        )
    except ValueError as exc:
        return write_error(400, "invalid_request", str(exc))
    except Exception as exc:  # noqa: BLE001
        return write_error(500, "internal_error", f"{type(exc).__name__}: {exc}")
    return result


@router.post(
    "/add",
    summary="Add a new lexicon type (ABE101 only)",
    response_model=None,
)
async def lexicon_add(
    request: Request, body: LexiconUpdateBody,
) -> dict[str, Any] | JSONResponse:
    gated = _gate(request)
    if not isinstance(gated, str):
        return gated
    if not engine.is_loaded():
        return write_error(503, "not_available", "engine is still loading")
    try:
        result = await run_in_threadpool(
            lexicon_edits.add_entry,
            engine._g2p,
            word=body.word,
            ipa_primary=body.ipa_primary,
            variants=body.variants,
            layer=body.layer,
            note=body.note,
            vav_yud_class=body.vav_yud_class,
            username=gated,
        )
    except ValueError as exc:
        return write_error(400, "invalid_request", str(exc))
    except Exception as exc:  # noqa: BLE001
        return write_error(500, "internal_error", f"{type(exc).__name__}: {exc}")
    return result


@router.get(
    "/edits",
    summary="List persisted ABE101 edits (ABE101 only)",
    response_model=None,
)
async def lexicon_edits_list(request: Request) -> dict[str, Any] | JSONResponse:
    gated = _gate(request)
    if not isinstance(gated, str):
        return gated
    return {
        "editor": gated,
        "persist": lexicon_edits.persist_status(),
        "edits": lexicon_edits.edits_snapshot(),
    }
