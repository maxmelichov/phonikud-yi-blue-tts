"""HTML UI router: the demo page and the legacy form endpoint.

`POST /generate` is a drop-in replacement for the old Flask endpoint that
static/script.js still talks to, so the shipped frontend keeps working
unchanged: it posts the form fields ``mode`` / ``text`` / ``phonemes`` and
reads ``diacritics``, ``phonemes`` and ``audio`` back out of the JSON.

It runs the SAME pipeline as ``/v1/audio/speech`` — ``routes_v1.analyze`` —
rather than a parallel one of its own. The UI and the API used to derive their
phonemes differently and therefore produced different audio for identical
input; there is now one authority chain and one place it lives.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from .. import __version__
from .. import audio
# Off the submodule, never `from .. import runtimes`: the package also has a
# registry function by that name, and which one you get depends on import order.
from ..runtimes import RuntimeNotAvailable, loaded as loaded_runtime
# One pipeline and one synthesis path for both routers, so /generate and
# /v1/audio/speech cannot phonemize, fold, drop or report differently.
from .routes_v1 import (
    Analysis,
    Busy,
    InputForm,
    analyze,
    check_voice,
    dropped_report,
    heavy_slot,
    inventory_error,
    render_text,
    resolve_runtime,
)
from .dto import MAX_INPUT_CHARS
from ..runtimes.blue_yi import UtteranceTooLongError

logger = logging.getLogger(__name__)

router = APIRouter()

# The Space is started from arbitrary working directories (Docker WORKDIR, uv run,
# HF's entrypoint), so anchor templates/ to the repo root instead of the CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
templates = Jinja2Templates(directory=str(REPO_ROOT / "templates"))

#: Real Hasidic (Unterland/Central) Yiddish sentences, undotted as the G2P expects.
SAMPLES: tuple[str, ...] = (
    "מיט א פאר יאר צוריק",
    "וואס האט ער געזאגט",
    "א דאנק פאר די גוטע נייעס",
    "איך האב געהערט אז ער וועט קומען מארגן",
    "די קינדער שפילן זיך אין דרויסן",
)

_LEGACY_MODE_ALIASES = {"diacritics": "nikud"}
_FORMS: dict[str, InputForm] = {
    "text": InputForm.TEXT,
    "nikud": InputForm.NIKUD,
    "phonemes": InputForm.PHONEMES,
}


def voice_names() -> list[str]:
    """Voices of the resident runtime, or [] before one is loaded."""
    rt = loaded_runtime()
    if rt is None:
        return []
    try:
        return list(rt.voices())
    except Exception:  # noqa: BLE001 - the page must render without a voice list
        logger.exception("voices() failed; rendering without a voice picker")
        return []


@router.get("/", include_in_schema=False)
async def index(request: Request):
    """Render templates/index.html.

    Template context keys (the template is written against exactly these):
      voices   list - voice names of the resident runtime, [] if none loaded
      samples  list - Yiddish example sentences for the text box
      version  str  - yiddish_phonikud.__version__

    The page is a demo, not a datasheet: the catalog, the engine metadata and
    the phone inventory are served by `/v1/models/sources`,
    `/v1/models/state` and `/v1/phonemes/inventory` for callers that want them,
    and are not rendered here.
    """
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "voices": voice_names(),
            "samples": list(SAMPLES),
            "version": __version__,
        },
    )


@router.post("/generate", include_in_schema=False)
async def generate(
    mode: str = Form("text"),
    text: str = Form(""),
    phonemes: str = Form(""),
    voice: str = Form(""),
    speed: float = Form(1.0),
    n_steps: int | None = Form(None),
    cfg_scale: float | None = Form(None),
    seed: int | None = Form(None),
):
    """Legacy form endpoint kept wire-compatible with the shipped static/script.js.

    Returns nikud, diacritics (legacy alias of nikud), phonemes, audio as a
    data:audio/wav;base64 URI, the token table and any unsupported/dropped phone
    units. `voice`, `speed`, `n_steps`, `cfg_scale` and `seed` are optional
    additions; a post that omits them behaves exactly as before.

    Every form field is range-checked here the way the `/v1` DTOs are checked by
    pydantic — `speed`, `n_steps`, `cfg_scale`, `seed` and the input length — so
    a caller error is a 400 with a message and never a 500 with a traceback.
    Synthesis goes through the same chunk-and-join path as `/v1/audio/speech`,
    so a pasted paragraph is spoken here too rather than refused by the acoustic
    model's one-utterance-per-call limit.

    The three modes map onto the pipeline's three input forms:

      text      unpointed Yiddish. The v5 model points it for display only.
      nikud     YOUR pointing, passed to the G2P verbatim — it changes the
                reading (מלך -> mˈajləx vs מֶלֶךְ -> mˈɛləx), which is the whole
                point of the tab. Nothing is stripped, and the pointing model
                does not run.
      phonemes  IPA typed by hand; G2P skipped, inventory still checked.
    """
    form = _FORMS.get(_LEGACY_MODE_ALIASES.get(mode, mode))
    if form is None:
        return _error(
            400,
            "invalid_request",
            f"unknown mode {mode!r}; expected one of {sorted(_FORMS)}",
        )
    if not 0.5 <= speed <= 2.0:
        return _error(400, "invalid_request", "speed must be between 0.5 and 2.0")
    if n_steps is not None and not 1 <= n_steps <= 32:
        return _error(400, "invalid_request", "n_steps must be between 1 and 32")
    if cfg_scale is not None and not 1.0 <= cfg_scale <= 8.0:
        return _error(400, "invalid_request", "cfg_scale must be between 1.0 and 8.0")
    # `seed` was the one knob this endpoint forwarded unchecked, so a negative
    # value became `expected non-negative integer` out of numpy's default_rng —
    # a 500 with a traceback for what /v1 answers as a 400.
    if seed is not None and not 0 <= seed <= 2**31 - 1:
        return _error(400, "invalid_request", f"seed must be between 0 and {2**31 - 1}")

    source = phonemes if form is InputForm.PHONEMES else text
    if not source.strip():
        return _error(400, "invalid_request", "nothing to synthesize")
    # The same cap the /v1 DTOs enforce, checked before any engine call. A Form
    # field carries no max_length, so this endpoint used to accept whatever it
    # was posted and run the 1.1 GB pointing model plus the token table over it:
    # a 25 200-character post (which /v1 rejects with 400) was accepted here and
    # took the process down.
    if len(source) > MAX_INPUT_CHARS:
        return _error(
            400,
            "invalid_request",
            f"input is {len(source)} characters; the limit is {MAX_INPUT_CHARS}",
        )

    options = {
        name: value
        for name, value in (("n_steps", n_steps), ("cfg_scale", cfg_scale), ("seed", seed))
        if value is not None
    }

    try:
        rt = await run_in_threadpool(resolve_runtime, "")
        check_voice(rt, voice)
        async with heavy_slot():
            # One analyze() over the whole text for the display material (nikud,
            # phonemes, token table); synthesis then goes through the shared
            # chunk-and-join path, so a pasted paragraph is spoken here exactly
            # as /v1/audio/speech speaks it instead of failing on the acoustic
            # model's one-utterance-per-call limit.
            result: Analysis = await run_in_threadpool(analyze, source, form)
            if form is InputForm.PHONEMES and result.unsupported:
                return _error(400, "invalid_request", inventory_error(result))
            if not result.phonemes.strip():
                return _error(400, "invalid_request", "no phonemes to synthesize")
            samples, dropped = await run_in_threadpool(
                render_text, rt, source, form,
                voice=voice, speed=speed, options=options,
            )
    except LookupError as exc:  # unknown voice or unknown runtime id
        return _error(400, "invalid_request", str(exc))
    except RuntimeNotAvailable as exc:
        return _error(503, "not_available", str(exc))
    except Busy as exc:
        return _error(503, "not_available", str(exc))
    except UtteranceTooLongError as exc:
        # One unsplittable run of words longer than the checkpoint can render.
        return _error(400, "invalid_request", str(exc))
    except Exception as exc:  # surfaced to the UI rather than a bare 500 page
        logger.exception("generate failed (mode=%s)", mode)
        return _error(500, "internal_error", str(exc))

    wav = audio.pcm16_wav(samples, rt.sample_rate)
    nikud = result.nikud or None
    return JSONResponse(
        {
            "nikud": nikud,
            "diacritics": nikud,  # legacy key the old frontend still reads
            "phonemes": result.phonemes,
            "audio": "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii"),
            "tokens": [
                {
                    "word": row.word,
                    "nikud": row.nikud,
                    "ipa": row.ipa,
                    "route": row.route,
                    "confidence": row.confidence,
                    "layer": row.layer,
                    "reason": row.reason,
                }
                for row in result.tokens
            ],
            "unsupported": dropped_report(result.unsupported, dropped),
            # Notation rewrites applied to caller IPA, so the Phonemes tab can
            # show what it changed. Nothing here is silent by design.
            "notation": [sub.as_dict() for sub in result.notation],
            "stress_warning": result.stress_warning,
            "runtime": rt.id,
            "voice": voice,
            "sample_rate": rt.sample_rate,
        }
    )


def _error(status: int, code: str, message: str) -> JSONResponse:
    # ErrorBody shape, plus the legacy keys nulled so the old frontend's
    # `data.audio` access degrades quietly instead of throwing.
    return JSONResponse(
        {
            "error": {"code": code, "message": message},
            "nikud": None,
            "diacritics": None,
            "phonemes": None,
            "audio": None,
            "tokens": [],
            "unsupported": [],
        },
        status_code=status,
    )
