"""Yiddish Phonikud TTS — FastAPI application factory and uvicorn entry point.

Run: `python app.py --host 0.0.0.0 --port 7860`
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from yiddish_phonikud import __version__, engine
from yiddish_phonikud.api.routes_ui import VOICE_CAVEAT
from yiddish_phonikud.api.routes_ui import router as ui_router
from yiddish_phonikud.api.routes_v1 import WarmupState, write_error
from yiddish_phonikud.api.routes_v1 import router as v1_router
from yiddish_phonikud.runtimes import load_default

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("yiddish_phonikud.app")

ROOT = Path(__file__).resolve().parent

#: Set this to 0/false to keep the warmup from loading the acoustic runtime.
WARM_RUNTIME_ENV = "PHONIKUD_YI_WARM_RUNTIME"

DESCRIPTION = f"""
Hasidic (Unterland/Central) Yiddish text-to-speech.

**Grapheme-to-phoneme**: [`notmax123/phonikud-yi-engine`](https://huggingface.co/notmax123/phonikud-yi-engine)
— a Yiddish adaptation of Phonikud ([arXiv:2506.12311](https://arxiv.org/abs/2506.12311)), which
added diacritization-driven G2P to Hebrew TTS. The same idea is applied here to undotted Hasidic
Yiddish orthography: a v5 pointing model restores nikud for *reading*, and a lexicon+rule G2P with a
fixed authority chain (native gold verdicts > corpus-audio corrections > published pointing > model
guesses) converts Yiddish text to a closed IPA inventory. Speech is synthesized from that G2P output
over your text; the pointing model's guesses are shown, not fed back in.

**Voice**: {VOICE_CAVEAT}

Endpoints: `/` (demo UI), `/docs`, `/redoc`, `/health`, and the `/v1/*` API
(`/v1/audio/speech`, `/v1/audio/phonemize`, `/v1/audio/diacritize`, `/v1/models/*`,
`/v1/phonemes/inventory`).

Every non-2xx response — including body-validation failures and unknown paths — uses the
`{{"error": {{"code", "message"}}}}` envelope.
""".strip()


def _warm_runtime_enabled() -> bool:
    return os.environ.get(WARM_RUNTIME_ENV, "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _warmup(state: WarmupState) -> None:
    """Download + import the engine, then load the default runtime, off the
    request path.

    The engine snapshot is ~1.23 GB. A Space that has not bound its port within the
    platform's health window is killed, so this must never run inside startup.

    Loading the acoustic runtime here as well is a deliberate trade. Blue 2.5 is
    ~280 MB of ONNX plus up to ~1.5 s of session construction, which this warmup
    really does pay by calling the runtime's optional ``prepare()`` hook — so
    the cost is real, but it is a cost *somebody* pays either way, and paying it
    in the background means (a) the first caller is not the one who waits, and
    (b) `/health` can actually reach `ready` on an idle Space instead of
    promising a state it can never reach without traffic. The memory stays
    resident for the life of the process, which is what we want anyway on a
    single-process Space; set `PHONIKUD_YI_WARM_RUNTIME=0` for a deployment
    that would rather keep an idle box small and let the first request pay.
    """
    logger.info("warmup: engine starting (snapshot download may take several minutes)")
    try:
        engine.load()
    except Exception as exc:  # noqa: BLE001 - recorded, then reported by /health
        state.fail_engine(f"{type(exc).__name__}: {exc}")
        logger.exception("warmup: engine FAILED")
    else:
        logger.info("warmup: engine ready; %s", engine.info().get("dir"))

    if _warm_runtime_enabled():
        logger.info("warmup: loading default TTS runtime")
        try:
            rt = load_default()
            # Constructing the runtime only reads metadata (tts.json, vocab.json,
            # stats.npz, the voice listing — 0.01 s); the four ONNX sessions are
            # built lazily and cost ~0.4 s more. Without this call "ready" meant
            # "the object exists": /health went green and the first POST
            # /v1/audio/speech still paid the session build, which is precisely
            # the cost this warmup exists to move off the request path.
            warm = getattr(rt, "prepare", None)
            if callable(warm):
                warm()
            # Reading the voice list here too, for the same reason: it parses
            # five voice JSONs and screens each style for checkpoint
            # compatibility, and /v1/voices is answered from it.
            voices = rt.voices()
        except Exception as exc:  # noqa: BLE001
            state.fail_runtime(f"{type(exc).__name__}: {exc}")
            logger.exception("warmup: runtime FAILED")
        else:
            logger.info(
                "warmup: runtime %s ready at %d Hz, graphs resident, voices: %s",
                rt.id, rt.sample_rate, ", ".join(voices) or "none",
            )
    else:
        # Nothing attempted, so /health must not report a missing runtime as
        # either a failure or an unfinished warmup.
        state.skip_runtime()
        logger.info("warmup: runtime load skipped (%s=0)", WARM_RUNTIME_ENV)

    state.done()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    state = WarmupState()
    # /health reads this object; it is the only record of a warmup failure, and
    # unlike the module global it replaces, something actually reads it.
    app.state.warmup = state
    thread = threading.Thread(
        target=_warmup, args=(state,), name="engine-warmup", daemon=True
    )
    thread.start()
    app.state.engine_warmup = thread
    yield


# --------------------------------------------------------------------------
# Error envelope
#
# Without these, FastAPI answers a validation failure or a bad path with its own
# {"detail": ...} shape, and every client that switches on `error.code` sees
# something the contract says cannot happen.
# --------------------------------------------------------------------------

#: HTTP status -> error code. Anything unlisted becomes internal_error (5xx) or
#: invalid_request (4xx).
_STATUS_CODES: dict[int, str] = {
    400: "invalid_request",
    404: "not_found",
    405: "method_not_allowed",
    # No 413 entry: nothing in this app emits it. Over-long bodies are rejected
    # by the DTOs' max_length as 422 -> invalid_request, and uvicorn has no body
    # size limit configured, so a 413 mapping would be a status this stack can
    # never produce. `_code_for` covers it anyway if that ever changes.
    422: "invalid_request",
    500: "internal_error",
    503: "not_available",
}


def _code_for(status_code: int) -> str:
    listed = _STATUS_CODES.get(status_code)
    if listed is not None:
        return listed
    return "invalid_request" if 400 <= status_code < 500 else "internal_error"


def _validation_message(exc: RequestValidationError) -> str:
    """Pydantic's error list, flattened into one readable sentence."""
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()) if item != "body")
        message = str(error.get("msg", "invalid value"))
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "request body failed validation"


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return write_error(400, "invalid_request", _validation_message(exc))


async def http_error_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return write_error(exc.status_code, _code_for(exc.status_code), detail)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # Reached only for exceptions no handler claimed; the traceback goes to the
    # log, and the client still gets the documented envelope rather than an
    # HTML 500 page.
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return write_error(500, "internal_error", f"{type(exc).__name__}: {exc}")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Yiddish Phonikud TTS",
        version=__version__,
        description=DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
    app.include_router(v1_router)
    app.include_router(ui_router)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
    return app


app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yiddish Phonikud TTS server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    # No debug/reload: the old Flask app ran debug=True in the published Space, which
    # exposes the Werkzeug console to the internet. Deliberately dropped.
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
