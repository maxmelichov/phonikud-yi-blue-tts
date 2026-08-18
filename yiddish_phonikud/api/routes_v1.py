"""`/v1/*` and `/health` — the MamboTTS-compatible HTTP surface, plus the ONE
canonical synthesis pipeline every caller in this app goes through.

Ported from `mambotts-server/src/server/handlers/{speech,load,state,metadata}.rs`
and `crates/mambotts-py/python/mambotts_server.py`, so a client written against
MamboTTS sees the same routes, the same JSON field names, and the same
`{"error": {"code", "message"}}` envelope on failure.

Two deliberate divergences from the Rust server:
  * phoneme input does not require `stream=true` here — both paths accept it;
  * `/health` never returns 503, because a cold Hugging Face Space is still
    downloading the 1.23 GB engine and must be diagnosable, not look dead.

THE AUTHORITY CHAIN (read this before changing `analyze`)
--------------------------------------------------------
The G2P owns the pronunciation. `engine.text_to_ipa(user_text)` already walks
the project's fixed authority chain internally — native gold verdicts >
corpus-audio corrections > published pointing > model guesses — so the IPA is
derived from the caller's own text and from nothing else. The v5 pointing model
output is a DISPLAY artefact: it is shown to the user and it fills the token
table's `nikud` column, and it is never fed back into the G2P.

That is not a style preference, it was measured. Over 120 real Hasidic Yiddish
sentences from the corpus (2 548 tokens), `text_to_ipa(text_to_nikud(t))`
differs from `text_to_ipa(t)` in 108 of 120 sentences and in 31% of all words,
and the differences are systematically worse: 95 tokens (61 distinct types) lose
an evidence-backed reading and get a mechanical read of a guessed pointing
instead —

    לך     ləxˈu  -> lxu     (audio-homograph verdict -> vowel-less lk-fallback)
    חסדך   xˈasdəxu -> xˈazdxu (audio-homograph verdict -> lk-fallback)
    אתה    ˈatu   -> ˈatə    (verified pointed edition -> Whole-Hebrew read)
    מצוותיו miʦvˈɔjsujv -> miʦvɔjsˈuji
    פין    pin    -> fin     (a native GOLD verdict overridden, because the
                              model guessed a rafe on the פ)

and the engine's `reason` histogram shows the whole rescue chain going dark:
`sefaria-pointed` 95 -> 0, `audio-homograph` 28 -> 0, `model-pointed-guess`
97 -> 0, replaced by `lk-fallback` 225. Nominal confidence goes *up*
(359 LOW->MED "upgrades") because pointing makes the rules unambiguous — which
is exactly the trap: confidence inflation on top of a tier-4 guess, and the
hut/hot/hat dialect mixing that `Phonikud-yi/src/README.md` §3 blames for the
released voice mixing dialects.

Pointing the caller *supplies themselves* is a different matter and is honoured
verbatim (see `InputForm.NIKUD`): the engine has real pointed-input paths —
`read_pointed_wh` / `read_pointed_merged`, the `_WH_WHEN_POINTED` gate, the
פּ/פֿ contradiction check, and every vowel rule gated on a point being present.
Hand-pointing `מלך` as `מֶלֶךְ` moves it from `mˈajləx` to `mˈɛləx`; `סוכה` as
`סוּכָּה` from `sˈixə` to `sˈikə`. Hand pointing is a real editorial control.
Model pointing fed forward is a tier inversion. Same mechanism, opposite value.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import AsyncIterator, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from urllib.parse import quote

import numpy as np
from fastapi import APIRouter, Body, Request, status
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from .. import __version__
from .. import audio, engine, phones, registry

# Imported off the submodule so the name always means the module, never the
# registry's runtimes() function (see yiddish_phonikud/__init__.py).
from ..runtimes import (
    Runtime,
    RuntimeNotAvailable,
    instance as runtime_instance,
    load as load_and_cache,
    load_default,
    loaded as loaded_runtime,
    state as runtime_state,
)
# blue_yi's module body is numpy + the registry only; onnxruntime is imported
# lazily inside _ensure_sessions, so naming the exception here costs nothing.
from ..runtimes.blue_yi import UtteranceTooLongError
from .dto import (
    DiacritizeResponse,
    ErrorBody,
    HealthResponse,
    LanguagesResponse,
    LoadBody,
    LoadResponse,
    ModelSource,
    ModelSourceFile,
    ModelSourcesResponse,
    PhonemeInventoryResponse,
    PhonemizeBody,
    PhonemizeResponse,
    RuntimeCapabilitiesResponse,
    SpeechBody,
    StateResponse,
    TokenRowDTO,
    VoicesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

LANGUAGE_ITEMS: tuple[dict[str, str], ...] = (
    {"code": "yi", "name": "Yiddish (Hasidic)"},
)

#: Response header carrying the units the pipeline could not deliver to the
#: acoustic model. Documented on /v1/audio/speech; also set on the stream.
DROPPED_HEADER = "X-Dropped-Units"

_ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    400: {"model": ErrorBody, "description": "invalid_request"},
    500: {"model": ErrorBody, "description": "internal_error"},
    503: {"model": ErrorBody, "description": "no_model / not_available"},
}


def write_error(status_code: int, code: str, message: str) -> JSONResponse:
    """The single error exit for this router — port of `errors.rs::write_error`.

    Handlers return this instead of raising `HTTPException`, which would leak
    FastAPI's `{"detail": ...}` shape into clients that parse `error.code`.
    (`app.py` also installs handlers so the framework's own 404s and body
    validation failures come back in this envelope.)
    """
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


# --------------------------------------------------------------------------
# Pointing (C10: the v5 model runs once per request, not twice)
# --------------------------------------------------------------------------


def pointing(text: str) -> str:
    """The pointing of record for `text` — the v5 model's output, for display.

    Only `/v1/audio/diacritize` needs the pointing *alone*. Everything that
    also needs IPA or the token table goes through `analyze()`, which calls
    `engine.analyze()`: that runs the pointing model exactly once and hands the
    same pointed string to the token table, so the old shape (`text_to_nikud`
    for the display string, then `token_table` pointing the same text again
    internally) cannot come back. Pointing is ~296 ms against the G2P's 1.7 ms,
    so the duplicate was most of a request on CPU-basic hardware.
    """
    return engine.text_to_nikud(text)


# --------------------------------------------------------------------------
# The canonical pipeline — used by /v1/audio/speech, /v1/audio/phonemize and
# /generate alike, so the three can never disagree about what a text sounds
# like (C1/C2/C3).
# --------------------------------------------------------------------------


class InputForm(str, Enum):
    """What the caller's string already is."""

    #: Unpointed (or incidentally pointed) Yiddish. The v5 model points it for
    #: display; the G2P reads the caller's text.
    TEXT = "text"
    #: Pointed Yiddish supplied by a human. Passed to the G2P verbatim — the
    #: marks change the reading — and echoed back as the pointing of record.
    NIKUD = "nikud"
    #: IPA over the closed inventory. No diacritizer, no G2P.
    PHONEMES = "phonemes"


@dataclass(frozen=True)
class Analysis:
    """One G2P pass over one string: everything every caller needs from it.

    `phonemes` and `tokens` come from the same pass over the same text, which
    is why they cannot contradict each other; `nikud` is display material and
    is not what produced `phonemes`.
    """

    form: InputForm
    text: str
    nikud: str
    phonemes: str
    tokens: tuple[engine.TokenRow, ...] = ()
    unsupported: tuple[str, ...] = ()


def analyze(
    text: str,
    form: InputForm = InputForm.TEXT,
    *,
    with_tokens: bool = True,
    with_nikud: bool = True,
) -> Analysis:
    """THE pipeline. Blocking (ONNX + table lookups) — call it off the loop.

    * `TEXT`     — `phonemes = engine.text_to_ipa(text)`, straight over the
      caller's text so the engine's authority chain decides every reading.
      `nikud` is the v5 pointing, for display only.
    * `NIKUD`    — identical, except the caller's marks reach the G2P (they do
      change readings) and are echoed back as `nikud` without running the
      pointing model.
    * `PHONEMES` — the string is already IPA; only inventory validation runs.

    `with_nikud=False` (and `with_tokens=False`) is how `/v1/audio/speech` calls
    this: speech needs none of the display material, so the v5 pointing model —
    the single most expensive thing in the request — does not run at all. The
    token table cannot avoid it, because pointing every row is what it is for.
    """
    if form is InputForm.PHONEMES:
        ipa = text.strip()
        return Analysis(
            form=form,
            text=text,
            nikud="",
            phonemes=ipa,
            tokens=(),
            unsupported=tuple(phones.validate(ipa)),
        )

    # Note the argument to every G2P call below: `text`, never `nikud`.
    # See THE AUTHORITY CHAIN above.
    if form is InputForm.NIKUD:
        # The human's pointing IS the pointing of record: the model never runs,
        # and the token table aligns its rows against the caller's own marks.
        nikud = text
        ipa = engine.text_to_ipa(text)
        rows = tuple(engine.token_table(text, nikud=nikud)) if with_tokens else ()
    elif with_nikud or with_tokens:
        # engine.analyze() is the single-pass bridge: one pointing call feeds
        # both the display string and the token table.
        result = engine.analyze(text, tokens=with_tokens)
        nikud = result.nikud
        ipa = result.ipa
        rows = tuple(result.tokens)
    else:
        # Speech: no display material is wanted, so the pointing model — the
        # single most expensive thing in the request — does not run at all.
        nikud = ""
        ipa = engine.text_to_ipa(text)
        rows = ()
    return Analysis(
        form=form,
        text=text,
        nikud=nikud,
        phonemes=ipa,
        tokens=rows,
        unsupported=tuple(phones.validate(ipa)),
    )


def input_form(body: SpeechBody | PhonemizeBody) -> InputForm:
    """The form flags of a request body, resolved in documented precedence."""
    if getattr(body, "input_is_phonemes", False):
        return InputForm.PHONEMES
    if getattr(body, "input_is_nikud", False):
        return InputForm.NIKUD
    return InputForm.TEXT


def sampler_options(body: SpeechBody) -> dict[str, object]:
    """The optional per-request knobs, omitting anything the caller left unset.

    Passed through `Runtime.synthesize(**options)`. Runtimes that have no
    sampler ignore them; only fields the caller actually sent are forwarded, so
    a default request reaches every runtime with an empty option set.
    """
    supplied = (
        ("n_steps", body.n_steps),
        ("cfg_scale", body.cfg_scale),
        ("seed", body.seed),
    )
    return {name: value for name, value in supplied if value is not None}


def dropped_report(unsupported: Iterable[str], dropped: Iterable[str]) -> list[str]:
    """Units the caller should be told about, deduped and in order.

    Two different casualties, one list: units outside the closed inventory, and
    units the loaded voice could not render even after folding. `dropped` comes
    back from `synthesize()` per call — it used to be read off
    `runtime.last_dropped`, per-instance state on a process-wide singleton, so
    concurrent requests reported each other's casualties.
    """
    out: list[str] = []
    for unit in (*unsupported, *dropped):
        if unit and unit not in out:
            out.append(unit)
    return out


def speak(
    rt: Runtime,
    ipa: str,
    *,
    voice: str = "",
    speed: float = 1.0,
    options: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """IPA -> (float32 mono samples, dropped units). Blocking."""
    samples, dropped = rt.synthesize(ipa, voice=voice, speed=speed, **dict(options or {}))
    return samples, list(dropped)


# --------------------------------------------------------------------------
# Chunking — a REQUIREMENT of the acoustic model, on every path
# --------------------------------------------------------------------------
#
# BlueTTS 2.5 renders exactly one utterance per call and refuses text longer
# than `blue_yi.MAX_TEXT_TOKENS`, so long text has to be cut into sentences and
# the pieces concatenated. That used to happen only when `stream=true`, which
# made `stream` a correctness switch rather than a delivery choice: a default
# POST /v1/audio/speech with an ordinary paragraph — and the demo UI's Generate
# button, which is never streamed — came back as 500 internal_error carrying a
# duration message, while the same text streamed fine.
#
# So both paths and both routers go through `text_chunks` + `render_chunks`
# below. The stream emits each chunk as it is rendered; the non-streaming path
# joins them (`audio.join_chunks`, one 60 ms gap per seam, RECIPE G13) and
# returns one WAV. Same chunker, same boundaries, same audio.


def text_chunks(text: str, form: InputForm) -> list[str]:
    """`text` cut into synthesizable pieces, at boundaries the G2P agrees with.

    The multiword index keeps a chunk boundary out of the middle of a
    whitespace-spelled lexicon entry, which the engine matches across the token
    stream: without it a chunked request says `a far jur` where an unchunked one
    says `a pˈur jur`. It is only meaningful for Yiddish text — caller-supplied
    IPA has no lexicon to protect, and asking the engine for the index would
    load the 1.23 GB bundle needlessly.
    """
    index = None
    if form is not InputForm.PHONEMES:
        try:
            index = engine.multiword()
        except Exception as exc:  # noqa: BLE001 - chunking works without it
            logger.warning("multiword index unavailable, chunking blind: %r", exc)
    return audio.chunk_text(text, multiword=index)


def render_chunk(
    rt: Runtime,
    chunk: str,
    form: InputForm,
    *,
    voice: str,
    speed: float,
    options: Mapping[str, object],
) -> tuple[np.ndarray, list[str]]:
    """Phonemize and synthesize ONE chunk: (samples, dropped units).

    An empty result is normal and not an error: a separator-only chunk, or text
    the G2P legitimately quarantines to nothing (digits, a URL, a lone geresh
    word) has no phonemes to speak.
    """
    piece = analyze(chunk, form, with_tokens=False, with_nikud=False)
    if form is InputForm.PHONEMES and piece.unsupported:
        raise ValueError(
            "phonemes outside the Yiddish inventory: " + " ".join(piece.unsupported)
        )
    if not piece.phonemes.strip():
        return np.zeros(0, dtype=np.float32), []
    return speak(rt, piece.phonemes, voice=voice, speed=speed, options=options)


def render_text(
    rt: Runtime,
    text: str,
    form: InputForm,
    *,
    voice: str = "",
    speed: float = 1.0,
    options: Mapping[str, object] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Chunk `text`, synthesize every chunk, join them: (samples, dropped).

    The single non-streaming synthesis entry point, shared by
    /v1/audio/speech and the UI's /generate so the two cannot diverge.
    """
    opts = options or {}
    parts: list[np.ndarray] = []
    dropped: list[str] = []
    for chunk in text_chunks(text, form):
        if not chunk.strip():
            continue
        samples, chunk_dropped = render_chunk(
            rt, chunk, form, voice=voice, speed=speed, options=opts
        )
        if samples.size:
            parts.append(samples)
        for unit in chunk_dropped:
            if unit not in dropped:
                dropped.append(unit)
    return audio.join_chunks(parts, rt.sample_rate), dropped


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def resolve_runtime(runtime_id: str) -> Runtime:
    """Returns the runtime to synthesize with, loading it if needed.

    A per-request `runtime` does NOT change what the process has loaded. It used
    to: `SpeechBody.runtime` went through the same loader `POST /v1/models/load`
    uses, so one caller asking for `piper_yi` left every other caller's
    `/v1/voices`, `/v1/models/state` and default sample rate switched to Piper —
    and the next request that named a Blue voice got `400 unknown voice
    'female' for runtime piper_yi`. `runtimes.instance()` keeps a small
    per-id cache instead and leaves the resident runtime alone; `/v1/models/load`
    remains the only way to change process state.

    Raises `LookupError` for an unknown id (-> 400) and `RuntimeNotAvailable`
    for a declared-but-unusable runtime (-> 503).
    """
    if not runtime_id:
        current = loaded_runtime()
        return current if current is not None else load_default()
    if registry.runtime(runtime_id) is None:
        raise LookupError(f"unknown runtime: {runtime_id}")
    return runtime_instance(runtime_id)


# --------------------------------------------------------------------------
# Admission control
# --------------------------------------------------------------------------
#
# Nothing else in the stack bounds concurrency. Starlette's threadpool admits 40
# blocking calls at once, and each in-flight synthesis holds its own duration
# predictor and flow-loop buffers while ONNX opens intra-op workers per session —
# so ten simultaneous requests at n_steps=32 put dozens of compute threads and
# several gigabytes on a 2-vCPU, 16 GB Space and every one of them gets slower.
# Two at a time is the right number for CPU-basic hardware: it keeps one request
# rendering while another phonemizes, and it is what `available_cpus()` sees
# there. Callers past the queue limit get 503 `not_available` with Retry-After
# rather than a box that swaps.


class Busy(RuntimeError):
    """Too many syntheses in flight; the caller should retry."""


#: Concurrent syntheses. Override for a bigger box.
MAX_CONCURRENT_SYNTHESES = max(
    1, int(os.environ.get("PHONIKUD_YI_MAX_CONCURRENCY", "2") or 2)
)
#: How long a request waits for a slot before it is turned away.
SYNTHESIS_QUEUE_SECONDS = 30.0

_synthesis_slots = threading.BoundedSemaphore(MAX_CONCURRENT_SYNTHESES)


@asynccontextmanager
async def heavy_slot() -> AsyncIterator[None]:
    """Holds one compute slot for the duration of the block.

    Wrap anything that runs an ONNX graph over caller-supplied text: synthesis,
    and the v5 pointing model behind /v1/audio/diacritize, /v1/audio/phonemize
    and /generate (296 ms and up to a gigabyte of activations per call).

    The wait is asynchronous on purpose. Blocking on the semaphore inside
    `run_in_threadpool` would park a threadpool worker per waiting request, and
    with 40 workers a burst would leave `/health` and `/v1/models/state` waiting
    behind the queue — a busy Space would read as a dead one. Polling from the
    event loop keeps every waiter off the threadpool.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + SYNTHESIS_QUEUE_SECONDS
    while not _synthesis_slots.acquire(blocking=False):
        if loop.time() >= deadline:
            raise Busy(
                f"the server is already rendering {MAX_CONCURRENT_SYNTHESES} "
                f"request(s) and a slot did not free up within "
                f"{SYNTHESIS_QUEUE_SECONDS:.0f}s; retry shortly"
            )
        await asyncio.sleep(0.05)
    try:
        yield
    finally:
        _synthesis_slots.release()


def check_voice(rt: Runtime, voice: str) -> None:
    """Raises `LookupError` unless `voice` is empty or offered by `rt`.

    Blue ships five distinct speakers, so `voice` finally selects something
    audible: silently falling back to the default would hand back audio in the
    wrong voice with a 200.
    """
    if not voice:
        return
    available = list(rt.voices())
    if voice not in available:
        raise LookupError(
            f"unknown voice {voice!r} for runtime {rt.id}; "
            f"available: {', '.join(available) or 'none'}"
        )


def dropped_header_value(units: Iterable[str]) -> str:
    """The `X-Dropped-Units` value: space-separated, percent-encoded UTF-8.

    HTTP header values are latin-1, and every unit worth reporting here (`ˈ`,
    `ʧ`, `ʤ`, `ː`) is outside it — sending them raw raises at response
    construction time and turns a successful synthesis into a 500. Percent
    encoding keeps the header ASCII and one `urllib.parse.unquote` from the real
    characters.
    """
    return " ".join(quote(unit, safe="") for unit in units)


def _speech_headers(rt: Runtime, dropped: Iterable[str], *, attachment: bool) -> dict[str, str]:
    headers = {
        "Cache-Control": "no-store",
        "X-Runtime": rt.id,
        "X-Sample-Rate": str(rt.sample_rate),
        DROPPED_HEADER: dropped_header_value(dropped),
    }
    if attachment:
        headers["Content-Disposition"] = 'attachment; filename="speech.wav"'
    return headers


# --------------------------------------------------------------------------
# Health & state
# --------------------------------------------------------------------------


@dataclass
class WarmupState:
    """What the background warmup has managed so far. Owned by `app.py`.

    `/health` reads this. The previous build wrote an `ENGINE_LOAD_ERROR`
    global that nothing ever read, so a hard `verify()` failure looked exactly
    like a still-downloading cold start — forever.
    """

    #: Verbatim `TypeName: message` of the engine failure, or "".
    engine_error: str = ""
    #: Verbatim failure of the default-runtime load, or "".
    runtime_error: str = ""
    #: False while the warmup thread is still running.
    finished: bool = False
    #: True when the deployment opted out of warming the runtime
    #: (PHONIKUD_YI_WARM_RUNTIME=0), so a missing runtime is neither a failure
    #: nor an unfinished warmup.
    runtime_skipped: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def fail_engine(self, message: str) -> None:
        with self.lock:
            self.engine_error = message

    def fail_runtime(self, message: str) -> None:
        with self.lock:
            self.runtime_error = message

    def skip_runtime(self) -> None:
        with self.lock:
            self.runtime_skipped = True

    def done(self) -> None:
        with self.lock:
            self.finished = True

    def snapshot(self) -> tuple[str, str, bool, bool]:
        with self.lock:
            return (
                self.engine_error,
                self.runtime_error,
                self.finished,
                self.runtime_skipped,
            )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness, warm-up state, and warm-up failures",
    description=(
        "Always 200, even before anything is loaded — a cold Space is still "
        "downloading the ~1.23 GB `notmax123/phonikud-yi-engine` snapshot and "
        "must be diagnosable rather than look dead.\n\n"
        "`status` is one of three states and they are genuinely "
        "distinguishable:\n\n"
        "* `warming` — the background warmup thread is still working.\n"
        "* `ready` — the G2P engine **and** a TTS runtime are resident. This is "
        "reachable on an idle Space: the warmup loads the default runtime after "
        "the engine, so nobody has to send a synthesis request to make "
        "`/health` go green.\n"
        "* `error` — warmup finished and something failed. `engine_error` "
        "and/or `runtime_error` carry the reason verbatim, and the state is "
        "terminal: retrying will not clear it.\n\n"
        "A failed *runtime* with a healthy engine still leaves "
        "`/v1/audio/phonemize` and `/v1/audio/diacritize` fully usable.\n\n"
        "A deployment that would rather keep an idle box small can set "
        "`PHONIKUD_YI_WARM_RUNTIME=0`; the runtime is then loaded by the first "
        "synthesis request, and `ready` means \"engine resident, runtime on "
        "demand\"."
    ),
)
async def health(request: Request) -> HealthResponse:
    rt = loaded_runtime()
    engine_ready = engine.is_loaded()
    warmup: WarmupState | None = getattr(request.app.state, "warmup", None)
    if warmup is None:
        # No lifespan ran (embedded/test use): report only what is observable,
        # and do not claim a warmup is in flight when none is.
        engine_error = runtime_error = ""
        finished, runtime_skipped = True, True
    else:
        engine_error, runtime_error, finished, runtime_skipped = warmup.snapshot()

    if engine_ready and (rt is not None or (runtime_skipped and finished)):
        # With runtime warmup disabled, the engine being loaded is all the
        # readiness there is to report: the first synthesis request loads the
        # acoustic model on demand.
        state_name = "ready"
    elif not finished:
        state_name = "warming"
    else:
        state_name = "error" if (engine_error or runtime_error) else "warming"

    return HealthResponse(
        status=state_name,
        engine_loaded=engine_ready,
        runtime_loaded=rt is not None,
        runtime=rt.id if rt is not None else "",
        engine_error=engine_error,
        runtime_error=runtime_error,
        warming=not finished,
        version=__version__,
    )


@router.get(
    "/v1/models/state",
    response_model=StateResponse,
    summary="Loaded model state",
    description=(
        "Reports which runtime is resident, the file it was loaded from, and its "
        "output sample rate. Port of `GET /v1/models` in the MamboTTS server."
    ),
)
async def model_state() -> StateResponse:
    return StateResponse(**runtime_state())


@router.get(
    "/v1/models/sources",
    response_model=ModelSourcesResponse,
    summary="Runtime catalog",
    description=(
        "Every runtime the registry declares, with its install set, capability "
        "flags, whether its files are present on this machine (`installed`), and "
        "whether this build implements it (`available`). Built from the registry "
        "the same way `sources.rs` builds it from `mambotts-registry`."
    ),
)
async def model_sources() -> ModelSourcesResponse:
    root = _model_root()
    return ModelSourcesResponse(
        runtimes=[_source_from_manifest(m, root) for m in registry.runtimes()],
        engine_repo=registry.ENGINE_REPO_ID,
        default_paths=[str(root)],
    )


def _model_root() -> Path:
    """Runtime files ship next to the package in the Space image."""
    return Path(__file__).resolve().parents[2]


def _source_from_manifest(
    manifest: registry.RuntimeManifest, root: Path
) -> ModelSource:
    return ModelSource(
        id=manifest.id,
        name=manifest.name,
        version=manifest.version,
        size=manifest.size,
        description=manifest.description,
        files=[ModelSourceFile(name=f.name, url=f.url) for f in manifest.files],
        directory=manifest.directory,
        installed=registry.is_installed(manifest, root),
        available=manifest.available,
        capabilities=RuntimeCapabilitiesResponse(
            yiddish=manifest.capabilities.yiddish,
            streaming=manifest.capabilities.streaming,
            voice_reference=manifest.capabilities.voice_reference,
            fixed_voices=manifest.capabilities.fixed_voices,
        ),
    )


@router.post(
    "/v1/models/load",
    response_model=LoadResponse,
    responses=_ERROR_RESPONSES,
    summary="Load a TTS runtime",
    description=(
        "Loads the named runtime into memory (idempotent — loading the resident "
        "runtime again is a no-op). Unknown ids return 400 `invalid_request`; a "
        "runtime that is catalogued but unimplemented or missing its files "
        "returns 503 `not_available`. An empty body loads the default runtime."
    ),
)
async def model_load(body: LoadBody | None = Body(default=None)) -> Response:
    request = body or LoadBody()
    runtime_id = request.runtime or registry.DEFAULT_RUNTIME_ID
    if registry.runtime(runtime_id) is None:
        return write_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            f"unknown runtime: {runtime_id}",
        )
    try:
        rt = await run_in_threadpool(load_and_cache, runtime_id)
    except RuntimeNotAvailable as err:
        return write_error(
            status.HTTP_503_SERVICE_UNAVAILABLE, "not_available", str(err)
        )
    except Exception as err:  # noqa: BLE001 - report any loader failure verbatim
        logger.exception("failed to load runtime %s", runtime_id)
        return write_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            f"failed to load model: {err}",
        )
    state = runtime_state()
    return JSONResponse(
        LoadResponse(
            status="loaded", runtime=rt.id, model=str(state.get("model", ""))
        ).model_dump()
    )


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


@router.get(
    "/v1/voices",
    response_model=VoicesResponse,
    responses=_ERROR_RESPONSES,
    summary="Voices of the loaded runtime",
    description=(
        "Names accepted in `SpeechBody.voice`; an unknown name is a 400 rather "
        "than a silent fallback. Returns 503 `no_model` when no runtime is "
        "loaded yet — check `/health` to tell warming from broken."
    ),
)
async def voices() -> Response:
    rt = loaded_runtime()
    if rt is None:
        return write_error(
            status.HTTP_503_SERVICE_UNAVAILABLE, "no_model", "no model loaded"
        )
    return JSONResponse(
        VoicesResponse(runtime=rt.id, voices=rt.voices()).model_dump()
    )


@router.get(
    "/v1/languages",
    response_model=LanguagesResponse,
    summary="Supported languages",
    description=(
        "This build is Yiddish-only: Hasidic (Unterland/Central) Yiddish in "
        "Hebrew script. Kept as a list for MamboTTS client compatibility."
    ),
)
async def languages() -> LanguagesResponse:
    return LanguagesResponse(
        languages=[item["code"] for item in LANGUAGE_ITEMS],
        items=[dict(item) for item in LANGUAGE_ITEMS],
    )


@router.get(
    "/v1/phonemes/inventory",
    response_model=PhonemeInventoryResponse,
    summary="Closed Yiddish phone inventory",
    description=(
        "The authoritative closed inventory the G2P may emit, split into vowels, "
        "consonants, and marks. When a runtime is loaded, "
        "`runtime_vocab_missing` lists inventory units the acoustic model's "
        "vocabulary lacks — empty for BlueTTS 2.5, which carries the whole "
        "inventory natively; `ʧ` and `ʤ` for the legacy Piper voice, where they "
        "are folded to `tʃ`/`dʒ` before synthesis and are therefore informative "
        "rather than an error."
    ),
)
async def phoneme_inventory() -> PhonemeInventoryResponse:
    missing: list[str] = []
    rt = loaded_runtime()
    if rt is not None:
        vocab = rt.vocab()
        missing = [
            unit
            for unit in sorted(phones.INVENTORY)
            if any(char not in vocab for char in unit)
        ]
    return PhonemeInventoryResponse(
        vowels=list(phones.VOWELS),
        consonants=list(phones.CONSONANTS),
        marks=list(phones.MARKS),
        inventory=sorted(phones.INVENTORY),
        runtime_vocab_missing=missing,
    )


# --------------------------------------------------------------------------
# Linguistics
# --------------------------------------------------------------------------


@router.post(
    "/v1/audio/diacritize",
    response_model=DiacritizeResponse,
    responses=_ERROR_RESPONSES,
    summary="Add Yiddish nikud",
    description=(
        "Runs the v5 pointing model over unpointed Yiddish and returns the "
        "pointed text, for reading and hand-correcting.\n\n"
        "Sending the result back to `/v1/audio/speech` with "
        "`input_is_nikud=true` speaks *your* pointing, which is a real "
        "editorial control: the engine reads points where they are present. "
        "Sending it back **unedited** is not neutral — this endpoint's output is "
        "a tier-4 model guess, and the synthesis pipeline deliberately does not "
        "feed it forward, because doing so measurably overrides native gold "
        "verdicts, audio-confirmed readings and verified pointed editions "
        "(31% of words change; see the module docstring for the numbers)."
    ),
)
async def diacritize(body: PhonemizeBody) -> Response:
    if not body.input.strip():
        return write_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            "request body must contain input",
        )
    try:
        async with heavy_slot():
            nikud = await run_in_threadpool(pointing, body.input)
    except Busy as err:
        return write_error(
            status.HTTP_503_SERVICE_UNAVAILABLE, "not_available", str(err)
        )
    except Exception as err:  # noqa: BLE001 - engine failures are 500s, reported verbatim
        logger.exception("diacritize failed")
        return write_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(err)
        )
    return JSONResponse(DiacritizeResponse(nikud=nikud).model_dump())


@router.post(
    "/v1/audio/phonemize",
    response_model=PhonemizeResponse,
    responses=_ERROR_RESPONSES,
    summary="Diacritize and phonemize, with a full G2P trace",
    description=(
        "One pass of the same pipeline `/v1/audio/speech` uses, with its "
        "workings exposed: the pointed text, the IPA transcription, a per-token "
        "table (route, confidence, lexical layer, engine reason), and "
        "`unsupported` — units the engine emitted that fall outside the closed "
        "inventory. Multiword lexicon entries appear as a single token row.\n\n"
        "`phonemes` and `tokens` come from the **same** pass over the text you "
        "sent, so the table always explains the transcription. `nikud` is "
        "display material: the IPA is derived from your text through the "
        "engine's authority chain (gold verdicts > corpus audio > published "
        "pointing > model guesses), not from the pointing model's output, so "
        "`phonemes` need not match a naive reading of `nikud`.\n\n"
        "Set `input_is_nikud=true` when you are sending pointed text: your "
        "pointing is then echoed back as `nikud` and the pointing model is not "
        "run. Marks in the input reach the G2P and change readings either way."
    ),
)
async def phonemize(body: PhonemizeBody) -> Response:
    if not body.input.strip():
        return write_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            "request body must contain input",
        )
    try:
        async with heavy_slot():
            result = await run_in_threadpool(analyze, body.input, input_form(body))
    except Busy as err:
        return write_error(
            status.HTTP_503_SERVICE_UNAVAILABLE, "not_available", str(err)
        )
    except Exception as err:  # noqa: BLE001
        logger.exception("phonemize failed")
        return write_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(err)
        )
    return JSONResponse(
        PhonemizeResponse(
            nikud=result.nikud,
            phonemes=result.phonemes,
            tokens=[
                TokenRowDTO(
                    word=row.word,
                    nikud=row.nikud,
                    ipa=row.ipa,
                    route=row.route,
                    confidence=row.confidence,
                    layer=row.layer,
                    reason=row.reason,
                )
                for row in result.tokens
            ],
            unsupported=list(result.unsupported),
        ).model_dump()
    )


# --------------------------------------------------------------------------
# Speech
# --------------------------------------------------------------------------


@router.post(
    "/v1/audio/speech",
    responses={
        200: {
            "content": {
                "audio/wav": {},
                "application/octet-stream": {},
            },
            "description": (
                "A single WAV body, or the framed chunk stream when "
                "`stream=true`. Both carry `X-Runtime`, `X-Sample-Rate` and "
                "`X-Dropped-Units`."
            ),
        },
        **_ERROR_RESPONSES,
    },
    summary="Synthesize Yiddish speech",
    description=(
        "Speaks Yiddish text (or IPA) as 16-bit mono PCM WAV at the serving "
        "runtime's sample rate (44 100 Hz for BlueTTS 2.5 — read "
        "`X-Sample-Rate`, do not assume). A `runtime` named in the body serves "
        "that one request and does not become the process default; only "
        "`POST /v1/models/load` changes what other callers see.\n\n"
        "**Pipeline.** Your text goes straight to the G2P, which walks the "
        "project's authority chain internally (native gold verdicts > corpus "
        "audio corrections > published pointing > model guesses) and returns "
        "IPA over the closed Yiddish inventory; that IPA is folded to the "
        "acoustic model's vocabulary and synthesized. The v5 pointing model is "
        "run only to *show* you the pointing (`/v1/audio/phonemize`, "
        "`/v1/audio/diacritize`); its guesses are never fed back into the G2P, "
        "because measured over real corpus text that changes 31% of words and "
        "systematically replaces gold and audio-confirmed readings with "
        "mechanical reads of a guess.\n\n"
        "`/v1/audio/phonemize` and the demo UI's `/generate` run this exact "
        "same pipeline, so all three produce the same phonemes — and therefore "
        "the same audio — for the same input.\n\n"
        "**Input forms.** `input_is_nikud=true` supplies pointed text: your "
        "marks are passed to the G2P verbatim and do change the reading (`מלך` "
        "-> `mˈajləx`, `מֶלֶךְ` -> `mˈɛləx`; `סוכה` -> `sˈixə`, `סוּכָּה` -> "
        "`sˈikə`), and the pointing model is skipped. `input_is_phonemes=true` "
        "supplies IPA and skips G2P entirely; that IPA must be inside the "
        "closed inventory or the request is a 400, since only a caller can be "
        "wrong about it. Unlike the MamboTTS server, phoneme input works with "
        "and without streaming.\n\n"
        "**Length and chunking.** `input` is capped at 4000 characters. Longer "
        "text than one utterance is split per sentence (200 characters, never "
        "through a multiword lexicon entry) and the pieces are joined with a "
        "60 ms gap — on **both** paths, so `stream` changes delivery and never "
        "the audio. The chunk budget does not scale with `speed`: the acoustic "
        "model's limit is on the length of the text, so `speed=0.5` renders the "
        "same chunk over twice as long, correctly. A single unsplittable run of "
        "words too long to render is a 400 naming the token count.\n\n"
        "**Load.** Synthesis is admission-controlled: when the box is already "
        "rendering its limit of concurrent requests and no slot frees up, the "
        "answer is 503 `not_available` rather than a machine that swaps.\n\n"
        "**Voices and sampler.** `voice` is validated against the serving "
        "runtime's `voices()` — an unknown name is a 400 listing the valid "
        "ones, never a silent substitution. `n_steps`, `cfg_scale` and `seed` "
        "are forwarded to runtimes that have a sampler (BlueTTS 2.5) and "
        "ignored by those that do not (Piper); `n_steps` is capped so one "
        "request cannot monopolise the CPU.\n\n"
        "**Dropped units.** Every response carries three headers: `X-Runtime`, "
        "`X-Sample-Rate`, and `X-Dropped-Units` — the **phones** the pipeline "
        "could not deliver to the voice (off-inventory units, plus anything the "
        "vocabulary fold had to drop). Punctuation is never listed: it is "
        "structure, not a phone, so the few characters the engine passes through "
        "that a vocabulary cannot spell are removed silently. It is a space-separated list of "
        "percent-encoded UTF-8 units, because header values are latin-1 and "
        "`ˈ`/`ʧ`/`ʤ` are not: `%CB%88` is `ˈ`. Empty means nothing was lost. On "
        "a stream the header is computed over the whole text before the first "
        "frame, since headers cannot be sent once the body has started.\n\n"
        "With `stream=true` the response is `application/octet-stream` carrying "
        "MamboTTS audio frames, byte-identical to the Rust contract: "
        "`[kind:u8][length:u32 big-endian][payload]`, where kind `1` is a "
        "self-contained playable WAV chunk, kind `2` is the complete "
        "concatenated WAV (sent last, for download/save), and kind `3` is UTF-8 "
        "error text if synthesis fails mid-stream.\n\n"
        "Only `response_format=\"wav\"` is supported."
    ),
)
async def speech(body: SpeechBody) -> Response:
    if not body.input.strip():
        return write_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            "request body must contain input",
        )
    if body.response_format and body.response_format != "wav":
        return write_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            "only wav response_format is supported",
        )
    try:
        rt = await run_in_threadpool(resolve_runtime, body.runtime)
    except LookupError as err:
        return write_error(status.HTTP_400_BAD_REQUEST, "invalid_request", str(err))
    except RuntimeNotAvailable as err:
        return write_error(
            status.HTTP_503_SERVICE_UNAVAILABLE, "not_available", str(err)
        )
    except Exception as err:  # noqa: BLE001
        logger.exception("failed to resolve runtime %r", body.runtime)
        return write_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            f"failed to load model: {err}",
        )

    # Only checkable once the serving runtime is known: which voices exist is a
    # property of the runtime, not of the catalog.
    try:
        check_voice(rt, body.voice)
    except LookupError as err:
        return write_error(status.HTTP_400_BAD_REQUEST, "invalid_request", str(err))

    form = input_form(body)
    options = sampler_options(body)

    # One pass over the whole text: it decides the phonemes, and (because
    # response headers must precede the body) it also decides the dropped-unit
    # report for the streaming case.
    try:
        full = await run_in_threadpool(
            analyze, body.input, form, with_tokens=False, with_nikud=False
        )
    except Exception as err:  # noqa: BLE001
        logger.exception("g2p failed")
        return write_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(err)
        )

    if not full.phonemes.strip():
        # The G2P quarantines digits, Latin text, URLs and lone geresh
        # abbreviations, so a request that is nothing but those legitimately
        # phonemizes to "". That is a caller error, not a pipeline failure: it
        # used to reach the runtime and come back as 500 internal_error
        # ("nothing to synthesize: the IPA string is empty") while /generate
        # answered 400 for the same input.
        return write_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            "no phonemes to synthesize: the engine read nothing speakable in "
            "this input (digits, Latin text, URLs and lone geresh "
            "abbreviations are quarantined, not read aloud)",
        )

    if form is InputForm.PHONEMES and full.unsupported:
        # Caller-supplied IPA is the one case where off-inventory units can only
        # be an error on the caller's side.
        return write_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_request",
            "phonemes outside the Yiddish inventory: " + " ".join(full.unsupported),
        )
    if full.unsupported:
        # Engine output is the authority on Yiddish; refusing to speak it would
        # turn a G2P quirk into an outage. Report it and speak.
        logger.warning(
            "engine emitted off-inventory units %s for %r",
            " ".join(full.unsupported), body.input[:80],
        )

    if body.stream:
        vocab_dropped = phones.fold_to_vocab(full.phonemes, rt.vocab())[1]
        report = dropped_report(full.unsupported, vocab_dropped)
        return StreamingResponse(
            _speech_frames(rt, body, form, options),
            media_type="application/octet-stream",
            headers=_speech_headers(rt, report, attachment=False),
        )

    # Chunked here too, not only on the stream: the acoustic model renders one
    # utterance per call, so an unchunked paragraph was a 500 with a duration
    # message on the default (stream=false) path.
    try:
        async with heavy_slot():
            samples, dropped = await run_in_threadpool(
                render_text, rt, body.input, form,
                voice=body.voice, speed=body.speed, options=options,
            )
    except Busy as err:
        return write_error(
            status.HTTP_503_SERVICE_UNAVAILABLE, "not_available", str(err)
        )
    except UtteranceTooLongError as err:
        # Only reachable for a single unsplittable run of words longer than the
        # checkpoint can render: a caller-side problem, so 400 with the
        # runtime's own message rather than 500 internal_error.
        return write_error(status.HTTP_400_BAD_REQUEST, "invalid_request", str(err))
    except Exception as err:  # noqa: BLE001
        logger.exception("synthesis failed")
        return write_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR, "internal_error", str(err)
        )
    return Response(
        content=audio.pcm16_wav(samples, rt.sample_rate),
        media_type="audio/wav",
        headers=_speech_headers(
            rt, dropped_report(full.unsupported, dropped), attachment=True
        ),
    )


async def _speech_frames(
    rt: Runtime,
    body: SpeechBody,
    form: InputForm,
    options: Mapping[str, object],
) -> AsyncIterator[bytes]:
    """Framed chunk stream, so playback can start before the whole text is spoken."""
    parts: list[np.ndarray] = []
    try:
        chunks = await run_in_threadpool(text_chunks, body.input, form)
        async with heavy_slot():
            for chunk in chunks:
                if not chunk.strip():
                    continue
                samples, _ = await run_in_threadpool(
                    render_chunk, rt, chunk, form,
                    voice=body.voice, speed=body.speed, options=options,
                )
                if samples.size == 0:
                    # A separator-only chunk, or one the G2P quarantined to
                    # nothing, yields no audio; an empty WAV frame can interrupt
                    # a client's queued playback, so skip it.
                    continue
                parts.append(samples)
                yield audio.frame(
                    audio.KIND_CHUNK, audio.pcm16_wav(samples, rt.sample_rate)
                )
        # The final frame carries the whole concatenated WAV, matching the Rust
        # contract: it is for download/save and does not delay playback, because
        # every chunk was already emitted above. Joined exactly the way the
        # non-streaming path joins them, so the two agree sample for sample.
        yield audio.frame(
            audio.KIND_FINAL,
            audio.pcm16_wav(audio.join_chunks(parts, rt.sample_rate), rt.sample_rate),
        )
    except Exception as err:  # noqa: BLE001 - the stream reports errors in-band
        logger.exception("streaming synthesis failed")
        yield audio.error_frame(str(err))
