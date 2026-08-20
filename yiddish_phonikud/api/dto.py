"""Pydantic v2 request/response models for the Yiddish Phonikud HTTP API.

Shapes are deliberately identical to the MamboTTS server DTOs
(`mambotts-server/src/server/{dto,errors,sources}.rs`) so an existing
MamboTTS client recognises every payload: same field names, same error
envelope, same code strings. Field descriptions are the Swagger copy —
`/docs` is a first-class deliverable of this port, so every field carries
one.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..registry import DEFAULT_RUNTIME_ID

# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

# Code strings borrowed verbatim from mambotts-server/src/server/errors.rs
# call sites, so clients can keep switching on them.
ERROR_CODES = (
    "invalid_request",
    "no_model",
    "not_available",
    "not_found",
    "method_not_allowed",
    "internal_error",
)


class ErrorDetail(BaseModel):
    """The inner half of the MamboTTS error envelope."""

    code: str = Field(
        ...,
        description=(
            "Machine-readable error code: `invalid_request` (bad or missing "
            "parameters, including request-body validation failures), "
            "`no_model` (no runtime loaded yet), `not_available` (the "
            "requested runtime exists in the catalog but cannot be used in "
            "this build/deployment), `not_found` (no such route), "
            "`method_not_allowed`, or `internal_error` (unexpected failure "
            "while diacritizing, phonemizing, or synthesizing)."
        ),
        examples=["invalid_request"],
    )
    message: str = Field(
        ...,
        description="Human-readable explanation, safe to show to an end user.",
        examples=["request body must contain input"],
    )


class ErrorBody(BaseModel):
    """Every non-2xx response from this API uses this envelope."""

    error: ErrorDetail = Field(..., description="The error code and message.")


# --------------------------------------------------------------------------
# Health / state
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness plus warm-up progress. Never returns a non-200 status."""

    status: str = Field(
        ...,
        description=(
            "Three states, and they are distinguishable — that is the point of "
            "this endpoint:\n\n"
            "* `warming` — the background warmup is still running (engine "
            "snapshot download, then the default runtime's own download and "
            "ONNX session build). The Space is alive; retry.\n"
            "* `ready` — the G2P engine **and** a TTS runtime are both in "
            "memory. Reachable on an idle Space, because the warmup loads the "
            "default runtime as well as the engine.\n"
            "* `error` — warmup finished and failed. `engine_error` or "
            "`runtime_error` carries the reason verbatim; this state never "
            "clears on its own, so alert on it instead of waiting."
        ),
        examples=["ready"],
    )
    engine_loaded: bool = Field(
        ...,
        description=(
            "True when the ~1.23 GB `notmax123/phonikud-yi-engine` snapshot has "
            "been downloaded and `yiddish_labels` imported. False on a cold "
            "start while the download is still running."
        ),
    )
    runtime_loaded: bool = Field(
        ...,
        description="True when a TTS runtime is loaded in memory.",
    )
    runtime: str = Field(
        ...,
        description="Id of the loaded runtime, or an empty string if none is loaded.",
        examples=[DEFAULT_RUNTIME_ID],
    )
    engine_error: str = Field(
        "",
        description=(
            "The G2P engine warmup failure, verbatim (`TypeName: message`), or "
            "an empty string. Non-empty means the label stack could not be "
            "loaded — a partial engine is a hard failure by design, because a "
            "missing lexicon table produces confidently wrong Yiddish."
        ),
    )
    runtime_error: str = Field(
        "",
        description=(
            "The TTS runtime warmup failure, verbatim, or an empty string. The "
            "engine can be healthy while the acoustic runtime is not; "
            "`/v1/audio/phonemize` still works in that state."
        ),
    )
    warming: bool = Field(
        False,
        description=(
            "True while the background warmup thread is still running. Lets a "
            "client tell `warming` from `error` without string-matching "
            "`status`."
        ),
    )
    version: str = Field(
        ..., description="Version of the `yiddish_phonikud` package serving this API."
    )


class StateResponse(BaseModel):
    """What the process currently holds in memory (port of `/v1/models`)."""

    loaded: bool = Field(..., description="Whether a TTS runtime is loaded.")
    runtime: str = Field(
        ..., description="Loaded runtime id, empty when nothing is loaded."
    )
    model: str = Field(
        ..., description="Human-readable name of the loaded acoustic model."
    )
    path: str = Field(
        ..., description="Filesystem path the model was loaded from, empty if unloaded."
    )
    sample_rate: int = Field(
        ...,
        description=(
            "Output sample rate in Hz of the loaded runtime (0 when unloaded). "
            "blue-yi renders at 44100 — read it here rather than hard-coding it."
        ),
        examples=[44100],
    )


# --------------------------------------------------------------------------
# Linguistics
# --------------------------------------------------------------------------


class TokenRowDTO(BaseModel):
    """One row of the G2P trace: how a single token was resolved.

    Multiword lexicon entries come back as a single row whose `word` holds the
    joined spelling, so do not assume one row per whitespace token.
    """

    word: str = Field(
        ..., description="The source token as spelled in the input.", examples=["צוריק"]
    )
    nikud: str = Field(
        ...,
        description=(
            "The same token, pointed: your own pointing when the request was "
            "flagged as pointed input, otherwise the v5 model's. **Display "
            "only** — it is not what produced `ipa`. Empty when the pointed "
            "text could not be aligned to this token."
        ),
        examples=["צוּרִיק"],
    )
    ipa: str = Field(
        ...,
        description="Primary IPA pronunciation, stress marked with `ˈ` before the stressed vowel.",
        examples=["ʦirˈik"],
    )
    route: str = Field(
        ...,
        description=(
            "Which of the engine's three routes answered, using the engine's "
            "own definitions (`yiddish_g2p.g2p_token`):\n\n"
            "* `lexicon` — a whole-token table hit: the abbreviation table, a "
            "multiword entry, the native-verdict gold lexicon, the legacy "
            "merged-loshn-koydesh / high-frequency / loan lists, or an "
            "audio-confirmed correction. A table decided **which entry "
            "applies**; where the token carries points, those points still "
            "steer the phones, so the pointed and unpointed spellings of one "
            "word can both come back `lexicon` with different readings (`מלך` "
            "-> `mˈajləx`, `מֶלֶךְ` -> `mˈɛləx`, both `lexicon`/`HIGH`).\n"
            "* `rule` — the Germanic or loshn-koydesh rule path decided it. "
            "This also covers the evidence-backed rescues that sit above the "
            "bare rules (audio-endorsed readings, homograph verdicts decided "
            "against episode audio, and readings taken from verified pointed "
            "editions) — read `reason` to tell them apart.\n"
            "* `fallback` — quarantine. The engine judged its own output unfit "
            "to emit (a vowel-less loshn-koydesh consonant string, an "
            "unlexiconed unpointed LK word, an out-of-inventory token such as a "
            "phone number or URL). `ipa` on such a row is a flagged "
            "approximation and is **excluded** from the sentence transcription "
            "— only the token's surrounding punctuation survives there."
        ),
        examples=["lexicon"],
    )
    confidence: str = Field(
        ...,
        description=(
            "`HIGH` a lexicon hit, `MED` an unambiguous rule application, `LOW` "
            "a defaulted ambiguous א/פ, an evidence-rescued or LK-fallback "
            "reading, or an inventory/shape violation. LOW is the "
            "human-verification queue, not an error."
        ),
        examples=["HIGH"],
    )
    layer: str = Field(
        ...,
        description=(
            "Lexical layer code, all six the engine emits: `G` Germanic, `L` "
            "loshn-koydesh (Hebrew/Aramaic component), `E` loanword, `N` proper "
            "name, `A` abbreviation/acronym, `X` empty/unclassified. `N` and `E` "
            "are not rare curiosities — a census over 18 967 corpus tokens found "
            "G 16 576, L 2 122, A 134, N 123, E 11, X 1, so any client "
            "switching on this field sees them on ordinary text (אמעריקע -> "
            "`E`, יוסף -> `N`)."
        ),
        examples=["G"],
    )
    reason: str = Field(
        ...,
        description=(
            "Short engine-supplied note naming the evidence or the defect: e.g. "
            "`sefaria-pointed`, `audio-homograph`, `pointed-audio-endorsed`, "
            "`audio-pe`, `alef-default`, `pe-default`, `lk-fallback`, "
            "`bad-phone`, `acronym-word`. Empty for a plain lexicon hit or a "
            "clean rule application."
        ),
    )


#: Longest `input` any endpoint accepts, in characters.
#:
#: 20000 used to be the documented figure and it was several times more than
#: this stack can serve: the acoustic model renders one ~10 s utterance per call,
#: so 20000 characters is ~100 chunks, ~17 minutes of audio and ~2.5 minutes of
#: CPU for ONE unauthenticated request, while the v5 pointing model's peak RSS
#: over a string that long measured 5.8 GB — enough to OOM-kill a 16 GB Space.
#: 4000 characters is ~3.5 minutes of speech, which is a generous paragraph and
#: still a request the box can finish. Raise it only together with the
#: concurrency limit in `routes_v1`.
MAX_INPUT_CHARS = 4000


class PhonemizeBody(BaseModel):
    """Text to run through the Yiddish diacritizer + G2P."""

    input: str = Field(
        ...,
        max_length=MAX_INPUT_CHARS,
        description=(
            "Yiddish text in Hebrew script. Unpointed Hasidic orthography is "
            "the normal case; partially or fully pointed text is accepted too "
            "and the pointing is **honoured**, never stripped (see "
            "`input_is_nikud`)."
        ),
        examples=["מיט א פאר יאר צוריק"],
    )
    input_is_nikud: bool = Field(
        False,
        description=(
            "Declare `input` as already-pointed text. The only effect is on the "
            "`nikud` field of the response and of every token row: your "
            "pointing is echoed back instead of the v5 model's, and the model "
            "is not run at all. It does **not** change how the pointing is "
            "treated by the G2P — pointing that is present in `input` is always "
            "read, with or without this flag."
        ),
    )
    input_is_phonemes: bool = Field(
        False,
        description=(
            "Declare `input` as IPA rather than Yiddish text. Nothing is "
            "diacritized and the G2P does not run: the string is normalised out "
            "of alternate notation (see `notation`) and checked against the "
            "closed inventory. This is the way to validate a hand-written "
            "transcription — which symbols are off-inventory, which are "
            "ambiguous, and whether stress is marked — without synthesizing "
            "anything. `tokens` is empty for this form, because there is no "
            "spelling to derive a per-word trace from."
        ),
    )


class PhonemizeResponse(BaseModel):
    """Full linguistic trace: pointing, IPA, per-token detail, and gaps."""

    nikud: str = Field(
        ...,
        description=(
            "The input after v5 diacritization, or your own pointing echoed "
            "back when `input_is_nikud` is set. **Display and hand-editing "
            "only**: `phonemes` is not derived from this string (see the "
            "endpoint description), so a difference between the two is not an "
            "inconsistency."
        ),
        examples=["מִיט אַ פּאָר יאָר צוּרִיק"],
    )
    phonemes: str = Field(
        ...,
        description=(
            "IPA transcription over the closed Yiddish inventory, produced by "
            "the G2P from `input` itself. This is exactly the string "
            "`/v1/audio/speech` would speak for the same request, and it is "
            "built in the same pass as `tokens` — the two can never contradict "
            "each other."
        ),
        examples=["mit a pˈur jur ʦirˈik"],
    )
    tokens: list[TokenRowDTO] = Field(
        default_factory=list,
        description=(
            "Per-token G2P trace, in input order, from the same pass as "
            "`phonemes`. Rows whose `route` is `fallback` are quarantined and "
            "their `ipa` does not appear in `phonemes`."
        ),
    )
    unsupported: list[str] = Field(
        default_factory=list,
        description=(
            "Phone units the engine emitted that are outside the closed Yiddish "
            "inventory, deduped and in order. Empty means a clean transcription; "
            "a non-empty list is a G2P bug worth reporting."
        ),
    )
    notation: list[NotationSubstitution] = Field(
        default_factory=list,
        description=(
            "**`input_is_phonemes` only.** Notation variants found in your IPA. "
            "A reviewer writing YIVO/Weinreich habitually types `mɪt`, `pʊr`, "
            "`tsɪrɪk` and ASCII `g`; those mean inventory phones spelled "
            "differently, so they are rewritten and reported here rather than "
            "refused. Empty for text and nikud input, whose IPA comes from the "
            "engine and is in the inventory by construction."
        ),
    )
    stress_warning: str | None = Field(
        None,
        description=(
            "**`input_is_phonemes` only.** Set when your IPA contains no `ˈ` at "
            "all and some word has more than one vowel — the utterance will be "
            "spoken flat. Where the stress falls is phonology (the engine "
            "derives it from the rule path and a lexical override table), so it "
            "cannot be supplied here; the warning names the words that lack it."
        ),
    )


class NotationSubstitution(BaseModel):
    """One rewrite applied to caller-supplied IPA, or one left for the caller."""

    source: str = Field(..., description="The symbol as it was typed.", examples=["ɪ", "o"])
    result: str = Field(
        ...,
        description=(
            "The inventory phone it became. When `ambiguous` is true this is "
            "the reading that was **assumed**, and `alternatives` lists the "
            "readings that were not chosen."
        ),
        examples=["i", ""],
    )
    applied: bool = Field(
        ...,
        description=(
            "True when the rewrite was made. Always true today; kept so a "
            "future rule that only reports has somewhere to say so."
        ),
    )
    ambiguous: bool = Field(
        ..., description="True when the symbol maps to more than one inventory phone."
    )
    alternatives: list[str] = Field(
        default_factory=list,
        description=(
            "For an ambiguous symbol, the inventory phones it could have meant. "
            "`o` is `ɔ` in גרויס (ɡrɔjs) but `u` in אוונט (uvnt) and `i` in "
            "אונז (inz) — which one is right depends on the word's historical "
            "vowel class, which the symbol does not carry. The most common "
            "reading is applied and reported here as an assumption; type the "
            "inventory symbol directly to override it, or use text input, where "
            "the engine's authority chain decides from the spelling."
        ),
        examples=[["ɔ", "u", "i"]],
    )


class DiacritizeResponse(BaseModel):
    """Pointing only, without running the G2P."""

    nikud: str = Field(
        ...,
        description="The input text with Yiddish diacritics applied.",
        examples=["מִיט אַ פּאָר יאָר צוּרִיק"],
    )


class PhonemeInventoryResponse(BaseModel):
    """The closed inventory, and what the loaded voice cannot say."""

    vowels: list[str] = Field(
        ..., description="Vowel and diphthong units of the closed Yiddish inventory."
    )
    consonants: list[str] = Field(
        ..., description="Consonant units of the closed Yiddish inventory."
    )
    marks: list[str] = Field(
        ...,
        description=(
            "Suprasegmental marks the G2P may emit. Exactly one: `ˈ`, primary "
            "stress, immediately before the stressed vowel. `ː` is **not** a "
            "member — length exists only inside the single inventory unit `aː`, "
            "so a bare `ː` is off-inventory and is reported in `unsupported` "
            "(`validate(\"ɔːbər\")` -> `[\"ː\"]`). Listing it here would make "
            "the inventory accept `ɛː`/`iː`/`uː`, which the spec forbids."
        ),
    )
    inventory: list[str] = Field(
        ...,
        description="Every unit the G2P may emit — vowels, consonants, and marks, sorted.",
    )
    runtime_vocab_missing: list[str] = Field(
        default_factory=list,
        description=(
            "Inventory units absent from the loaded runtime's phoneme "
            "vocabulary. Empty for BlueTTS 2.5, whose char vocab covers the "
            "whole closed inventory (`ʦ`, `ʧ`, `ʤ`, `ŋ`, `ˈ`, `ː` included); "
            "a voice lacking a unit would list it here, which "
            "are folded to `tʃ`/`dʒ` before synthesis. Also empty when no "
            "runtime is loaded — this field says nothing then."
        ),
    )


# --------------------------------------------------------------------------
# Speech
# --------------------------------------------------------------------------


class SpeechBody(BaseModel):
    """A synthesis request. Mirrors MamboTTS `SpeechBody` plus `speed`, the two
    input-form flags, and the optional Blue sampler knobs."""

    input: str = Field(
        ...,
        max_length=MAX_INPUT_CHARS,
        description=(
            "Text to speak. Unpointed Hasidic Yiddish by default; already-pointed "
            "Yiddish when `input_is_nikud` is set; IPA over the closed inventory "
            "when `input_is_phonemes` is set. Pointing present in the text is "
            "always read by the G2P — nothing is stripped."
        ),
        examples=["מיט א פאר יאר צוריק"],
    )
    runtime: str = Field(
        "",
        description=(
            "Runtime id to synthesize with. Empty uses the already-loaded "
            f"runtime, loading the default (`{DEFAULT_RUNTIME_ID}`) if none is loaded yet."
        ),
    )
    voice: str = Field(
        "",
        description=(
            "Voice name within the runtime. Empty picks the runtime's default "
            "voice. Validated against `GET /v1/voices` of the runtime that will "
            "actually serve the request: an unknown name is a 400 "
            "`invalid_request` listing the valid ones, never a silent fallback."
        ),
        examples=["female"],
    )
    response_format: str = Field(
        "wav",
        description="Audio container. Only `wav` is supported; anything else is rejected with 400.",
        examples=["wav"],
    )
    input_is_phonemes: bool = Field(
        False,
        description=(
            "Treat `input` as IPA and skip diacritization and G2P entirely. The "
            "IPA is validated against the closed inventory (off-inventory units "
            "are a 400, because they can only be a caller error) and then folded "
            "to the runtime vocabulary. Works for both streaming and non-streaming."
        ),
    )
    input_is_nikud: bool = Field(
        False,
        description=(
            "Declare `input` as already-pointed Yiddish. The pointing is passed "
            "to the G2P **verbatim** and changes the pronunciation — the engine "
            "reads points where they are present, and its Whole-Hebrew / merged "
            "loshn-koydesh registers and its פּ/פֿ check are all driven by them "
            "(so e.g. `מלך` reads `mˈajləx` unpointed but `מֶלֶךְ` reads "
            "`mˈɛləx`). The only other effect is that the v5 pointing model is "
            "not run at all. Ignored when `input_is_phonemes` is set."
        ),
    )
    speed: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description=(
            "Speaking rate multiplier; above 1.0 is faster. Bounded on both "
            "sides on purpose: a rate near zero would stretch a short request "
            "into minutes of synthesis."
        ),
        examples=[1.0],
    )
    stream: bool = Field(
        False,
        description=(
            "Stream framed audio instead of one WAV body. See the endpoint "
            "description for the frame protocol."
        ),
    )
    n_steps: int | None = Field(
        None,
        ge=1,
        le=32,
        description=(
            "Flow-matching sampler steps, for runtimes with a diffusion-style "
            "decoder (BlueTTS 2.5; default 8). More steps means marginally "
            "cleaner audio and linearly more CPU — measured on this hardware at "
            "~121 ms for 4 steps, ~178 ms for 8 and ~295 ms for 16 per 1.2 s of "
            "speech, doubled again whenever `cfg_scale` > 1. Capped at 32 so a "
            "single request cannot monopolise the box. Ignored by runtimes that "
            "have no sampler."
        ),
        examples=[8],
    )
    cfg_scale: float | None = Field(
        None,
        ge=1.0,
        le=8.0,
        description=(
            "Classifier-free guidance strength for BlueTTS 2.5 (default 4.0). "
            "`1.0` disables guidance and halves the work, because the "
            "unconditional pass is then skipped. Ignored by runtimes without "
            "guidance."
        ),
        examples=[4.0],
    )
    seed: int | None = Field(
        None,
        ge=0,
        le=2**31 - 1,
        description=(
            "Seed for the sampler's initial noise, for reproducible output from "
            "a stochastic runtime. Omit for a fresh draw each call. Ignored by "
            "deterministic runtimes."
        ),
        examples=[1234],
    )


# --------------------------------------------------------------------------
# Runtimes / catalog
# --------------------------------------------------------------------------


class LoadBody(BaseModel):
    """Which runtime to load. An empty body loads the default."""

    runtime: str = Field(
        DEFAULT_RUNTIME_ID,
        description="Runtime id from `GET /v1/models/sources`.",
        examples=[DEFAULT_RUNTIME_ID],
    )


class LoadResponse(BaseModel):
    """Result of a successful load."""

    status: str = Field(
        ..., description="Always `loaded` on success.", examples=["loaded"]
    )
    runtime: str = Field(..., description="Id of the runtime now loaded.")
    model: str = Field(..., description="Human-readable name of the loaded model.")


class ModelSourceFile(BaseModel):
    """One downloadable file of a runtime's install set."""

    name: str = Field(..., description="File name as it must appear in the runtime directory.")
    url: str = Field(
        ..., description="Download URL, empty when the file ships with the image."
    )


class RuntimeCapabilitiesResponse(BaseModel):
    """What a runtime can do, straight from its manifest."""

    yiddish: bool = Field(
        ..., description="Accepts Yiddish text/IPA (the Hebrew-script `yi` pipeline)."
    )
    streaming: bool = Field(..., description="Supports chunked framed streaming.")
    voice_reference: bool = Field(
        ..., description="Can clone a voice from a reference recording."
    )
    fixed_voices: bool = Field(
        ..., description="Ships a fixed set of named voices rather than cloning."
    )


class ModelSource(BaseModel):
    """A runtime as advertised by the catalog (port of `sources.rs`)."""

    id: str = Field(..., description="Runtime id used by `/v1/models/load`.")
    name: str = Field(..., description="Display name.")
    version: str = Field(..., description="Manifest version string.")
    size: str = Field(..., description="Approximate on-disk size of the install set.")
    description: str = Field(..., description="What this runtime is and its caveats.")
    files: list[ModelSourceFile] = Field(
        default_factory=list, description="Files required to install the runtime."
    )
    directory: str = Field(
        ..., description="Directory name the files are installed into, under the model root."
    )
    installed: bool = Field(
        ..., description="True when every required file is present on this machine."
    )
    available: bool = Field(
        ...,
        description=(
            "False for a runtime declared in the catalog but not implemented in "
            "this build; loading it returns 503 `not_available`."
        ),
    )
    capabilities: RuntimeCapabilitiesResponse = Field(
        ..., description="Capability flags for this runtime."
    )


class ModelSourcesResponse(BaseModel):
    """The runtime catalog plus where models live."""

    runtimes: list[ModelSource] = Field(
        default_factory=list, description="Every runtime known to the registry."
    )
    engine_repo: str = Field(
        ...,
        description="Hugging Face repo id of the Yiddish G2P/diacritizer engine.",
        examples=["notmax123/phonikud-yi-engine"],
    )
    default_paths: list[str] = Field(
        default_factory=list,
        description="Filesystem locations searched for installed runtime files.",
    )


class VoicesResponse(BaseModel):
    """Voices offered by the loaded runtime."""

    runtime: str = Field(..., description="Runtime the voices belong to.")
    voices: list[str] = Field(
        default_factory=list,
        description=(
            "Voice names accepted in `SpeechBody.voice`. An unknown name is "
            "rejected with 400 `invalid_request` rather than silently falling "
            "back to the default."
        ),
        examples=[["female", "libri_female_1088", "libri_male_6209"]],
    )


class LanguagesResponse(BaseModel):
    """Supported languages. This build is Yiddish-only."""

    languages: list[str] = Field(
        default_factory=list,
        description="Language codes accepted by the pipeline.",
        examples=[["yi"]],
    )
    items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Language records as `{\"code\", \"name\"}` pairs for UI menus.",
        examples=[[{"code": "yi", "name": "Yiddish (Hasidic)"}]],
    )
