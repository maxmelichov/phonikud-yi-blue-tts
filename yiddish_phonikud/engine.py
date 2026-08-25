"""Bridge to the published phonikud-yi engine (text -> nikud -> IPA).

The labels this module returns carry a fixed authority chain, highest first --
native gold verdicts > corpus audio corrections (PhoneticXeus over 900 episode
chunks) > published pointing (Sefaria) > model guesses. A lower tier never
overrides a higher one, and the bottom two tiers come back marked LOW
confidence for human review (verified: every 'sefaria-pointed' and every
'model-pointed-guess' record in a full-corpus census carried confidence=LOW).
That chain lives entirely in the engine's seven generated tables, which is why
a deployment missing a table is treated here as a hard failure rather than a
degraded mode.

Table sizes as deployed at ENGINE_REVISION, read out of ``verify(strict=False)``
rather than from prose -- GOLD_LEXICON 509, _AUDIO_PE 77, _AUDIO_VOWEL 120,
_AUDIO_ENDORSED 107, _HOMOGRAPH_LK 215, _SEFARIA_POINTED 3460,
_MODEL_POINTED 7633. The engine's own floors (``yiddish_labels._EXPECTED``) sit
at or below every one of those and ask for GOLD_LEXICON >= 502, so a "502 words"
figure was the floor the guard enforces, never the count the deployment holds.
(One floor has no slack at all: _AUDIO_PE's is 77 and the deployed table holds
exactly 77, so losing a single pe verdict trips the import guard.)
Call `info()` for the live numbers; do not restate them from memory.

Two calls, two very different costs, both measured on this CPU over 40 unseen
corpus rows of ~45 words: `text_to_ipa` 1.7 ms, `text_to_nikud` 296 ms -- the
v5 ONNX pointing model is ~176x the G2P and dominates any request that touches
it. `analyze()` exists so it runs exactly ONCE per request.

The engine is downloaded from Hugging Face at first use and imported LAZILY:
``yiddish_labels`` runs its deployment guard at import time, so it must not be
imported before the snapshot exists.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import unicodedata as ud
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .audio import MultiwordIndex, multiword_index
from .registry import ENGINE_REPO_ID, ENGINE_REVISION

log = logging.getLogger(__name__)

ENGINE_DIR_ENV = "PHONIKUD_YI_ENGINE_DIR"
# The revision lives in the registry: it is catalog metadata, not engine state.
__all__ = ["Analysis", "ENGINE_DIR_ENV", "ENGINE_REVISION", "EngineLoadError",
           "TokenRow", "analyze", "engine_dir", "info",
           "is_loaded", "load", "multiword", "multiword_keys", "text_to_ipa",
           "text_to_nikud", "token_table"]

_LOCK = threading.Lock()
_labels: ModuleType | None = None
_g2p: ModuleType | None = None
_dir: Path | None = None
_multiword_cache: MultiwordIndex | None = None


class EngineLoadError(RuntimeError):
    """The Yiddish label stack could not be loaded, or loaded incomplete."""


@dataclass(frozen=True)
class TokenRow:
    """One routed token: spelling, pointing, phonemes, and how it was decided.

    The verdict fields mean exactly what `g2p_token` in the engine says they
    mean. Shares below come from a route/confidence census over every 20th
    corpus row (89,053 tokens) -- measured, not paraphrased:

      route       'lexicon'  a table answered: gold, abbreviation, multiword or
                             the legacy loshn-koydesh list. 65.1% of tokens.
                  'rule'     no table knew the word, so the Germanic or the LK
                             rule path produced it. 34.5%.
                  'fallback' the output is NOT fit for emission and belongs in
                             quarantine -- a vowel-less LK consonant string, an
                             unlexiconed unpointed LK word, a phone number or a
                             URL. 0.4%. `hebrew_to_ipa` keeps only such a
                             token's punctuation and drops its phones.
      confidence  HIGH  a lexicon hit (63.9%).
                  MED   an unambiguous rule path, or a lexicon hit that rests
                        on an audio correction rather than a native verdict
                        (14.8%).
                  LOW   a defaulted ambiguous א/פ, a reading recovered from
                        Sefaria pointing or from the v5 model, a mined
                        multiword entry, an LK fallback, or a §1 shape
                        violation (21.3%).
      reason      the tag naming WHICH of those applied: 'alef-default',
                  'pe-default', 'sefaria-pointed', 'model-pointed-guess',
                  'audio-pe', 'audio-vowel', 'audio-homograph', 'mwe-mined',
                  'lk-fallback', 'bad-phone', 'prefix-rescue:gold+גע', ...
                  Empty for a plain lexicon hit (76.5% of tokens).

    LOW is the human-verification queue, not an error list: it marks the least
    certain readings in the stack, and they are still the engine's best answer.
    """

    word: str
    nikud: str
    ipa: str
    route: str
    confidence: str
    layer: str
    reason: str


def engine_dir() -> Path:
    """Where the engine lives locally, downloading it on first use.

    ``PHONIKUD_YI_ENGINE_DIR`` points at an already-unpacked bundle (e.g. the
    repo's ``dist/phonikud-yi-engine``) and suppresses the download entirely,
    so local development never touches the network. Otherwise the ~1.23 GB
    snapshot is fetched — that download is the slow part of a cold start.
    ``HF_HOME`` (cache location) and ``HF_TOKEN`` are honoured from the
    environment; the token is passed explicitly so a future private revision
    keeps working.
    """
    global _dir
    if _dir is not None:
        return _dir

    override = os.environ.get(ENGINE_DIR_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not (path / "yiddish_labels.py").is_file():
            raise EngineLoadError(
                f"{ENGINE_DIR_ENV}={override} does not contain yiddish_labels.py; "
                "point it at an unpacked phonikud-yi-engine bundle, or unset it to "
                f"download {ENGINE_REPO_ID} from Hugging Face."
            )
        _dir = path
        return _dir

    from huggingface_hub import snapshot_download  # imported late: network dependency

    # `local_dir` is load-bearing, not a preference. Without it the snapshot is a
    # tree of symlinks into the cache's blobs/ directory, and yiddish_labels
    # locates the engine with `Path(__file__).resolve().parent` -- resolve()
    # follows the symlink, so `_HERE` lands in blobs/ and its search for a
    # sibling data/ directory fails. The engine then refuses to load rather than
    # running without its tables, which is the guard behaving correctly on a
    # layout it cannot read. Materialising real files is the fix; it also keeps
    # onnxruntime's external-data lookup (onnx_yiddish_v5/model.onnx.data, 1.2 GB
    # beside a 220 KB graph) resolving next to a real file.
    target = _engine_local_dir()
    log.info("downloading engine %s@%s into %s (~1.23 GB on a cold cache)",
             ENGINE_REPO_ID, ENGINE_REVISION, target)
    snapshot_download(
        repo_id=ENGINE_REPO_ID,
        revision=ENGINE_REVISION,
        token=os.environ.get("HF_TOKEN"),
        local_dir=str(target),
    )
    _dir = target
    return _dir


def _engine_local_dir() -> Path:
    """Real-file destination for the engine snapshot, keyed by revision.

    Sits under HF_HOME when set (the container points that at a writable path
    owned by uid 1000) so the engine shares the cache volume everything else
    uses, and is revision-scoped so moving ENGINE_REVISION cannot land a new
    export on top of an old one.
    """
    root = os.environ.get("HF_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "huggingface"
    return (base / "phonikud-yi-engine" / ENGINE_REVISION).resolve()


def load() -> None:
    """Import the engine once. Idempotent; safe to call from every request."""
    global _labels, _g2p
    if _labels is not None and _g2p is not None:
        return
    with _LOCK:
        if _labels is not None and _g2p is not None:
            return
        path = engine_dir()

        # POSITION 0, not append: Phonikud-yi carries an older yiddish_nikud.py
        # aimed at a superseded export, and importing that one regenerates the
        # exact labels this stack replaces. yiddish_labels forces the same
        # ordering internally for the same reason; we must not hand it a
        # sys.path where a stale checkout already shadows the bundle.
        entry = str(path)
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)

        try:
            import yiddish_labels  # noqa: PLC0415 — must follow the download
            import yiddish_g2p  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            # yiddish_labels runs verify() at import and raises when any of the
            # 7 tables is missing. That is NOT downgraded to a warning: without
            # the tables the engine still emits plausible Yiddish — פעקל as
            # fɛkl instead of pɛkl, יארצייט as jˈarʦajt instead of jˈurʦajt —
            # with zero native verdicts and zero audio corrections, and says
            # nothing about it. A confidently wrong voice is the failure this
            # whole design exists to prevent, so it stays fatal.
            raise EngineLoadError(
                f"Yiddish label stack failed to load from {path}: {exc}\n\n"
                "If the message above names a data table, the deployment is "
                "incomplete rather than broken: yiddish_g2p swallows a missing "
                "table and returns {}, which silently drops every native verdict "
                "and audio correction (פעקל -> fɛkl instead of pɛkl, "
                "יארצייט -> jˈarʦajt instead of jˈurʦajt). Re-download "
                f"{ENGINE_REPO_ID} in full — a partial engine must not serve."
            ) from exc

        _labels, _g2p = yiddish_labels, yiddish_g2p
        log.info("engine loaded from %s", path)
        # וי policy + persisted ABE101 gold overlays sit on top of the frozen
        # snapshot so a Space restart does not lose native corrections, and so
        # the downloaded engine's old אויך=oʊx row cannot win.
        try:
            from . import lexicon_edits  # noqa: PLC0415 — after a successful load
            lexicon_edits.apply_to_engine(_g2p)
        except Exception as exc:  # noqa: BLE001 - overlay must not take down TTS
            log.exception("lexicon overlay failed: %s", exc)


def is_loaded() -> bool:
    return _labels is not None and _g2p is not None


def info() -> dict[str, Any]:
    """Engine identity plus per-table entry counts.

    The counts come from ``verify(strict=False)`` so /health and the UI can
    prove the deployment is *complete*, not merely running.
    """
    load()
    assert _labels is not None
    report = _labels.verify(strict=False)
    return {
        "repo": ENGINE_REPO_ID,
        "revision": ENGINE_REVISION,
        "dir": str(engine_dir()),
        "tables": dict(report.get("sizes", {})),
        "loaded": is_loaded(),
    }


def text_to_nikud(text: str) -> str:
    """Pointed Yiddish via phonikud-yi v5 (the ONNX session loads on first use)."""
    load()
    assert _labels is not None
    return _labels.text_to_nikud(text)


def text_to_ipa(text: str) -> str:
    """Phonemes for Hebrew-script Yiddish, through the full authority chain."""
    load()
    assert _labels is not None
    return _labels.text_to_ipa(text)


def _skeleton(text: str) -> str:
    """Base letters only — pointing and punctuation removed.

    Diacritics are the only thing the nikud model adds, so the skeleton is a
    stable key for matching a pointed token back to its G2P record.
    """
    return "".join(
        ch for ch in ud.normalize("NFD", text)
        if not ud.combining(ch) and (ch.isalpha() or ch.isdigit())
    )


def token_table(text: str, nikud: str | None = None) -> list[TokenRow]:
    """Routing records joined to the pointed form of the same text.

    ``g2p_tokens`` returns one record per token *except* for multiword lexicon
    entries, which come back as a single record whose ``word`` is the joined
    spelling and therefore consume as many pointed tokens as it has parts.
    Alignment is best-effort and verified by letter skeleton: where it does not
    match, the row's ``nikud`` is left empty and the walk resyncs. A
    confidently wrong pointing shown next to a word would be worse than a
    blank.

    Pass ``nikud`` when the caller has ALREADY pointed this exact text, and the
    v5 model is not run again -- that is the whole of the saving `analyze()`
    delivers (296 ms a request, measured). Leave it None and the pointing is
    fetched here, which is what every pre-existing caller keeps getting.
    """
    load()
    assert _g2p is not None
    records: list[dict[str, Any]] = _g2p.g2p_tokens(text)
    if nikud is not None:
        pointed = nikud.split()
    else:
        try:
            pointed = text_to_nikud(text).split()
        except Exception as exc:  # noqa: BLE001 — IPA is still useful without pointing
            log.warning(
                "pointing failed for token table, rows will have no nikud: %r", exc)
            pointed = []

    rows: list[TokenRow] = []
    cursor = 0
    for rec in records:
        word = str(rec.get("word") or "")
        parts = max(len(word.split()), 1)
        want = _skeleton(
            f"{rec.get('lead') or ''}{word}{rec.get('trail') or ''}"
        )

        nikud = ""
        take = pointed[cursor:cursor + parts]
        if len(take) == parts and _skeleton(" ".join(take)) == want:
            nikud = " ".join(take)
            cursor += parts
        else:
            # Resync: the pointer may have gained or lost a token (punctuation
            # that the model attached differently). Scan a short window for the
            # skeleton we expect rather than dragging the offset through the
            # rest of the table.
            for offset in range(1, 4):
                probe = pointed[cursor + offset:cursor + offset + parts]
                if len(probe) == parts and _skeleton(" ".join(probe)) == want:
                    nikud = " ".join(probe)
                    cursor += offset + parts
                    break
            else:
                cursor += parts

        rows.append(
            TokenRow(
                word=word,
                nikud=nikud,
                ipa=str(rec.get("ipa_primary") or ""),
                route=str(rec.get("route") or ""),
                confidence=str(rec.get("confidence") or ""),
                layer=str(rec.get("layer") or ""),
                reason=str(rec.get("reason") or ""),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# One pointing pass per request (the C10 bridge)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    """Everything a route needs about one piece of text, from ONE pointing pass.

    ``ipa`` is `text_to_ipa` applied to the RAW text, which is the only correct
    input for it: `hebrew_to_ipa` runs the full authority chain itself (gold
    lexicon, then audio corrections, then published pointing, then -- only where
    nothing else knows the word -- the v5 model's guess, reached through the
    engine's own `_MODEL_POINTED` table). Feeding `nikud` back in as text would
    let a tier-4 guess re-decide readings a higher tier had already settled.
    ``nikud`` is for DISPLAY and for the token table's second column.

    ``tokens`` is aligned to ``nikud``: same pointing string, so the table can
    never contradict the phonemes shown beside it.
    """

    text: str
    nikud: str
    ipa: str
    tokens: list[TokenRow]


def analyze(text: str, *, tokens: bool = True) -> Analysis:
    """Nikud + IPA + the token table, with the v5 model running exactly once.

    THIS is the function routes should call. The old shape -- `text_to_nikud`
    for the display string and `token_table` for the table -- pointed the same
    text twice, and pointing is 296 ms against the G2P's 1.7 ms (measured over
    40 unseen ~45-word corpus rows), so the duplicate was most of the request.
    One `analyze()` halves it.

    Pointing is best-effort: if the ONNX session fails, ``nikud`` comes back
    empty and the IPA and the routing table are still returned in full, because
    the G2P does not depend on the pointing model.

    ``tokens=False`` skips the per-token table for callers that only need the
    two strings; the pointing pass still happens once, never twice.
    """
    load()
    try:
        nikud = text_to_nikud(text)
    except Exception as exc:  # noqa: BLE001 — the G2P does not need the pointing
        log.warning("pointing failed, returning IPA without nikud: %r", exc)
        nikud = ""
    return Analysis(
        text=text,
        nikud=nikud,
        ipa=text_to_ipa(text),
        tokens=token_table(text, nikud=nikud) if tokens else [],
    )


# ---------------------------------------------------------------------------
# Multiword bridge for the chunker
# ---------------------------------------------------------------------------


def multiword_keys() -> list[str]:
    """The engine's multiword lexicon keys, exactly as it holds them.

    Both tables the router consults: ``_MULTIWORD`` (the curated §8 table plus
    the mined additions) and ``_MULTIWORD_LEGACY`` (the space-containing keys of
    the legacy loshn-koydesh list). 83 keys at ENGINE_REVISION, of which 43 are
    whitespace-spelled -- the other 40 are the maqaf-joined spellings the engine
    generates from the same entries, single whitespace tokens that no
    whitespace split can break. Longest entry: 3 words.
    """
    load()
    assert _g2p is not None
    return sorted(set(_g2p._MULTIWORD) | set(_g2p._MULTIWORD_LEGACY))


def multiword() -> MultiwordIndex:
    """The chunker's no-split index, built from `multiword_keys()` and cached.

    `audio.chunk_text` must not split a whitespace-spelled collocation across
    two chunks: the engine matches those over the token stream, so a split makes
    stream=true mispronounce (``א פאר יאר`` -> "a far jur") what stream=false
    gets right (``a pˈur jur``). Pass the result as
    ``chunk_text(text, max_chars, multiword=engine.multiword())``.

    The index itself holds no engine reference and costs nothing to use, so
    `audio` stays importable and testable without the 1.23 GB bundle; this
    function is the only place the two meet.
    """
    global _multiword_cache
    if _multiword_cache is None:
        _multiword_cache = multiword_index(multiword_keys())
    return _multiword_cache
