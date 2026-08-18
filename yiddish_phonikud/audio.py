"""WAV packing, streaming text chunking, and the MamboTTS binary frame format.

These three concerns live together because they are all about getting audio out
of the synthesiser and onto the wire: `chunk_text` decides what a streamed
chunk is (and refuses the boundaries that would change how the G2P reads the
text -- see the long note above it), `pcm16_wav` turns a chunk into a self-contained playable file, and
`frame` wraps it in the length-prefixed protocol the MamboTTS clients speak.
"""

from __future__ import annotations

import struct
import unicodedata as ud
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

__all__ = [
    "EMPTY_MULTIWORD",
    "MAX_CHUNK_CHARS",
    "CHUNK_GAP_SECONDS",
    "join_chunks",
    "KIND_CHUNK",
    "KIND_ERROR",
    "KIND_FINAL",
    "STREAM_MEDIA_TYPE",
    "MultiwordIndex",
    "chunk_text",
    "error_frame",
    "frame",
    "multiword_index",
    "pcm16_wav",
]

# ---------------------------------------------------------------------------
# WAV
# ---------------------------------------------------------------------------

_BITS_PER_SAMPLE: Final = 16
_CHANNELS: Final = 1
_PCM_FORMAT: Final = 1  # WAVE_FORMAT_PCM
_HEADER_BYTES: Final = 44

# int16 spans [-32768, 32767]; scaling by 32767 keeps +1.0 representable and
# makes the mapping symmetric, which is what the reference packer does.
_FULL_SCALE: Final = 32767.0


#: Character budget for one synthesis chunk, and the default for `chunk_text`.
#:
#: Derived from the acoustic model, not chosen for taste. BlueTTS 2.5 refuses
#: text longer than `blue_yi.MAX_TEXT_TOKENS` (240) tokens — the point past
#: which its duration predictor stops lengthening the utterance and starts
#: cramming — and its encoder adds three tokens to the character count (a
#: trailing "." plus one padding space at each end). `chunk_text` may also
#: overrun its budget by up to `max_words - 1` words rather than cut through a
#: multiword lexicon entry, so the budget sits far enough below 237 to absorb
#: that: 200 characters leaves room for a 36-character collocation tail.
#:
#: Note what this budget is NOT a function of: `speed`. The cap it serves is on
#: the length of the TEXT, so `speed=0.5` needs no smaller chunk — it renders
#: the same 200 characters over twice as long, which BlueTTS does correctly
#: (T_lat is symbolic in every graph). A budget scaled by speed would refuse
#: renders that work.
MAX_CHUNK_CHARS: Final = 200

#: Silence inserted between concatenated chunks (RECIPE G13). Each chunk is
#: rendered as its own utterance and carries its own utterance-final fall, so
#: butting two of them together reads as a rushed elision; 60 ms is shorter than
#: any real Yiddish pause and long enough to stop the seam sounding like a
#: dropped syllable.
CHUNK_GAP_SECONDS: Final = 0.06


def join_chunks(
    parts: Sequence[np.ndarray],
    sample_rate: int,
    gap_seconds: float = CHUNK_GAP_SECONDS,
) -> np.ndarray:
    """Concatenate per-chunk waveforms into one utterance, with a silence gap.

    Returns an empty float32 array when there is nothing to join, so callers can
    hand the result straight to `pcm16_wav`.
    """
    usable = [np.asarray(part, dtype=np.float32).reshape(-1) for part in parts]
    usable = [part for part in usable if part.size]
    if not usable:
        return np.zeros(0, dtype=np.float32)
    if len(usable) == 1:
        return usable[0]
    gap = np.zeros(max(int(round(gap_seconds * sample_rate)), 0), dtype=np.float32)
    joined: list[np.ndarray] = []
    for i, part in enumerate(usable):
        if i and gap.size:
            joined.append(gap)
        joined.append(part)
    return np.concatenate(joined)


def pcm16_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    """Pack mono float samples into a 16-bit PCM WAV (44-byte RIFF header).

    Vectorised on purpose: the reference Python packer struct-packs one sample
    at a time, which costs seconds per utterance at 22 kHz.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    # Clip before scaling: unclipped peaks would wrap around in int16 and
    # render as loud clicks rather than as harmless clipping.
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * _FULL_SCALE).astype("<i2", copy=False)
    payload = pcm.tobytes()

    byte_rate = sample_rate * _CHANNELS * _BITS_PER_SAMPLE // 8
    block_align = _CHANNELS * _BITS_PER_SAMPLE // 8
    header = b"".join(
        (
            b"RIFF",
            # RIFF size counts everything after this field: the 36 remaining
            # header bytes plus the payload.
            struct.pack("<I", _HEADER_BYTES - 8 + len(payload)),
            b"WAVE",
            b"fmt ",
            struct.pack(
                "<IHHIIHH",
                16,  # fmt chunk size for PCM
                _PCM_FORMAT,
                _CHANNELS,
                sample_rate,
                byte_rate,
                block_align,
                _BITS_PER_SAMPLE,
            ),
            b"data",
            struct.pack("<I", len(payload)),
        )
    )
    return header + payload


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------
#
# Two things make a chunk boundary WRONG rather than merely unlucky, and both
# were found by comparing stream=true against stream=false on real text. The
# second one is measured: chunking 1,500 corpus rows at max_chars=60 and
# re-phonemizing each chunk on its own, 21 of 13,623 chunks came back carrying
# IPA that does not occur anywhere in the whole-line IPA. Passing the multiword
# index in: 0 of 13,625.
#
#  1. Geresh ׳ (U+05F3) and gershayim ״ (U+05F4) are not sentence marks. In
#     Yiddish they mark abbreviations and acronyms -- ר׳ יאנקעלע (Reb Yankele),
#     ביהמ״ד (beis-hamedresh), שליט״א. Treating them as sentence-final put the
#     STRONGEST available split between a title and the name it introduces.
#     Gershayim is word-internal, so it was never a split risk once it stopped
#     being a boundary; geresh usually ends its token (ר׳), so it additionally
#     glues that token to the next one (`_ABBREV_MARKS`).
#     Sof pasuq ׃ (U+05C3) stays: it really is the Hebrew full stop, and the
#     engine passes it through into the IPA. Paseq ׀ (U+05C0) is gone from every
#     set -- it is a disjunction mark inside a verse, not a sentence end, and it
#     is not in the engine's edge-punctuation class either.
#
#  2. A chunk boundary must not fall inside a multiword lexicon entry. The
#     engine matches whitespace-spelled collocations ACROSS the token stream
#     (`_multiword_match` / `g2p_tokens` in yiddish_g2p.py): א פאר יאר is one
#     record reading "a pˈur jur", בעל הבית is one record reading "bˈaləbus".
#     Split the span and each half routes on its own, so stream=true says
#     "a far jur" where stream=false says "a pˈur jur" -- the exact dialect
#     error the multiword table exists to prevent.
#
# HOW THE MULTIWORD GUARD IS WIRED (and what it costs)
#     `chunk_text` takes an optional `multiword=` index and knows nothing about
#     the engine: no import, no 1.23 GB bundle, no ONNX session. The entry list
#     is CONSULTED from the engine rather than hardcoded -- `engine.multiword()`
#     builds the index from the engine's own `_MULTIWORD` / `_MULTIWORD_LEGACY`
#     tables and caches it -- and the caller passes it in. Callers that already
#     have the engine loaded (every synthesis path does, by definition) get the
#     guard for free; `chunk_text(text)` alone stays pure and instant.
#     Cost: building the index folds 83 engine keys down to the 43
#     whitespace-spelled spans; using it is at most `max_words` (= 3) set probes
#     per word. Measured end to end: 0.14-0.18 ms to chunk a corpus row at
#     `MAX_CHUNK_CHARS`, index included.
#
#     LIMITS, stated plainly:
#       * `_affixes` + `_fold_core` reimplement the subset of `split_affixes`
#         and `lexicon_key` that matters for whitespace keys -- the same
#         three-way punctuation split, NFC, dash unification, nikud removal,
#         final-letter and ligature folding. The engine's own tables are the
#         source for the first three, kept assertable by the equalities noted
#         beside `_FINAL_FOLD` below. They do NOT reproduce the clitic
#         handling, so a collocation written with a leading clitic folds to a
#         word the index does not know and the guard does not fire there. The
#         failure mode there is the pre-existing one, not a new one.
#       * Punctuation between members blocks the engine's match, so it blocks
#         this one: "א בעל. הבית" is two sentences, not a collocation, and the
#         boundary stays available (`opaque` in `_glued`).
#       * `max_chars` is an upper bound EXCEPT when a single multiword entry
#         plus its window offers no legal cut; then the chunk is extended by up
#         to `max_words - 1` words rather than split inside the entry. A
#         mispronounced collocation is worse than a slightly long chunk.

# Sentence-final marks, including the Hebrew-script sof pasuq ׃ that appears in
# Yiddish typography. Verified against the engine: every one of these survives
# `hebrew_to_ipa` into the phoneme string, so each is a real prosodic break.
_SENTENCE_MARKS: Final = frozenset(".!?:;…׃")

# Clause-level breaks. Weaker than a sentence end, still a prosodic pause.
_CLAUSE_MARKS: Final = frozenset(",،")

# Abbreviation marks. A token ending in one of these opens an abbreviation and
# belongs with the token that follows it (ר׳ + name, ה׳ + divine name), so no
# chunk boundary is allowed after it. `normalize_surface` maps every one of
# these to ASCII ' before the engine sees it, which is why they are listed
# together.
_ABBREV_MARKS: Final = frozenset("׳'’ʼ`")

# Trailing closers that sit *after* the real punctuation, e.g. `.")`.
_CLOSERS: Final = frozenset(")]}»”’\"'")

_LEVEL_SENTENCE: Final = 2
_LEVEL_CLAUSE: Final = 1
_LEVEL_WORD: Final = 0

# NOTE: the maqaf ־ (U+05BE) is deliberately absent from every mark set above,
# and splitting only ever happens at whitespace, so a maqaf can never become a
# split point. בעל־הבית is a single lexical item to the G2P — its lexicon and
# stress lookups key on the whole hyphenated form, so cutting at the maqaf
# would silently downgrade a HIGH-confidence lexicon hit to two fallbacks.

# Mirrors of the engine constants a multiword key is compared through. Copied,
# not imported: this module must stay engine-free (no 1.23 GB bundle on
# sys.path, no ONNX session) so the chunker is testable on its own. All four
# equalities hold today and are cheap to assert in a test:
#   _FINAL_FOLD    == yiddish_g2p._FINAL_FOLD_TABLE
#   _LIGATURE_FOLD == yiddish_g2p._LIGATURE_FOLD_TABLE
#   _EDGE_PUNCT    == yiddish_g2p._EDGE_PUNCT
#   _DASHES        == the character class of yiddish_g2p._DASH_CHARS
_FINAL_FOLD: Final = {0x05DA: "כ", 0x05DD: "מ", 0x05DF: "נ", 0x05E3: "פ",
                      0x05E5: "צ"}
_LIGATURE_FOLD: Final = {0x05F2: "יי", 0x05F1: "וי", 0x05F0: "וו"}
_DASHES: Final = "־‐‑‒–—"
_EDGE_PUNCT: Final = " \t\n\r.,!?;:()[]{}<>«»„‚‹›…׃״“”\"*/\\|-"


def _affixes(word: str) -> tuple[str, str, str]:
    """(lead punctuation, core, trail punctuation) -- mirrors `split_affixes`.

    Same character class, same three-way split, so the chunker sees a token the
    way the engine's router sees it. A token that is nothing but punctuation
    comes back with an empty core, which is how a boundary-blocking full stop
    stays visible below.
    """
    core = word.strip(_EDGE_PUNCT)
    if not core:
        return "", "", word
    start = word.index(core)
    return word[:start], core, word[start + len(core):]


def _fold_core(core: str) -> str:
    """A token core reduced to the form the engine's lexicon keys use.

    Mirrors `yiddish_g2p.lexicon_key`: NFC, dash unification, nikud removal,
    final-letter fold, ligature fold. That is the whole of what a
    whitespace-separated multiword key is compared against.
    """
    text = ud.normalize("NFC", core)
    for dash in _DASHES:
        text = text.replace(dash, "-")
    text = ud.normalize(
        "NFC",
        "".join(ch for ch in ud.normalize("NFD", text)
                if ud.category(ch) != "Mn"),
    )
    return text.translate(_FINAL_FOLD).translate(_LIGATURE_FOLD)


@dataclass(frozen=True)
class MultiwordIndex:
    """Whitespace-spelled lexicon entries the chunker must not cut through.

    `spans` holds each entry as a tuple of folded words; `starts` is the set of
    folded first words, so the common case (a word that opens nothing) costs one
    set lookup. Build one with `multiword_index`.
    """

    spans: frozenset[tuple[str, ...]]
    starts: frozenset[str]
    max_words: int

    def match_len(self, folded: Sequence[str], i: int, limit: int) -> int:
        """Words in the longest entry starting at `folded[i]`, else 1.

        `limit` caps the span the way punctuation caps the engine's: no match
        may reach past a boundary where one token's trail or the next token's
        lead punctuation sits.
        """
        if folded[i] not in self.starts:
            return 1
        for n in range(min(self.max_words, limit, len(folded) - i), 1, -1):
            span = tuple(folded[i:i + n])
            if "" in span:  # a punctuation-only token is not a member
                continue
            if span in self.spans:
                return n
        return 1


EMPTY_MULTIWORD: Final = MultiwordIndex(frozenset(), frozenset(), 1)


def multiword_index(keys: Iterable[str]) -> MultiwordIndex:
    """Build a `MultiwordIndex` from engine lexicon keys.

    Keys with no space (the maqaf-joined spellings, which are single whitespace
    tokens and can never be split) are ignored; the rest are folded through
    `_fold_core` so they match tokens taken straight out of user text.
    """
    spans = set()
    for key in keys:
        parts = tuple(_fold_core(part) for part in key.split())
        if len(parts) > 1 and all(parts):
            spans.add(parts)
    if not spans:
        return EMPTY_MULTIWORD
    return MultiwordIndex(
        spans=frozenset(spans),
        starts=frozenset(span[0] for span in spans),
        max_words=max(len(span) for span in spans),
    )


def _break_level(word: str) -> int:
    """Prosodic strength of the boundary *after* `word`."""
    tail = word.rstrip("".join(_CLOSERS))
    if not tail:
        return _LEVEL_WORD
    last = tail[-1]
    if last in _SENTENCE_MARKS:
        return _LEVEL_SENTENCE
    if last in _CLAUSE_MARKS:
        return _LEVEL_CLAUSE
    return _LEVEL_WORD


def _glued(words: list[str], multiword: MultiwordIndex) -> list[bool]:
    """`out[i]` is True when no chunk boundary may fall after `words[i]`.

    Two sources: a multiword lexicon entry spanning the boundary, and an
    abbreviation mark that leaves its token expecting the next one.
    """
    total = len(words)
    glued = [False] * total
    closers = "".join(_CLOSERS)
    for i, word in enumerate(words[:-1]):
        # Both spellings: the geresh itself, and the ASCII ' a normaliser may
        # already have produced. `_CLOSERS` holds ' and ’ too, so strip closers
        # only as a second chance rather than before the test.
        stripped = word.rstrip(closers)
        if word[-1] in _ABBREV_MARKS or (stripped and stripped[-1] in _ABBREV_MARKS):
            glued[i] = True
    if multiword.max_words > 1:
        split = [_affixes(word) for word in words]
        folded = [_fold_core(core) for _, core, _ in split]
        # Punctuation between two members blocks the engine's match
        # (`_multiword_match` rejects a span whose inner tokens carry trail, or
        # whose later tokens carry lead, punctuation), so it must block ours: a
        # writer who put a full stop inside "א בעל. הבית" did not write that
        # collocation, and the boundary there stays available.
        opaque = [
            bool(split[i][2]) or bool(split[i + 1][0]) for i in range(total - 1)
        ]
        for i in range(total):
            # How far a span may reach from here: never past an opaque boundary,
            # and never more than the longest entry in the table (so this stays
            # O(n * max_words), not O(n^2), on punctuation-free text).
            limit = 1
            while (limit < multiword.max_words
                   and i + limit - 1 < total - 1
                   and not opaque[i + limit - 1]):
                limit += 1
            # A span of n words forbids the n-1 boundaries inside it. The last
            # boundary of the span (after words[i + n - 1]) stays legal.
            for j in range(i, i + multiword.match_len(folded, i, limit) - 1):
                glued[j] = True
    return glued


def chunk_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    multiword: MultiwordIndex | None = None,
) -> list[str]:
    """Split `text` into streamable chunks of at most `max_chars` characters.

    Words (whitespace-delimited, so maqaf-joined compounds stay intact) are
    packed greedily; the cut point inside each filled window is the strongest
    available boundary that is LEGAL — sentence-final punctuation, then clause
    commas, then plain whitespace. Punctuation stays attached to the chunk it
    ends.

    A boundary is illegal when it would fall inside a multiword lexicon entry
    (pass `multiword=engine.multiword()`, which is what the synthesis paths do)
    or straight after an abbreviation mark such as the geresh of ר׳. When no
    legal boundary exists inside the window the chunk is extended to the first
    legal one instead, so `max_chars` is exceeded by at most a collocation's
    width rather than a collocation being cut in half.

    No-loss invariant: ``"".join(chunk_text(t))`` contains every non-whitespace
    character of ``t``, in order. It holds because the only transformation
    applied is whitespace normalisation via `str.split`; words themselves are
    never edited, reordered, or dropped.
    """
    words = text.split()
    if not words:
        return []
    if max_chars <= 0:  # no limit requested: one chunk, still whitespace-normalised
        return [" ".join(words)]

    levels = [_break_level(word) for word in words]
    glued = _glued(words, multiword or EMPTY_MULTIWORD)
    total = len(words)
    chunks: list[str] = []
    start = 0

    while start < total:
        # Widest window of whole words that fits, always at least one word so a
        # single overlong word is emitted intact rather than cut mid-word.
        end = start
        length = 0
        while end < total:
            extra = len(words[end]) + (1 if end > start else 0)
            if end > start and length + extra > max_chars:
                break
            length += extra
            end += 1

        cut = end
        if end < total:
            # Retreat to the strongest LEGAL boundary inside the window.
            for level in (_LEVEL_SENTENCE, _LEVEL_CLAUSE):
                candidate = next(
                    (i for i in range(end - 1, start - 1, -1)
                     if levels[i] == level and not glued[i]),
                    None,
                )
                if candidate is not None:
                    cut = candidate + 1
                    break
            else:
                # No punctuation boundary: fall back to whitespace, walking back
                # from the window edge until the boundary is legal.
                cut = next(
                    (i + 1 for i in range(end - 1, start - 1, -1) if not glued[i]),
                    None,
                )
                if cut is None:
                    # Everything in the window is glued: extend forward to the
                    # first legal boundary. `glued[total - 1]` is never set, so
                    # this always terminates.
                    cut = next(
                        i + 1 for i in range(end, total) if not glued[i]
                    )

        chunks.append(" ".join(words[start:cut]))
        start = cut

    return chunks


# ---------------------------------------------------------------------------
# Framed stream
# ---------------------------------------------------------------------------

KIND_CHUNK: Final = 1  # a self-contained playable WAV for one text chunk
KIND_FINAL: Final = 2  # the complete concatenated WAV, for download/save
KIND_ERROR: Final = 3  # UTF-8 error text; terminates the stream

STREAM_MEDIA_TYPE: Final = "application/octet-stream"

# 1 byte kind + 4 bytes big-endian length. ">BI" would be padding-free here,
# but ">B" then ">I" is explicit about the 5-byte layout and matches the Rust
# side byte for byte (`out.push(kind); out.extend(len.to_be_bytes())`).
_KIND_STRUCT: Final = struct.Struct(">B")
_LEN_STRUCT: Final = struct.Struct(">I")
FRAME_HEADER_BYTES: Final = _KIND_STRUCT.size + _LEN_STRUCT.size  # == 5

_MAX_PAYLOAD: Final = 0xFFFF_FFFF


def frame(kind: int, payload: bytes) -> bytes:
    """Encode one stream frame: ``[kind:u8][len:u32 big-endian][payload]``."""
    if not 0 <= kind <= 0xFF:
        raise ValueError(f"frame kind must fit in a u8, got {kind}")
    if len(payload) > _MAX_PAYLOAD:
        raise ValueError(f"frame payload exceeds u32 length: {len(payload)} bytes")
    return _KIND_STRUCT.pack(kind) + _LEN_STRUCT.pack(len(payload)) + payload


def error_frame(message: str) -> bytes:
    """Terminal error frame carrying UTF-8 text."""
    return frame(KIND_ERROR, message.encode("utf-8"))
