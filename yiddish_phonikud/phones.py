"""The closed Yiddish phone inventory: segmentation, validation, vocab folding.

This module is the single gatekeeper between the G2P engine's output and any
acoustic model. The G2P spec (`data/spec/g2p_spec_v3.md` §1) declares a CLOSED
set of phones; QA gate (b) of the corpus pipeline enforces that nothing else is
ever emitted. Here we enforce the other half of the contract: a phone that is
in the inventory but absent from a voice's vocabulary must be rewritten into
something the voice can actually say, or reported as dropped. Silence is never
an acceptable answer to an unsupported phone -- a visible warning is.

WHAT THE ENGINE ACTUALLY EMITS (measured, not assumed)
    ``hebrew_to_ipa`` does not return phones alone. ``g2p_tokens`` hands every
    record a ``lead``/``trail`` pair holding the punctuation that sat around the
    token, and ``hebrew_to_ipa`` rebuilds the line with it -- so a legitimate
    IPA string carries spaces, commas, full stops, brackets and quotation marks
    interleaved with phones. Multiword lexicon entries add two more: a space
    (``א פאר`` -> ``a pˈur``) and a hyphen (``בית מדרש`` -> ``bis-mˈɛdrəʃ``).
    ``validate()`` therefore has to tell "not a phone" apart from "not allowed",
    and the boundary between them was derived by running the engine rather than
    guessed -- see `_NON_PHONE` below for the measurement.

UNICODE TRAP -- this has bitten this codebase before:
    ɡ is U+0261 LATIN SMALL LETTER SCRIPT G, *not* ASCII "g" (U+0067).
    ʦ is U+02A6, ʧ is U+02A7, ʤ is U+02A4 (single-codepoint affricates,
    not the two-character sequences ts / tʃ / dʒ).
Copy these characters, never retype them; `"ɡ" == "g"` is False and a mixed-up
script-g silently turns every /ɡ/ in the corpus into an unknown phone.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The inventory (g2p_spec_v3 §1 / docs/yiddish_phoneme_set.md). Closed set.
# ---------------------------------------------------------------------------

VOWELS: tuple[str, ...] = (
    "a",
    "aː",   # class 34 flattened ay (haːnt); the ONLY place ː ever appears
    "ɛ",
    "ə",    # every unstressed ɛ
    "i",
    "u",
    "ɔ",
    "ej",
    "aj",
    "ɔj",
    "oʊ",   # class-54 vov-yud, lexical list (hoʊz)
)

CONSONANTS: tuple[str, ...] = (
    "b", "d", "f", "ɡ", "h", "j", "k", "l", "m", "n", "p", "r",
    "s", "t", "v", "z", "x", "ʃ", "ʒ", "ʦ", "ʧ", "ʤ", "ŋ",
)
# ŋ is declared by §1 but is UNATTESTED in engine output: a census of
# hebrew_to_ipa over all 23,666 corpus rows (both the raw-text and the pointed
# column) produced zero ŋ -- the letter table writes נג as n + ɡ (זינגען ->
# zˈinɡən, לאנג -> laːnɡ). It stays in the inventory because the spec declares
# it and a voice must still be able to say it; do not read its presence here as
# evidence the G2P emits it.

# ˈ sits immediately BEFORE the stressed vowel, at most one per word.
#
# ː is DELIBERATELY NOT A MEMBER. §1 says "ː (only in aː)", so a bare ː is a
# spec violation, and listing it as a unit made validate() accept ɛː/iː/uː --
# strictly weaker than the rule it claims to enforce. The unit model carries the
# rule instead of a separate check: "aː" is one inventory unit and ː alone is
# none, so phone_units' longest match still yields ["h", "aː", "n", "t"] for
# haːnt, while ɔː segments as ["ɔ", "ː"] and validate() names the stray ː.
# Measured over the same full-corpus census: 114,808 ː, every one of them the
# second half of aː, zero bare. e / o / ʊ are non-members for exactly the same
# reason (they exist only inside ej and oʊ) and the census agrees -- zero bare e,
# zero bare o, zero bare ʊ in 23,666 rows.
MARKS: tuple[str, ...] = ("ˈ",)

INVENTORY: frozenset[str] = frozenset(VOWELS + CONSONANTS + MARKS)

# ---------------------------------------------------------------------------
# Non-phones: structure the engine deliberately passes through
# ---------------------------------------------------------------------------
#
# Mirror of `yiddish_g2p._EDGE_PUNCT` -- the exact character class
# `split_affixes()` peels off a token into `lead`/`trail`, which
# `hebrew_to_ipa` then splices back around the phones. Copied rather than
# imported so this module never needs the 1.23 GB engine on sys.path; keep it
# assertable with
#     set(_ENGINE_EDGE_PUNCT) == set(yiddish_g2p._EDGE_PUNCT)
_ENGINE_EDGE_PUNCT = " \t\n\r.,!?;:()[]{}<>«»„‚‹›…\u05c3״“”\"*/\\|-"

# The quote family is the one part of `lead`/`trail` that does NOT reach the
# output: normalize_surface() unifies ״ “ ” to ASCII " and hebrew_to_ipa strips
# it (`r["lead"].replace('"', "")`) because §2.2 treats quotes as carrying no
# prosodic content, unlike . , ! ? which a TTS reads as phrase breaks.
_ENGINE_STRIPS = '"“”״'

# Characters that are structure, not phones. DERIVED FROM EVIDENCE, not from
# imagination -- the previous hand-written set omitted [ ] … « » „ ‚ ‹ › * / \
# | < > { } and ׃, so validate() denounced ordinary Yiddish typography as an
# off-inventory phone and /v1/audio/speech answered 400 on valid input ([ ]
# alone occur 234 times in the corpus census).
#
# HOW THIS WAS MEASURED (both experiments reproducible against the pinned
# engine at dist/phonikud-yi-engine):
#   1. Census: hebrew_to_ipa over all 23,666 rows of
#      data/corpus/yiddish_tts_dataset_v2.tsv, for the raw `text` column and
#      the pre-pointed `nikud` column. Every character emitted was either an
#      inventory phone or one of: space , . ? : - ! ; [ ] « » „ ( ).
#   2. Saturation: each character of `_ENGINE_EDGE_PUNCT` plus the apostrophe,
#      geresh, gershayim, ellipsis and every dash, injected into a Yiddish
#      carrier phrase in all five positions (word-initial, word-final,
#      mid-word, line-initial, line-final). Exactly `_ENGINE_EDGE_PUNCT` minus
#      whitespace minus `_ENGINE_STRIPS` came back in the IPA. Nothing else
#      can: a character outside that class is not peeled into lead/trail, so it
#      stays in the lexicon key and is consumed by the router.
# So the allowed set is that difference, and the derivation below is the claim.
NON_PHONE: frozenset[str] = frozenset(_ENGINE_EDGE_PUNCT) - frozenset(_ENGINE_STRIPS)
#: Historic private name, kept because this module's own docstrings cite it.
_NON_PHONE = NON_PHONE

# Longest-match-first matcher, built FROM the inventory so it can never drift
# out of sync with it. Descending length is what makes "aː" one unit rather than
# "a" + "ː", and "aj"/"ɔj"/"ej"/"oʊ" one unit rather than vowel + glide.
_UNIT_RE = re.compile(
    "|".join(re.escape(p) for p in sorted(INVENTORY, key=lambda p: (-len(p), p)))
)


def phone_units(ipa: str) -> list[str]:
    """Segment `ipa` into inventory units, longest match first.

    Anything the inventory does not cover comes back as a single-character unit,
    so callers can see exactly which character is foreign rather than being
    handed a silently repaired string. That is also how the §1 length rule is
    enforced: "aː" wins the longest match, so a ː that follows any other vowel
    has nothing to attach to and falls out as its own foreign unit.
    """
    units: list[str] = []
    pos = 0
    end = len(ipa)
    while pos < end:
        m = _UNIT_RE.match(ipa, pos)
        if m is not None:
            units.append(m.group(0))
            pos = m.end()
        else:
            units.append(ipa[pos])
            pos += 1
    return units


def validate(ipa: str) -> list[str]:
    """Return the unknown units in `ipa`, in order of appearance, deduped.

    An empty list means the string is inside the closed set -- the same check the
    corpus pipeline's QA gate (b) applies, but per-request. Punctuation and
    whitespace are not phones and not violations: `_NON_PHONE` holds exactly the
    characters the engine can splice around a token, so ``bˈaləbus, ju!`` and
    ``far vus [mkˈɔjrɔjs?]`` come back clean while ``mit ɐ θejl`` reports
    ``["ɐ", "θ"]`` and ``ɔːbər`` reports ``["ː"]``.
    """
    seen: set[str] = set()
    unknown: list[str] = []
    for unit in phone_units(ipa):
        if unit in INVENTORY or unit in _NON_PHONE or unit in seen:
            continue
        seen.add(unit)
        unknown.append(unit)
    return unknown


# ---------------------------------------------------------------------------
# Folding to a voice's vocabulary
# ---------------------------------------------------------------------------
#
# Candidates are tried in order; the first whose EVERY CHARACTER is in the vocab
# wins. Per-character is the right test because a Piper voice's vocabulary is a
# `phoneme_id_map` keyed by single characters -- "tʃ" is not an entry, it is the
# entries "t" and "ʃ" used back to back.
#
# Where a rule lists the phone itself first, that identity candidate is a no-op
# for any vocab that already has the phone (fold_to_vocab only consults the
# rules for units the vocab lacks) and documents the intended precedence: keep
# the true phone, degrade only as far as necessary.
FOLD_RULES: dict[str, tuple[str, ...]] = {
    # REQUIRED for the shipped Piper voice: its phoneme_id_map has ʦ, ʃ, ʒ, ɡ,
    # ŋ, ə, ɛ, ɔ, x, ˈ, ː -- but no ʧ and no ʤ. Without these two folds every
    # word with a tsh/dzh (ʧ, ʤ) loses a consonant. There is no rule for a bare
    # ː because a bare ː is not a unit (see MARKS); "aː" is the unit and its
    # rule is below.
    "ʧ": ("tʃ",),        # U+02A7 -> t + ʃ, the same affricate spelled out
    "ʤ": ("dʒ",),        # U+02A4 -> d + ʒ
    # Defensive: correct for this voice, insurance for the next one.
    "ʦ": ("ts",),        # U+02A6 -> t + s, should this voice ever lack ʦ
    "ɡ": ("g",),         # U+0261 script g -> ASCII g for vocabs keyed in ASCII
    "aː": ("aː", "a"),   # length is a nicety; losing the vowel is not an option
    "oʊ": ("oʊ", "o"),   # ʊ-less vocab: the o carries the syllable
    "ə": ("ə", "e"),     # schwa -> plain e, the nearest thing every vocab has
    "ŋ": ("ŋ", "n"),     # velar nasal -> alveolar; place is the cheapest loss
    "x": ("x", "χ", "k"),  # /x/ -> uvular χ (same sound, other symbol) -> k
}


def _usable(candidate: str, vocab: Collection[str]) -> bool:
    """True when every character of `candidate` is in the vocab (see above)."""
    return all(ch in vocab for ch in candidate)


def fold_to_vocab(ipa: str, vocab: Collection[str]) -> tuple[str, list[str]]:
    """Rewrite `ipa` into what `vocab` can pronounce.

    Returns ``(folded_ipa, dropped_units)``. Units the vocab already covers pass
    through untouched; units it lacks are replaced by the first usable candidate
    in FOLD_RULES; units with no usable candidate are omitted from the output and
    listed in `dropped_units`, deduped and in order. A phone the voice cannot say
    must surface as a warning the caller can show, never as unexplained silence.

    The stress mark ˈ is an ordinary unit here: it is never reordered and, since
    it has no fold rule, it is either kept verbatim or reported as dropped. That
    keeps it immediately before its vowel, which is the whole of its meaning.

    PUNCTUATION IS NOT A PHONE, here as in `validate()`. `_NON_PHONE` units the
    vocabulary lacks are removed from the output silently and are NOT reported:
    they carry no phonetic content, so a vocab that cannot represent them loses
    nothing a listener could hear. Reporting them was a real defect, not a
    cosmetic one -- blue_yi's char vocab has no `[ ] ׃ „ ‚ ‹ › | < > { }` and
    piper's additionally has no `… « » * / \\`, while the engine deliberately
    passes all of them through in lead/trail, so ordinary bracketed or
    sof-pasuq Yiddish (`[ ]` alone occurs 234 times in the corpus census) came
    back as `X-Dropped-Units: %5B %5D` and lit the UI's "the audio does not
    match the IPA at [ ]" strip, as though a consonant had gone missing.
    `dropped` names inventory phones only.
    """
    out: list[str] = []
    dropped: list[str] = []
    seen_dropped: set[str] = set()

    for unit in phone_units(ipa):
        if _usable(unit, vocab):
            out.append(unit)
            continue

        if unit in _NON_PHONE:
            # Structure the vocab cannot spell. Nothing audible is lost.
            continue

        replacement = next(
            (c for c in FOLD_RULES.get(unit, ()) if _usable(c, vocab)), None
        )
        if replacement is not None:
            out.append(replacement)
            continue

        if unit not in seen_dropped:
            seen_dropped.add(unit)
            dropped.append(unit)
            logger.warning(
                "dropping phone %r (U+%04X): absent from the voice vocabulary "
                "with no usable fold candidate",
                unit,
                ord(unit[0]),
            )

    return "".join(out), dropped
