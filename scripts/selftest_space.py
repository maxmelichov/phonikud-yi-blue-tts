#!/usr/bin/env python3
"""Run this before deploying: python3 scripts/selftest_space.py

Proves the Space is wired correctly — registry, phone inventory, audio packing,
the Yiddish label stack, the blue-yi TTS runtime, and the HTTP routes — end to
end. Exits non-zero on any failure, so it works as a Docker build gate.

Modelled on Phonikud-yi/src/selftest.py, and it carries that file's canaries
forward: the G2P readings that only the seven generated lexicon tables can
produce. That check (§4 below) is the deployment guard. A missing table makes
yiddish_g2p return {} for that table and keep going, so the Space would serve
plausible-but-wrong Yiddish — פעקל as fɛkl instead of pɛkl — and say nothing.

The Blue 2.5 block (§5) carries the second deployment guard, and it is the more
subtle one. Blue's flow-matching pipeline produces audio-shaped float32 of
exactly the right length even when the 144->24 latent fold is interleaved wrongly
or the stats.npz denormalization is skipped — the two mistakes most likely to
ship unnoticed. Shape, peak and duration checks all pass on that garbage. The
only thing that does not is the physics: each voice's median F0 must land on the
pitch its model card documents. So §5 estimates F0 with a dependency-free
autocorrelation tracker (the same one that validated the port; see
BLUE25_RECIPE.md §10) and asserts it per voice.

Checks whose third-party dependency is genuinely not installed are SKIPPED with
a reason instead of failing: this doubles as a quick sanity run on a laptop that
does not have onnxruntime. Anything else is a failure.
"""

from __future__ import annotations

import logging
import os
import re
import struct
import sys
import wave
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Imported at module level, unlike every other package import in this file: the
# registry is the source of truth for which bundle is pinned, and constants
# below derive their paths from it. It is safe here because registry.py pulls in
# nothing beyond the standard library -- no onnxruntime, no numpy -- so it
# cannot turn a missing optional dependency into an import-time crash.
from yiddish_phonikud import registry  # noqa: E402

# Third-party packages whose absence is a missing dependency (skip), not a bug
# (fail). A missing yiddish_phonikud.* module is always a bug.
OPTIONAL_DEPS = frozenset(
    {
        "numpy",
        "onnxruntime",
        "fastapi",
        "pydantic",
        "jinja2",
        "starlette",
        "huggingface_hub",
    }
)

# The floor of the route census: paths that must exist whatever docs/API.md
# says, so a docs regression cannot quietly empty this check. The rest of the
# census is parsed out of docs/API.md itself (§7) and compared both ways, which
# is what keeps the growing route list honest without hardcoding it twice.
CORE_ROUTES: tuple[str, ...] = (
    "/health",
    "/v1/models/sources",
    "/v1/models/load",
    "/v1/models/state",
    "/v1/voices",
    "/v1/languages",
    "/v1/phonemes/inventory",
    "/v1/audio/diacritize",
    "/v1/audio/phonemize",
    "/v1/audio/speech",
    "/",
    "/generate",
)

# The canaries from Phonikud-yi/src/selftest.py, unchanged.
CANARY_TEXT = "מיט א פאר יאר צוריק"
CANARY_IPA = "mit a pˈur jur ʦirˈik"

# BlueTTS 2.5 model card F0, per voice. These are the numbers the port was
# validated against (BLUE25_RECIPE.md §10: measured 123.5-130.9 Hz for the 128 Hz
# male, 208.0-216.2 for the 211 Hz female, 170.3-195.6 for the 180 Hz female).
# A 30 % band absorbs that honest utterance-to-utterance spread — this Space
# Figures are blue-yi's own model card, which documents lower pitches than the
# BlueTTS 2.5 bundle did for the same LibriTTS-R speakers -- the styles were
# re-encoded for this checkpoint, so they are not the same vectors.
#
# The band is deliberately wide (±30 %): this Space renders at speed 1.0 rather
# than the reference's 1.2, and measured F0 sits ~10-16 % above the card. It is
# still tight enough to catch what it exists to catch -- a wrong 144->24 latent
# fold produced a 231 Hz tonal drone on the 115 Hz male voice, far outside the
# band, while every wrong fold still yields audio of exactly the right length.
BLUE_VOICE_F0_HZ: dict[str, float] = {
    "Berl": 115.0,          # libri_male_6209
    "Hershl": 110.0,        # libri_male_8088
    "Sheyndl": 165.0,       # libri_female_1088
    "Rukhl": 184.0,         # libri_female_6147
}

#: Which offered voices are male. The style-file stems used to carry this
#: ("libri_male_6209"), so a substring test was enough; the public names do not,
#: so the mapping has to be explicit.
BLUE_MALE_VOICES = {"Berl", "Hershl"}

BLUE_F0_TOLERANCE = 0.30
BLUE_VOICES: tuple[str, ...] = (
    "Berl",
    "Hershl",
    "Rukhl",
    "Sheyndl",
)
# Env var the Blue adapter reads to use an already-downloaded bundle.
BLUE_DIR_ENV = "BLUE25_MODEL_DIR"
# The local HF cache snapshot of the bundle the registry pins, used only when
# BLUE25_MODEL_DIR is unset. Derived from the registry rather than written out:
# it used to name BlueTTS2.5-onnx and its revision by hand, and when the default
# runtime moved to blue-yi this kept pointing at the superseded bundle. A stale
# path here does not fail — it silently tests the wrong weights, or falls
# through to a cold-cache download.
BLUE_LOCAL_SNAPSHOT = Path(
    os.path.expanduser("~/.cache/huggingface/hub")
) / f"models--{registry.BLUE_REPO_ID.replace('/', '--')}" / "snapshots" / registry.BLUE_REVISION

fails: list[str] = []
skips: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'}  {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(label)


def skip(label: str, detail: str = "") -> None:
    print(f"skip  {label}" + (f"  {detail}" if detail else ""))
    skips.append(label)


def missing_dep(exc: BaseException) -> str | None:
    """The optional package this exception blames, or None if it blames us."""
    if isinstance(exc, ModuleNotFoundError) and exc.name:
        root = exc.name.split(".")[0]
        if root in OPTIONAL_DEPS:
            return exc.name
    return None


def section(name: str) -> None:
    print(f"\n-- {name} " + "-" * max(0, 60 - len(name)))


def median_f0(samples, rate: int, fmin: float = 60.0, fmax: float = 400.0,
              rms_floor: float = 0.02, hnr_floor: float = 0.3) -> tuple[float, int, float]:
    """(median F0 in Hz, voiced frame count, median harmonicity) — numpy only.

    A 40 ms / 10 ms autocorrelation tracker, ported from the f0.py that measured
    the numbers in BLUE25_RECIPE.md §10 so this check and that table are the same
    estimator. librosa is deliberately not a dependency of this Space: a lag-domain
    peak on a normalized autocorrelation is all a pitch *sanity* check needs, and
    it reproduced the reference measurements to within 1 Hz.
    """
    import numpy as np

    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    win, hop = int(0.04 * rate), int(0.01 * rate)
    if win < 8 or x.size < win:
        return float("nan"), 0, float("nan")
    lag_lo, lag_hi = int(rate / fmax), int(rate / fmin)
    f0s: list[float] = []
    hnrs: list[float] = []
    for start in range(0, x.size - win, hop):
        frame = x[start : start + win]
        if float(np.sqrt(np.mean(frame * frame))) < rms_floor:
            continue  # silence and breath: no pitch to find
        frame = frame - frame.mean()
        ac = np.correlate(frame, frame, "full")[win - 1 :]
        ac = ac / (ac[0] + 1e-12)
        lag = lag_lo + int(np.argmax(ac[lag_lo:lag_hi]))
        if ac[lag] > hnr_floor:  # below this the frame is noise, not voicing
            f0s.append(rate / lag)
            hnrs.append(float(ac[lag]))
    if not f0s:
        return float("nan"), 0, float("nan")
    return float(np.median(f0s)), len(f0s), float(np.median(hnrs))


# ---------------------------------------------------------------------------
# Model sources: prefer already-unpacked bundles so the selftest never pulls
# the ~1.23 GB engine or the ~280 MB Blue snapshot on a run.
# ---------------------------------------------------------------------------

from yiddish_phonikud.engine import ENGINE_DIR_ENV  # noqa: E402  (needs sys.path)

# The Yiddish G2P project is normally checked out beside this Space; its
# make_bundle.py output is byte-identical to the published model repo.
LOCAL_BUNDLE = ROOT.parent / "Phonikud-yi" / "dist" / "phonikud-yi-engine"

if os.environ.get(ENGINE_DIR_ENV):
    engine_source = f"{ENGINE_DIR_ENV}={os.environ[ENGINE_DIR_ENV]}"
elif (LOCAL_BUNDLE / "yiddish_labels.py").is_file():
    os.environ[ENGINE_DIR_ENV] = str(LOCAL_BUNDLE)
    engine_source = f"local bundle {LOCAL_BUNDLE}"
else:
    engine_source = "huggingface_hub snapshot_download (cold cache: ~1.23 GB)"

if os.environ.get(BLUE_DIR_ENV):
    blue_source = f"{BLUE_DIR_ENV}={os.environ[BLUE_DIR_ENV]}"
elif (BLUE_LOCAL_SNAPSHOT / "onnx" / "vocab.json").is_file():
    os.environ[BLUE_DIR_ENV] = str(BLUE_LOCAL_SNAPSHOT)
    blue_source = f"local snapshot {BLUE_LOCAL_SNAPSHOT}"
else:
    blue_source = "huggingface_hub snapshot_download (cold cache: ~281 MB)"

BLUE_DIR = Path(os.environ.get(BLUE_DIR_ENV, ""))

print(f"repo   : {ROOT}")
print(f"python : {sys.version.split()[0]}")
print(f"engine : {engine_source}")
print(f"blue   : {blue_source}")


# ---------------------------------------------------------------------------
# 1. registry
# ---------------------------------------------------------------------------
section("registry")
try:
    from yiddish_phonikud import registry

    ids = [m.id for m in registry.runtimes()]
    check(sorted(ids) == ["blue_yi"], "blue_yi is the whole catalog", ", ".join(ids))
    check(
        registry.DEFAULT_RUNTIME_ID == "blue_yi",
        "default runtime is blue_yi",
        registry.DEFAULT_RUNTIME_ID,
    )

    blue = registry.runtime("blue_yi")
    check(blue is not None and blue.available, "blue_yi available in this build")
    check(registry.runtime("nope") is None, "an unknown id resolves to no manifest")
    if blue is not None:
        caps = blue.capabilities
        check(
            caps.fixed_voices and not caps.voice_reference,
            "blue_yi capabilities: fixed_voices, no voice_reference",
            # duration_predictor.onnx needs a z_ref this bundle cannot produce
            # (no AE encoder export), so cloning from a wav is impossible.
            f"fixed_voices={caps.fixed_voices} voice_reference={caps.voice_reference}",
        )
        check(caps.yiddish and caps.streaming, "blue_yi is Yiddish + streaming")
        # The rate is catalog metadata: clients size their players from it.
        declared = getattr(blue, "sample_rate", 0) or (
            44100 if ("44.1" in blue.description or "44100" in blue.description) else 0
        )
        check(declared == 44100, "blue_yi declares 44.1 kHz", f"{declared} Hz")
        if BLUE_DIR.is_dir():
            absent = [n for n in blue.required_files if not (BLUE_DIR / n).is_file()]
            check(
                not absent,
                "every blue_yi required file exists in the bundle",
                f"{len(blue.required_files)} files" if not absent
                else f"MISSING: {', '.join(absent)}",
            )
    check(
        blue is not None and registry.is_installed(blue, ROOT),
        "the blue_yi bundle is installed (hub cache or BLUE25_MODEL_DIR)",
        f"{len(blue.required_files)} required files" if blue is not None else "",
    )
except Exception as exc:  # noqa: BLE001
    check(False, "registry", repr(exc))


# ---------------------------------------------------------------------------
# 2. phones
# ---------------------------------------------------------------------------
section("phones")
try:
    from yiddish_phonikud import phones

    units = phones.phone_units(CANARY_IPA)
    foreign = [u for u in units if u not in phones.INVENTORY and not u.isspace()]
    check(
        not foreign,
        "phone_units segments the canary to known units",
        f"{len(units)} units" if not foreign else f"foreign: {foreign}",
    )
    # Longest-match-first is the whole point: the digraphs must not fall apart.
    check(
        phones.phone_units("haːnt hoʊz ʦvaj vejɡ ʃɔjn")
        == ["h", "aː", "n", "t", " ", "h", "oʊ", "z", " ", "ʦ", "v", "aj", " ",
            "v", "ej", "ɡ", " ", "ʃ", "ɔj", "n"],
        "multi-character units segment as one unit",
        "aː oʊ aj ej ɔj",
    )

    check(phones.validate(CANARY_IPA) == [], "validate() clean on the canary", CANARY_IPA)
    bad = "mit ɐ θejl"  # ɐ and θ are outside the closed set
    offenders = phones.validate(bad)
    check(offenders == ["ɐ", "θ"], "validate() names the offenders", repr(offenders))

    # fold_to_vocab is exercised against a synthetic vocab: every inventory
    # character except the two single-codepoint affricates. No shipped runtime
    # needs this — blue_yi's vocab covers everything (§5 asserts it) — but the
    # folding path stays under test for the next runtime that does not.
    vocab = ({ch for unit in phones.INVENTORY for ch in unit} | {" "}) - {"ʧ", "ʤ"}
    folded, dropped = phones.fold_to_vocab("ʧalnt ʤab mɛnʧ", vocab)
    check(
        folded == "tʃalnt dʒab mɛntʃ" and not dropped,
        "fold_to_vocab folds ʧ->tʃ and ʤ->dʒ",
        folded,
    )
    # ˈ has no fold rule on purpose, so a vocab without it exercises the
    # report-a-drop path. Silence is never an acceptable answer to a phone the
    # voice cannot say.
    # The drop is the point here, so mute the module's (correct) warning.
    logging.getLogger("yiddish_phonikud.phones").setLevel(logging.ERROR)
    folded, dropped = phones.fold_to_vocab("pˈur", set("pur"))
    check(
        folded == "pur" and dropped == ["ˈ"],
        "fold_to_vocab reports a dropped unit",
        f"{folded!r} dropped={dropped}",
    )
    # --- alternate input notation (the Phonemes tab) ----------------------
    # A reviewer typing from YIVO/Weinreich habit writes mɪt / pʊr / tsɪrɪk and
    # ASCII g. Those are the same sounds spelled differently, so they convert.
    out, subs = phones.normalize_notation("mɪt a pʊr jʊr tsɪrɪk")
    check(
        out == "mit a pur jur ʦirik" and not phones.validate(out),
        "normalize_notation converts YIVO notation to the inventory",
        f"{out!r} via {[(x.source, x.result) for x in subs]}",
    )
    out, _ = phones.normalize_notation("gut")
    check(out == "ɡut", "ASCII g converts to ɡ U+0261", f"{out!r}")
    out, _ = phones.normalize_notation("mit'n")
    check(out == "mitˈn", "ASCII apostrophe converts to the stress mark ˈ", f"{out!r}")

    # Both of these were real bugs in the first implementation, which used a
    # series of str.replace calls. They must not come back.
    # hoʊz / loʊt / oʊx are the engine's own readings of הויז / לויט / אויך —
    # all three lexicon hits at HIGH confidence. Use verified readings as test
    # data, never invented ones: an example string that looks like a sanctioned
    # form but is not gets copied into commit messages and docs and then read
    # back as if the engine produced it.
    out, subs = phones.normalize_notation("hoʊz loʊt oʊx")
    check(
        out == "hoʊz loʊt oʊx" and not subs,
        "a legitimate oʊ survives conversion untouched",
        f"{out!r} — a bare ʊ->u rule used to reach inside oʊ and split it",
    )
    out, subs = phones.normalize_notation("eːbn")
    eː = [x for x in subs if x.source == "eː"]
    check(
        out == "ejbn" and eː and eː[0].applied and eː[0].ambiguous
        and eː[0].alternatives == ("ɛ",),
        "an ambiguous eː resolves and reports the assumption",
        f"{out!r} — eː -> ej, alternative ɛ offered",
    )

    # Idempotence is the property that a str.replace chain cannot hold: it used
    # to rewrite its own output (e->ɛ cascading into the ej that eː->ej had just
    # produced, turning eːbn into ɛjbn). One protected left-to-right pass
    # consumes each source position exactly once, so a second pass is a no-op.
    for probe in ("mɪt a pʊr jʊr tsɪrɪk", "gut", "hoʊz loʊt oʊx", "eːbn", "ɡrojs",
                  "mit a pˈur jur ʦirˈik"):
        once, _ = phones.normalize_notation(probe)
        twice, _ = phones.normalize_notation(once)
        if once != twice:
            check(False, "normalize_notation is idempotent", f"{probe!r}: {once!r} -> {twice!r}")
            break
    else:
        check(True, "normalize_notation is idempotent", "a second pass changes nothing")

    # Ambiguous symbols are reported, never guessed: `o` is ɔ in ɡrɔjs but u in
    # uvnt and i in inz, and the symbol does not carry the vowel class.
    # Blocking on the first `o` of a paragraph made the tab unusable, so the
    # common reading is applied and flagged as an assumption instead.
    out, subs = phones.normalize_notation("ɡrojs")
    ambiguous = [x for x in subs if x.ambiguous]
    check(
        out == "ɡrɔjs" and not phones.validate(out) and ambiguous
        and ambiguous[0].applied and ambiguous[0].result == "ɔ"
        and set(ambiguous[0].alternatives) == {"u", "i"},
        "an ambiguous vowel resolves and reports what it assumed",
        f"o -> ɔ, alternatives {ambiguous[0].alternatives if ambiguous else None}",
    )

    # Engine output must be a strict no-op here: it is already in the inventory.
    canon = "mit a pˈur jur ʦirˈik"
    out, subs = phones.normalize_notation(canon)
    check(out == canon and not subs, "spec-v3 notation passes through unchanged", canon)

    check(
        phones.stress_report("mit a pur jur ʦirik") is not None
        and phones.stress_report(canon) is None
        and phones.stress_report("mit") is None,
        "missing stress is reported, present stress and monosyllables are not",
        "multi-vowel word with no ˈ warns; one-syllable input does not",
    )
except Exception as exc:  # noqa: BLE001
    check(False, "phones", repr(exc))


# ---------------------------------------------------------------------------
# 3. audio
# ---------------------------------------------------------------------------
section("audio")
try:
    import numpy as np

    from yiddish_phonikud import audio

    # Multi-sentence Hasidic Yiddish, with a maqaf-joined compound (בעל־הבית):
    # the G2P keys its lexicon on the whole hyphenated form, so a split there
    # would downgrade a HIGH-confidence hit to two fallbacks.
    paragraph = (
        "דער בעל־הבית האט געזאגט אז מען וועט מאכן א גרויסע שמחה. "
        "די קינדער שפילן זיך אין דרויסן, און די מאמע רופט זיי אריין. "
        "וואס האט ער געזאגט? מיט א פאר יאר צוריק איז דאס געווען אנדערש; "
        "היינט איז אלעס אנדערש."
    )
    chunks = audio.chunk_text(paragraph, max_chars=60)
    dense = lambda s: "".join(s.split())  # noqa: E731
    check(
        dense("".join(chunks)) == dense(paragraph),
        "chunk_text loses no non-whitespace character",
        f"{len(chunks)} chunks, {len(dense(paragraph))} chars",
    )
    check(
        sum("בעל־הבית" in c for c in chunks) == 1
        and not any("בעל־" == c[-4:] or c.startswith("הבית") for c in chunks),
        "chunk_text never splits at a maqaf",
        "בעל־הבית intact",
    )
    check(
        all(len(c) <= 60 or " " not in c for c in chunks),
        "chunk_text honours max_chars",
        f"longest {max(len(c) for c in chunks)}",
    )
    # Geresh/gershayim mark abbreviations and acronyms in Yiddish, never a
    # sentence end: splitting there separates a title from its name.
    abbrev = "ר׳ מנדל האט געזאגט אז די ליטװישע ישיבֿות זענען אנדערש."
    check(
        all("ר׳" not in c or "מנדל" in c for c in audio.chunk_text(abbrev, max_chars=200)),
        "chunk_text does not split at a geresh abbreviation",
        "ר׳ מנדל intact",
    )

    # The chunk budget is not a taste setting: it has to leave every chunk
    # inside the acoustic model's token cap, encoder padding and the chunker's
    # own collocation overrun included. Blue's encoder adds 3 tokens (a trailing
    # "." plus a padding space at each end), and chunk_text may overshoot
    # max_chars rather than cut through a multiword entry.
    from yiddish_phonikud.runtimes.blue_yi import MAX_TEXT_TOKENS

    check(
        audio.MAX_CHUNK_CHARS + 3 <= MAX_TEXT_TOKENS,
        "the chunk budget fits inside the model's token cap",
        f"{audio.MAX_CHUNK_CHARS} chars + 3 encoder tokens <= {MAX_TEXT_TOKENS}",
    )
    gap_n = round(audio.CHUNK_GAP_SECONDS * 44100)
    parts = [np.full(100, 0.5, dtype=np.float32), np.full(50, -0.5, dtype=np.float32)]
    joined = audio.join_chunks(parts, 44100)
    check(
        joined.size == 150 + gap_n
        and float(np.abs(joined[100:100 + gap_n]).max()) == 0.0,
        "join_chunks inserts one silence gap per seam",
        f"{joined.size} samples = 150 + {gap_n} of silence",
    )
    check(
        audio.join_chunks([], 44100).size == 0
        and audio.join_chunks([parts[0]], 44100).size == 100,
        "join_chunks degrades cleanly to 0 and 1 part",
        "no leading or trailing gap",
    )

    payload = b"x" * 300  # >255 so the length field proves its byte order
    f = audio.frame(audio.KIND_CHUNK, payload)
    check(
        len(f) == 5 + len(payload)
        and f[0] == audio.KIND_CHUNK
        and struct.unpack(">I", f[1:5])[0] == len(payload)
        and f[5:] == payload,
        "frame() = 5 header bytes + big-endian length",
        f"header {f[:5].hex()}",
    )
    check(
        audio.error_frame("boom")[0] == audio.KIND_ERROR
        and audio.error_frame("boom")[5:] == b"boom",
        "error_frame() carries UTF-8 text under kind 3",
    )

    n = 2205  # 0.1 s at 22050 Hz
    tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(n) / 22050)).astype(np.float32)
    wav = audio.pcm16_wav(tone, 22050)
    check(len(wav) == 44 + 2 * n, "pcm16_wav length = 44-byte header + 16-bit frames",
          f"{len(wav)} bytes")
    with wave.open(BytesIO(wav)) as w:
        parsed = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes())
    check(parsed == (1, 2, 22050, n), "pcm16_wav parses as mono 16-bit PCM", str(parsed))
    # blue-yi runs at 44.1 kHz, so the packer must carry the rate it is given
    # rather than a hardcoded constant.
    wav441 = audio.pcm16_wav(tone, 44100)
    with wave.open(BytesIO(wav441)) as w:
        check(w.getframerate() == 44100, "pcm16_wav honours a 44.1 kHz rate", "44100 Hz")
except Exception as exc:  # noqa: BLE001
    dep = missing_dep(exc)
    if dep:
        skip("audio", f"{dep} not installed")
    else:
        check(False, "audio", repr(exc))


# ---------------------------------------------------------------------------
# 4. engine — THE DEPLOYMENT GUARD
# ---------------------------------------------------------------------------
section("engine (deployment guard)")
try:
    from yiddish_phonikud import engine

    engine.load()
    check(engine.is_loaded(), "engine.load() succeeded", str(engine.engine_dir()))

    got = engine.text_to_ipa(CANARY_TEXT)
    check(got == CANARY_IPA, f"text_to_ipa({CANARY_TEXT}) == canary", got)

    # פאר is a homograph: the preposition is far, the quantifier in "א פאר יאר"
    # is pur. Only the lexicon tables know the difference.
    prep = engine.text_to_ipa("ער דאוונט פאר די קהילה")
    check(" far " in f" {prep} ", "preposition פאר stays far", prep)

    nikud = engine.text_to_nikud("מיט א פאר יאר צוריק אין שפיטאל")
    check("פּאָר" in nikud and "פֿאַר" not in nikud, "v5 points אַ פּאָר (not פֿאַר)", nikud)

    info = engine.info()
    tables = info.get("tables", {})
    empty = sorted(name for name, size in tables.items() if not size)
    check(len(tables) == 7, "info() reports all 7 lexicon tables", f"{len(tables)} tables")
    check(
        not empty,
        "every lexicon table is non-empty",
        # A missing table is not a degraded mode: it is confidently wrong IPA.
        ", ".join(f"{k}={v}" for k, v in tables.items()) if not empty
        else f"EMPTY: {', '.join(empty)} — redownload the engine, do not serve",
    )
    # --- respelling as a correction channel (the REYD approach) -------------
    # REYD's Yiddish TTS sidesteps loshn-koydesh entirely: its corpus respells
    # Hebrew-origin words phonetically (תכשיט -> טאַכשעט) so that letters ARE
    # phonemes, and 0 of 4362 rows contain ת. The same trick works here as an
    # editorial channel, and it matters because it is far easier for a Yiddish
    # speaker to respell a word than to write IPA.
    #
    # Two properties have to hold for that to be usable, and both are checked:
    #   1. a Hasidic respelling reproduces the word's gold reading exactly;
    #   2. words the engine cannot resolve from Hebrew spelling ARE resolvable
    #      from a respelling.
    respell_ok = engine.text_to_ipa("שאַבעס") == engine.text_to_ipa("שבת")
    check(
        respell_ok,
        "a Hasidic respelling reproduces the gold reading",
        f'שאַבעס -> {engine.text_to_ipa("שאַבעס")} == שבת (gold)',
    )

    # The vowel LETTER carries the dialect, so a YIVO-convention respelling
    # yields standard-Yiddish vowels, not Hasidic ones. This is why REYD's own
    # respellings cannot be imported verbatim: they would inject Litvish
    # readings over native Hasidic verdicts.
    yivo = engine.text_to_ipa("שאָבעס")
    check(
        yivo != engine.text_to_ipa("שבת") and yivo == "ʃˈubəs",
        "a YIVO respelling reads standard, not Hasidic",
        f"שאָבעס -> {yivo} vs שאַבעס -> {engine.text_to_ipa('שאַבעס')}",
    )

    rescued = engine.text_to_ipa("טאַכשעט")
    check(
        not engine.text_to_ipa("תכשיט").strip() and rescued == "tˈaxʃət",
        "a respelling rescues a word the engine quarantines",
        f"תכשיט -> (quarantined), טאַכשעט -> {rescued}",
    )

    # --- pointing-model head wiring ----------------------------------------
    # yiddish_nikud unpacks session.run() as (nikud, shin, rafe) positionally,
    # and its own guard only compares the SORTED head sizes -- [2, 2, 22]
    # against [2, 2, 22]. shin and rafe are both 2, so that guard passes just as
    # happily with those two heads swapped, and a swap is silent: rafe marks
    # would land on ש and shin dots on בכפגדת. The classes ARE distinguishable
    # by content even though the sizes are not, so check the content. (The
    # deployed export is wired correctly today; this is what would notice if a
    # re-export ever reordered the heads.)
    import json as _json  # noqa: PLC0415

    from yiddish_nikud import YiddishNikud  # noqa: PLC0415

    _n = YiddishNikud()
    _meta = _n._session.get_modelmeta().custom_metadata_map
    _names = [o.name for o in _n._session.get_outputs()]
    check(
        _names == ["nikud_logits", "shin_logits", "rafe_logits"],
        "pointing heads are in the order the decoder unpacks them",
        " -> ".join(_names),
    )
    check(
        _json.loads(_meta["shin_classes"]) == ["\u05c1", "\u05c2"],
        "the shin head really carries the shin/sin dots",
        repr(_json.loads(_meta["shin_classes"])),
    )
    check(
        _json.loads(_meta["rafe_classes"]) == ["", "\u05bf"],
        "the rafe head really carries the rafe mark",
        repr(_json.loads(_meta["rafe_classes"])),
    )
    # End to end: a swap would put the wrong mark on both letters at once.
    _pointed = engine.text_to_nikud("שבת")
    check(
        "\u05c1" in _pointed and "\u05bf" not in _pointed,
        "heads land on the right letters end to end",
        f"שבת -> {_pointed}",
    )

except Exception as exc:  # noqa: BLE001
    dep = missing_dep(exc)
    if dep:
        skip("engine", f"{dep} not installed")
    else:
        check(False, "engine", repr(exc))




# ---------------------------------------------------------------------------
# 5. blue-yi runtime (the default) — THE SECOND DEPLOYMENT GUARD
# ---------------------------------------------------------------------------
section("blue-yi runtime (default)")
try:
    import numpy as np

    from yiddish_phonikud import audio, phones
    from yiddish_phonikud import runtimes

    blue_rt = runtimes.load_default()
    check(blue_rt.id == "blue_yi", "load_default() loads blue-yi", blue_rt.id)
    check(
        blue_rt.sample_rate == 44100,
        "blue runtime reports 44100 Hz",
        f"{blue_rt.sample_rate} Hz (tts.json ae.sample_rate)",
    )
    voices = blue_rt.voices()
    check(
        sorted(voices) == list(BLUE_VOICES),
        "blue lists its four fixed voices",
        ", ".join(voices),
    )

    # --- vocab coverage ------------------------------------------------------
    # Every character of the closed inventory, affricates included, is a single
    # id in vocab.json's char_to_id, so nothing is ever folded or dropped for
    # this runtime. Assert ZERO missing — a regression here (a stale bundle, a
    # vocab read through the wrong key) would be silent, because fold_to_vocab
    # would quietly start folding and the audio would still play.
    blue_vocab = blue_rt.vocab()
    blue_missing = sorted(
        unit for unit in phones.INVENTORY
        if not all(ch in blue_vocab for ch in unit)
    )
    check(
        blue_missing == [],
        "blue vocab covers the ENTIRE Yiddish inventory",
        f"{len(phones.INVENTORY)} units, 0 missing" if not blue_missing
        else f"MISSING: {' '.join(blue_missing)} — wrong/stale bundle, do not serve",
    )
    check(
        all(ch in blue_vocab for ch in ("ʦ", "ʧ", "ʤ", "ŋ", "ɡ", "ˈ", "ː")),
        "the single-codepoint affricates and marks are native",
        "ʦ ʧ ʤ ŋ ɡ ˈ ː all in char_to_id",
    )
    folded, blue_folds = phones.fold_to_vocab(CANARY_IPA + " ʧalnt ʤab", blue_vocab)
    check(
        folded == CANARY_IPA + " ʧalnt ʤab" and not blue_folds,
        "folding against blue's vocab is a no-op",
        "no ʧ->tʃ / ʤ->dʒ rewriting, nothing dropped",
    )

    # --- the two ASCII/IPA collision traps ---------------------------------
    # Both traps are silent by construction: ASCII ' is id 5 and ASCII g is id
    # 154, so neither raises — the model just says an apostrophe instead of a
    # stress mark, or a different phone instead of /ɡ/. The only proof that the
    # runtime keeps them apart is that they tokenize to different audio.
    check(
        ord("ˈ") == 0x02C8 and ord("'") == 0x0027 and "'" in blue_vocab,
        "ASCII ' and ˈ U+02C8 are distinct vocab members",
        "' is id 5 (apostrophe), never an alias for the stress mark",
    )
    check(
        ord("ɡ") == 0x0261 and "g" in blue_vocab and "ɡ" in blue_vocab
        and "ɡ" in phones.INVENTORY and "g" not in phones.INVENTORY,
        "ASCII g and ɡ U+0261 are distinct vocab members",
        "inventory holds ɡ U+0261 only",
    )
    trap_kw = dict(voice="libri_male_6209", speed=1.0, n_steps=4, cfg_scale=4.0, seed=1234)
    stress_ipa, _ = blue_rt.synthesize("pˈur", **trap_kw)
    ascii_ipa, _ = blue_rt.synthesize("p'ur", **trap_kw)
    check(
        stress_ipa.shape != ascii_ipa.shape
        or not np.array_equal(stress_ipa, ascii_ipa),
        "ˈ is not silently accepted as ASCII '",
        "same seed, different audio",
    )
    g_script, _ = blue_rt.synthesize("ɡut", **trap_kw)
    g_ascii, _ = blue_rt.synthesize("gut", **trap_kw)
    check(
        g_script.shape != g_ascii.shape or not np.array_equal(g_script, g_ascii),
        "ɡ U+0261 is not confused with ASCII g",
        "same seed, different audio",
    )

    # --- synthesis shape, level and duration -------------------------------
    result = blue_rt.synthesize(CANARY_IPA, voice="libri_male_6209", speed=1.0, seed=1234)
    check(
        isinstance(result, tuple) and len(result) == 2
        and isinstance(result[0], np.ndarray) and isinstance(result[1], list),
        "blue synthesize() returns (ndarray, list)",
        f"({type(result[0]).__name__}, {type(result[1]).__name__})"
        if isinstance(result, tuple) and len(result) == 2 else repr(type(result)),
    )
    samples, blue_dropped = result
    check(
        not hasattr(blue_rt, "last_dropped"),
        "blue runtime exposes no last_dropped",
        "dropped units come back from synthesize()",
    )
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
    check(
        isinstance(samples, np.ndarray) and samples.dtype == np.float32
        and samples.ndim == 1 and samples.size > 0,
        "blue synthesize() returns float32 mono",
        f"{samples.size} samples, dtype {samples.dtype}, dropped={blue_dropped}",
    )
    # Blue's vocoder legitimately overshoots ±1.0 (1.32 measured), so the
    # adapter has to peak-limit before the contract's [-1, 1] holds.
    check(
        0.05 <= peak <= 1.0 and 0.005 <= rms <= 0.5,
        "blue level is sane and peak-limited",
        f"peak {peak:.3f}, rms {rms:.4f}",
    )
    n_phones = len([u for u in phones.phone_units(CANARY_IPA)
                    if not u.isspace() and u != "ˈ"])
    seconds = samples.size / blue_rt.sample_rate
    per_phone = seconds / max(n_phones, 1)
    # 0.059-0.065 s/phone measured at speed 1.2 (RECIPE §11); speed 1.0 is
    # slower, and the band is wide enough that only a duration-predictor or
    # edge-trim mistake (a 3072-sample stub, or 16.5 s of saturation) fails it.
    check(
        0.03 <= per_phone <= 0.15,
        "blue duration tracks the phoneme count",
        f"{n_phones} phones, {seconds:.2f} s, {per_phone:.3f} s/phone",
    )

    # --- the too-long guard, on the TEXT and before any graph runs ---------
    # The old guard tested predicted seconds AFTER the pace blend and the speed
    # division, and the raw predictor's ceiling (~15.0 s) sits below the 16 s it
    # compared against — so it was unreachable in the direction that matters
    # (879 characters rendered as a 3x cram) and fired in the direction that
    # does not (265 characters at speed 0.8 refused, though the same text
    # renders correctly at speed 1.0).
    from yiddish_phonikud.runtimes.blue_yi import (
        MAX_TEXT_TOKENS as BLUE_MAX_TOKENS,
        UtteranceTooLongError,
    )

    at_cap = ("a " * (BLUE_MAX_TOKENS // 2))[: BLUE_MAX_TOKENS - 3]
    over_cap = "a " * BLUE_MAX_TOKENS
    ok_wav, _ = blue_rt.synthesize(at_cap, voice="libri_male_6209", n_steps=1, seed=1)
    try:
        blue_rt.synthesize(over_cap, voice="libri_male_6209", n_steps=1, seed=1)
        refused_long, why_long = False, "rendered text past the cap"
    except UtteranceTooLongError as exc:
        refused_long, why_long = True, str(exc)[:70]
    check(
        ok_wav.size > 0 and refused_long,
        "blue renders text at the cap and refuses text past it",
        f"{BLUE_MAX_TOKENS - 3} chars -> {ok_wav.size / blue_rt.sample_rate:.1f} s; "
        f"over the cap -> {why_long}",
    )
    # Slow speech is a legitimate request, not saturation: the cap is on the
    # text, so speed<1 must render rather than raise.
    slow, _ = blue_rt.synthesize(
        at_cap[:200], voice="libri_male_6209", speed=0.5, n_steps=1, seed=1
    )
    fast, _ = blue_rt.synthesize(
        at_cap[:200], voice="libri_male_6209", speed=2.0, n_steps=1, seed=1
    )
    check(
        slow.size > fast.size * 3,
        "speed does not narrow what blue will render",
        f"speed 0.5 -> {slow.size / blue_rt.sample_rate:.1f} s, "
        f"speed 2.0 -> {fast.size / blue_rt.sample_rate:.1f} s",
    )

    # --- the limiter is static, so chunks of one paragraph match in level ---
    # `0.95/peak` is a per-signal gain: it delivered one sentence of a paragraph
    # 2.9 dB quieter than its neighbours purely because of where its loudest
    # sample landed, which pumps at every chunk boundary. A static soft knee has
    # no such dependence.
    levels = []
    for utt in ("mit a pˈur jur ʦirˈik", "a dank far də ɡˈitə nˈaːjəs",
                "vus hut ɛr ɡəzˈuɡt"):
        piece, _ = blue_rt.synthesize(
            utt, voice="libri_male_6209", speed=1.0, n_steps=8, cfg_scale=4.0, seed=1234
        )
        rms = float(np.sqrt(np.mean(piece.astype(np.float64) ** 2)))
        levels.append(20.0 * float(np.log10(max(rms, 1e-12))))
        check(
            float(np.abs(piece).max()) <= 1.0,
            f"blue output stays inside [-1, 1] ({utt.split()[0]})",
            f"peak {float(np.abs(piece).max()):.3f}",
        )
    spread = max(levels) - min(levels)
    check(
        spread < 1.5,
        "consecutive sentences of one paragraph match in level",
        f"{spread:.2f} dB spread over {', '.join(f'{v:.1f}' for v in levels)} dBFS",
    )

    # --- punctuation is not a casualty ------------------------------------
    # blue's char vocab has no [ ] ׃ „ ‚ ‹ › | < > { }, and the engine passes
    # them through in lead/trail. They carry no sound, so they are removed
    # without being reported: reporting them lit the UI's "the audio does not
    # match the IPA at [ ]" strip on ordinary bracketed Yiddish.
    bracketed, bracket_dropped = blue_rt.synthesize(
        "ʦi, ɔjb [nit] farvˈus?", voice="libri_male_6209", n_steps=1, seed=1
    )
    # A genuinely unmapped character still has to be reported. This vocab is a
    # 245-symbol multilingual IPA set, so ɐ and θ are *in* it (they are simply
    # the wrong phone for Yiddish, which is validate()'s business, not this
    # method's); § and ¶ are not in it and are not punctuation the engine can
    # emit either, so they are the honest probe for "no embedding".
    _, foreign_dropped = blue_rt.synthesize(
        "mit § ju ¶", voice="libri_male_6209", n_steps=1, seed=1
    )
    check(
        bracket_dropped == [] and bracketed.size > 0 and foreign_dropped == ["§", "¶"],
        "blue reports off-vocab PHONES and never punctuation",
        f"brackets -> {bracket_dropped}, §/¶ -> {foreign_dropped}",
    )

    # --- proof it is speech, not noise (BLUE25_RECIPE.md §10) --------------
    # This is the ONLY check here that a wrong 144->24 latent fold or a skipped
    # stats.npz denormalization fails: both still emit float32 mono of exactly
    # the right length and a plausible peak. The wrong fold (channel =
    # phase*24 + ch) produced a 231 Hz drone on the 128 Hz male voice.
    f0_by_voice: dict[str, float] = {}
    for voice, documented in BLUE_VOICE_F0_HZ.items():
        wav_v, _ = blue_rt.synthesize(
            CANARY_IPA, voice=voice, speed=1.0, n_steps=8, cfg_scale=4.0, seed=1234
        )
        est, voiced, hnr = median_f0(wav_v, blue_rt.sample_rate)
        f0_by_voice[voice] = est
        lo = documented * (1.0 - BLUE_F0_TOLERANCE)
        hi = documented * (1.0 + BLUE_F0_TOLERANCE)
        check(
            voiced >= 10 and lo <= est <= hi and hnr > 0.4,
            f"blue F0 matches the model card for {voice}",
            f"{est:.1f} Hz vs documented {documented:.0f} Hz "
            f"(band {lo:.0f}-{hi:.0f}), {voiced} voiced frames, hnr {hnr:.2f}",
        )
    male = f0_by_voice.get("Berl", float("nan"))
    females = [v for k, v in f0_by_voice.items() if k not in BLUE_MALE_VOICES]
    check(
        bool(females) and all(male < f for f in females),
        "blue keeps speaker identity: the male voice is the lowest",
        f"male {male:.1f} Hz < females {', '.join(f'{f:.1f}' for f in females)}",
    )

    # An unknown voice must be refused, never quietly rendered in some other
    # speaker: fixed_voices=True is a promise the caller can rely on.
    try:
        blue_rt.synthesize(CANARY_IPA, voice="no_such_voice")
        refused, why = False, "returned audio for a voice that does not exist"
    except ValueError as exc:
        refused, why = True, f"ValueError: {exc}"
    check(refused, "blue refuses an unknown voice", why)

    # --- determinism --------------------------------------------------------
    a, _ = blue_rt.synthesize(CANARY_IPA, voice="Berl", speed=1.0, n_steps=4, seed=7)
    b, _ = blue_rt.synthesize(CANARY_IPA, voice="Berl", speed=1.0, n_steps=4, seed=7)
    c, _ = blue_rt.synthesize(CANARY_IPA, voice="Berl", speed=1.0, n_steps=4, seed=8)
    check(
        a.shape == b.shape and np.array_equal(a, b),
        "same seed gives bit-identical samples",
        f"seed 7 twice, {a.size} samples",
    )
    check(
        a.shape != c.shape or not np.array_equal(a, c),
        "a different seed gives different samples",
        "seed 7 != seed 8 (the flow-matching noise is really seeded)",
    )

    # --- end to end at 44.1 kHz --------------------------------------------
    wav = audio.pcm16_wav(samples, blue_rt.sample_rate)
    with wave.open(BytesIO(wav)) as w:
        parsed = (w.getnchannels(), w.getframerate(), w.getnframes())
    check(
        parsed == (1, 44100, samples.size),
        "end to end: IPA -> blue samples -> 44.1 kHz WAV",
        f"{parsed[2]} frames @ {parsed[1]} Hz",
    )
    check(
        runtimes.state().get("runtime") == "blue_yi"
        and runtimes.state().get("sample_rate") == 44100,
        "runtimes.state() reports the loaded blue runtime",
        f"{runtimes.state()}",
    )
    # A per-request `runtime` must serve that request only and must never swap
    # the process-wide singleton. With one runtime in the catalog the only
    # remaining ways to reach instance() are the resident id and an unknown one:
    # the first must hand back the resident object rather than rebuild sessions,
    # and the second must raise without disturbing what is loaded.
    still = runtimes.loaded()
    check(
        runtimes.instance("blue_yi") is still,
        "instance() hands back the resident runtime for its own id",
        "no session rebuild per request",
    )
    try:
        runtimes.instance("no_such_runtime")
        raised = "no exception"
    except runtimes.RuntimeNotAvailable as exc:
        raised = type(exc).__name__
    except Exception as exc:  # noqa: BLE001
        raised = repr(exc)
    after = runtimes.loaded()
    check(
        raised == "RuntimeNotAvailable"
        and after is still
        and runtimes.state().get("runtime") == "blue_yi"
        and runtimes.state().get("sample_rate") == 44100,
        "instance() rejects an unknown runtime without touching the loaded one",
        f"{raised}, loaded -> {after.id if after else None}",
    )
except Exception as exc:  # noqa: BLE001
    dep = missing_dep(exc)
    if dep:
        skip("blue runtime", f"{dep} not installed")
    else:
        check(False, "blue runtime", repr(exc))


# ---------------------------------------------------------------------------
# 6. the pipeline — the authority chain, one pass, one pointing call
#
# This is the block that guards the review's CRITICAL finding (C1). Every other
# check here can pass while the routes quietly feed the v5 pointing model's
# guesses back into the G2P, which is what shipped before and what measurably
# overrides gold and audio-confirmed readings. So: assert the served phonemes
# ARE text_to_ipa(raw text), assert they are NOT text_to_ipa(nikud) on a
# sentence where the two demonstrably differ, assert the token table agrees with
# them (C2), and count the pointing-model calls a request makes (C10).
# ---------------------------------------------------------------------------
section("pipeline (authority chain)")
try:
    from yiddish_phonikud import engine
    from yiddish_phonikud.api import routes_v1 as rv

    engine.load()

    # A sentence whose evidence-backed readings the inverted pipeline destroys:
    # לך ləxˈu -> lxu, אתה ˈatu -> ˈatə, רבותי rabˈɔjsaj -> rabˈɔjsaː.
    PROBE = "לך אתה רבותי האבן געזאגט א שיינע תשובה"
    raw_ipa = engine.text_to_ipa(PROBE)
    pointed = engine.text_to_nikud(PROBE)
    inverted_ipa = engine.text_to_ipa(pointed)
    check(
        inverted_ipa != raw_ipa,
        "the C1 discriminator still discriminates",
        f"text_to_ipa(raw)={raw_ipa!r} vs text_to_ipa(nikud)={inverted_ipa!r}"
        if inverted_ipa != raw_ipa
        else "the two pipelines now agree on this sentence — pick a new probe, "
             "this check has no teeth as written",
    )

    result = rv.analyze(PROBE, rv.InputForm.TEXT)
    check(
        result.phonemes == raw_ipa,
        "pipeline phonemes come from the RAW text (authority chain intact)",
        result.phonemes,
    )
    check(
        result.phonemes != inverted_ipa,
        "pipeline phonemes are NOT text_to_ipa(text_to_nikud(text))",
        "the v5 model's guesses are not fed back into the G2P",
    )
    check(
        result.nikud == pointed,
        "the pointing is still returned, for display",
        result.nikud,
    )

    # C2: the table must explain the transcription it is shown beside.
    stripped = [w.strip(".,!?;:()[]") for w in result.phonemes.split()]
    table = " ".join(row.ipa for row in result.tokens if row.ipa).split()
    check(
        table == [w for w in stripped if w],
        "the token table agrees with the phonemes word for word",
        f"{len(result.tokens)} rows == {len(table)} phoneme words",
    )

    # C4: hand-supplied pointing reaches the G2P and changes the reading. The
    # old build stripped every combining mark before G2P, so the Nikud tab was
    # decoration.
    pairs = (("מלך", "מֶלֶךְ"), ("סוכה", "סוּכָּה"))
    readings = []
    changed = True
    for plain, marked in pairs:
        bare = rv.analyze(plain, rv.InputForm.TEXT, with_tokens=False, with_nikud=False)
        hand = rv.analyze(marked, rv.InputForm.NIKUD, with_tokens=False, with_nikud=False)
        readings.append(f"{plain} {bare.phonemes} != {marked} {hand.phonemes}")
        changed = changed and bare.phonemes != hand.phonemes
    check(changed, "hand-applied nikud changes the reading (nothing is stripped)",
          "; ".join(readings))
    hand = rv.analyze("מֶלֶךְ", rv.InputForm.NIKUD)
    check(
        hand.nikud == "מֶלֶךְ",
        "NIKUD form echoes the caller's own pointing as the pointing of record",
        hand.nikud,
    )

    # C10: count real invocations of the v5 pointing model per pipeline call.
    labels = engine._labels
    calls = {"n": 0}
    real_point = labels.text_to_nikud

    def _counted(text, *a, **k):
        calls["n"] += 1
        return real_point(text, *a, **k)

    labels.text_to_nikud = _counted
    try:
        calls["n"] = 0
        rv.analyze(PROBE, rv.InputForm.TEXT)
        with_table = calls["n"]
        calls["n"] = 0
        rv.analyze(PROBE, rv.InputForm.TEXT, with_tokens=False, with_nikud=False)
        speech_path = calls["n"]
        calls["n"] = 0
        rv.analyze("מֶלֶךְ האט געזאגט", rv.InputForm.NIKUD)
        hand_path = calls["n"]
    finally:
        labels.text_to_nikud = real_point
    check(
        with_table == 1,
        "nikud + IPA + token table cost exactly ONE pointing pass",
        f"{with_table} call(s) — engine.analyze() feeds the table its pointing",
    )
    check(
        speech_path == 0,
        "the speech path runs the pointing model not at all",
        f"{speech_path} call(s) for a synthesis request",
    )
    check(
        hand_path == 0,
        "hand-pointed input runs the pointing model not at all",
        f"{hand_path} call(s) — the caller's marks are the pointing of record",
    )
except Exception as exc:  # noqa: BLE001
    dep = missing_dep(exc)
    if dep:
        skip("pipeline", f"{dep} not installed")
    else:
        check(False, "pipeline", repr(exc))


# ---------------------------------------------------------------------------
# 7. app routes — the census, both directions against docs/API.md
# ---------------------------------------------------------------------------
section("app")
try:
    from app import create_app

    app_ = create_app()

    def walk(routes: object) -> set[str]:
        """Every path in the tree.

        FastAPI >= 0.140 wraps an included router in a `_IncludedRouter` whose own
        `path` is "", so a flat scan of `app.routes` finds none of the /v1 routes.
        """
        found: set[str] = set()
        for route in routes or ():  # type: ignore[union-attr]
            path = getattr(route, "path", "")
            if path:
                found.add(path)
            for holder in (route, getattr(route, "original_router", None),
                           getattr(route, "router", None)):
                nested = getattr(holder, "routes", None)
                if nested:
                    found |= walk(nested)
        return found

    paths = walk(app_.routes)
    # The documented set is parsed out of docs/API.md rather than duplicated
    # here, so the census grows with the API instead of going stale. Headings
    # look like "### `GET /v1/audio/speech`" / "#### `POST /generate`".
    api_md = (ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    documented = {
        m.group(2)
        for m in re.finditer(
            r"^#{3,4}\s+`(GET|POST|PUT|PATCH|DELETE)\s+(/\S*)`", api_md, re.M
        )
    }
    check(
        documented >= set(CORE_ROUTES),
        "docs/API.md still documents every core route",
        f"{len(documented)} documented"
        if documented >= set(CORE_ROUTES)
        else f"UNDOCUMENTED: {', '.join(sorted(set(CORE_ROUTES) - documented))}",
    )
    expected = documented | set(CORE_ROUTES)
    absent = sorted(p for p in expected if p not in paths)
    check(
        not absent,
        "every documented route is exposed",
        f"{len(expected)} routes: {', '.join(sorted(expected))}" if not absent
        else f"MISSING: {', '.join(absent)}",
    )
    # And the other direction: a /v1 path nobody documented is a contract gap.
    undocumented = sorted(p for p in paths if p.startswith("/v1") and p not in expected)
    check(
        not undocumented,
        "no undocumented /v1 route is exposed",
        f"{sum(p.startswith('/v1') for p in paths)} /v1 paths" if not undocumented
        else f"UNDOCUMENTED: {', '.join(undocumented)}",
    )
    check("/docs" in paths and "/redoc" in paths, "OpenAPI docs mounted", "/docs /redoc")
    check(
        "/openapi.json" in paths and "/static" in paths,
        "schema and static assets mounted",
        "/openapi.json /static",
    )
    # C5: FastAPI's {"detail": ...} must never reach a client — the contract is
    # ErrorBody{error:{code,message}} on every failure, 404s included.
    from fastapi.exceptions import RequestValidationError
    from starlette.exceptions import HTTPException as StarletteHTTPException

    handlers = app_.exception_handlers
    check(
        RequestValidationError in handlers and StarletteHTTPException in handlers,
        "ErrorBody handlers installed for validation errors and HTTPException",
        ", ".join(sorted(getattr(k, "__name__", str(k)) for k in handlers)),
    )
except Exception as exc:  # noqa: BLE001
    dep = missing_dep(exc)
    if dep:
        skip("app", f"{dep} not installed")
    else:
        check(False, "app", repr(exc))


# ---------------------------------------------------------------------------
print()
if skips:
    print(f"{len(skips)} SKIPPED (dependency not installed): {', '.join(skips)}")
print(
    "ALL CHECKS PASSED"
    if not fails
    else f"{len(fails)} FAILURE(S): {', '.join(fails)}"
)
sys.exit(1 if fails else 0)
