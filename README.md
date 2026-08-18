---
title: Yiddish Phonikud TTS
emoji: 🕯️
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
models:
  - notmax123/phonikud-yi-engine
  - notmax123/BlueTTS2.5-onnx
tags:
  - yiddish
  - text-to-speech
  - grapheme-to-phoneme
  - g2p
  - nikud
  - diacritization
  - phonikud
  - ipa
  - low-resource
---

# Yiddish Phonikud TTS

A text-to-speech Space for contemporary **Hasidic (Unterland / Central) Yiddish**.
Yiddish text goes in; nikud, IPA and audio come out. The linguistic work is done by
[**phonikud-yi**](https://huggingface.co/notmax123/phonikud-yi-engine) — a deterministic
text → nikud → IPA stack adapted from *Phonikud*
([arXiv 2506.12311](https://arxiv.org/abs/2506.12311)) — and the waveform by
[**BlueTTS 2.5**](https://huggingface.co/notmax123/BlueTTS2.5-onnx), a flow-matching ONNX
model at 44.1 kHz with five saved voices. A 22.05 kHz Piper voice stays in the image as a
lightweight fallback. Read [Limitations](#limitations) before you judge the audio.

The engine (~1.23 GB, including the v5 pointing model) and the acoustic bundle (~282 MB) are
pulled with `huggingface_hub` on a background thread at startup, so the page serves
immediately and the first synthesis request waits for the warm-up to finish.

## Why Yiddish needs a G2P stack at all

Yiddish orthography is a hybrid. The Germanic component (~75 % of running text) is spelled
essentially phonemically, so rules handle it. The **loshn-koydesh** component — words of
Hebrew/Aramaic origin — keeps its historical Hebrew spelling, unvocalized, while the Yiddish
pronunciation drifted centuries away from it. Read letter for letter, `שבת` gives `ʃbs` — no
vowels at all, because the spelling does not write them, and no clue that the `ת` is said `s`
in Yiddish. This
stack answers with a lexicon of native verdicts rather than a morpho-phonological analyzer,
so those two words come out as they are actually said:

```
שבת       → ʃˈabəs
בעל-הבית  → bˈaləbus
מדרש      → mˈɛdrəʃ
יארצייט   → jˈurʦajt
```

Every IPA string in this file that is presented as this stack's output was produced by running
the engine, not written by hand. The counter-examples are labelled as such where they appear:
`ʃbs` (שבת read letter for letter), `fɛkl` (פעקל with the lexicon tables missing), `hot` (another
dialect's reading), and `tʃ`/`dʒ` (what `phones.fold_to_vocab` writes for the Piper voice, not
what the G2P emits).

## Three input modes

The UI and the API accept the same three entry points into the pipeline:

| Mode | Input | What runs |
|---|---|---|
| `text` | plain Yiddish, `מיט א פאר יאר צוריק` | G2P (which points the text internally) → synthesis |
| `nikud` | already-pointed Yiddish, `מִיט אַ פּאָר יאָר צוּרִיק` | G2P over your pointing → synthesis |
| `phonemes` | IPA, `mit a pˈur jur ʦirˈik` | synthesis only |

The G2P reads diacritics as evidence wherever it finds them, so hand-pointing is not
decoration: `א` alone is `a`, `אָ` is `u`; `פאר` is `far`, `פּאר` is `par`. Where your text is
unpointed the engine's own authority chain decides, and the v5 pointing model's output is
shown for inspection rather than fed forward.

Hand-written IPA is still validated against the closed inventory and folded to the voice's
vocabulary — nothing bypasses those checks.

## The closed phone inventory

Nothing outside this set is ever emitted; the source project enforces it corpus-wide.

- **vowels** `a aː ɛ ə i u ɔ ej aj ɔj oʊ`
- **consonants** `b d f ɡ h j k l m n p r s t v z x ʃ ʒ ʦ ʧ ʤ ŋ`
- **marks** `ˈ`, immediately before the stressed vowel, at most one per word

Length is not a separate unit: `aː` is one inventory unit and it is the only place `ː` ever
occurs, so a bare `ː` after any other vowel is a violation and is reported as one. `ŋ` is
declared by the spec but unattested in engine output — `זינגען` is `zˈinɡən`.

`ɡ` is U+0261 (script g), not ASCII `g`, and `ʦ ʧ ʤ` are single codepoints (U+02A6, U+02A7,
U+02A4), not the sequences `ts tʃ dʒ`. `GET /v1/phonemes/inventory` returns the live set plus
the units the loaded voice's vocabulary lacks. For **BlueTTS 2.5 that list is empty**: its
character vocab carries every unit of the inventory, marks included, so every phone reaches the
model unfolded and none is dropped. (Phones. A few punctuation characters the engine passes
through — `[ ] ׃ „ ‚ ‹ › | < > { }` — are outside that vocab and are removed silently; they
carry no phonetic content and are not reported.) For the **Piper fallback** the list is `ʧ` and
`ʤ`, which are folded to `tʃ` and `dʒ` before synthesis. A fold is not a loss, so it is reported
here and in `runtime_vocab_missing`, not in `unsupported` — `unsupported` names only units that
reached no model at all.

## API

Interactive docs at `/docs` (ReDoc at `/redoc`), health at `/health`. Full reference with
request/response fields and working curl commands: [`docs/API.md`](docs/API.md).

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness, warm-up state, and warm-up failures — never 503 |
| `GET /v1/models/sources` | the runtime catalog |
| `POST /v1/models/load` | load a runtime by id |
| `GET /v1/models/state` | loaded runtime, model path, sample rate |
| `GET /v1/voices` | voices of the loaded runtime |
| `GET /v1/languages` | `yi` only |
| `GET /v1/phonemes/inventory` | the inventory + the voice's gaps |
| `POST /v1/audio/diacritize` | text → nikud |
| `POST /v1/audio/phonemize` | text → nikud + IPA + per-token table |
| `POST /v1/audio/speech` | WAV, or a framed stream |

Speech, on the default runtime (44.1 kHz):

```bash
curl -X POST https://<space-host>/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "מיט א פאר יאר צוריק", "voice": "libri_male_6209", "speed": 1.0}' \
  -o out.wav
```

Phonemize, with the token table that explains every decision:

```bash
curl -X POST https://<space-host>/v1/audio/phonemize \
  -H 'Content-Type: application/json' \
  -d '{"input": "מיט א פאר יאר צוריק"}'
```

The response below is real engine output for that sentence, copied from a run:

```json
{
  "nikud": "מִיט אַ פּאָר יאָר צוּרִיק",
  "phonemes": "mit a pˈur jur ʦirˈik",
  "tokens": [
    {"word":"מיט","nikud":"מִיט","ipa":"mit","route":"lexicon","confidence":"HIGH","layer":"G","reason":""},
    {"word":"א פאר","nikud":"אַ פּאָר","ipa":"a pˈur","route":"lexicon","confidence":"HIGH","layer":"L","reason":""},
    {"word":"יאר","nikud":"יאָר","ipa":"jur","route":"lexicon","confidence":"HIGH","layer":"G","reason":""},
    {"word":"צוריק","nikud":"צוּרִיק","ipa":"ʦirˈik","route":"lexicon","confidence":"HIGH","layer":"G","reason":""}
  ],
  "unsupported": []
}
```

Five input words, **four rows**: `א פאר` is a multiword lexicon entry, so it comes back as one
record whose `word` is the joined spelling. That entry is why the reading is `pˈur` ("a few")
and not `far` ("for").

`route`, `confidence` and `layer` are the engine's own verdicts, and they mean exactly what
`yiddish_g2p.g2p_token` says they mean:

- **`lexicon`** — a whole-token table hit: the abbreviation table, a multiword entry, the
  native-verified gold lexicon, or a legacy merged-loshn-koydesh / high-frequency list.
- **`rule`** — no table knew the word, so the Germanic or loshn-koydesh rule path derived it
  (`שטיקלעך` → `ʃtˈikləx`, MED).
- **`fallback`** — the engine judged its own output unfit to emit and quarantined it: a
  vowel-less loshn-koydesh skeleton, an unlexiconed unpointed LK word, or an out-of-inventory
  token such as a number or a URL. A quarantined token contributes only its punctuation to the
  spoken string, so `דער טעלעפאן נומער איז 5` speaks as `dɛr tˈɛləfan nˈimər iz` — the digit is
  not read aloud.
- **`HIGH`** is a lexicon hit; **`MED`** an unambiguous rule derivation; **`LOW`** the least
  certain tier — a defaulted ambiguous `א`/`פ` (`וואלד` → `vald`, reason `alef-default`), a
  rescued loshn-koydesh form, a corpus-mined collocation (`בעל-הבית` → `bˈaləbus`, reason
  `mwe-mined`), or an inventory violation. **LOW is the human-verification queue, not an
  error.**
- **`layer`** is `G` Germanic, `L` loshn-koydesh, `E` loanword, `N` proper name, `A`
  abbreviation, `X` unclassified.

`response_format` must be `wav`; anything else is a 400. Errors are always
`{"error": {"code": ..., "message": ...}}`.

### Streaming

Send `"stream": true` and the response becomes `application/octet-stream` carrying
length-prefixed frames:

```
[kind: u8][length: u32 big-endian][payload: length bytes]
```

`kind = 1` is a self-contained WAV chunk (one sentence-sized piece, split so a word or a
multiword lexicon entry is never cut in half), `kind = 2` is the full concatenated WAV at the
end, and `kind = 3` is a UTF-8 error message if synthesis fails mid-stream. Full decoding
rules and a Python client are in [`docs/API.md`](docs/API.md#streaming-frame-format).

## Runtime catalog

`GET /v1/models/sources` returns a MamboTTS-style catalog. Each manifest carries both
`installed` (are the required files present?) and `available` (does this build implement the
runtime?):

- **`blue_yi`** — *the default.* BlueTTS 2.5, ~282 MB, **44.1 kHz**, five saved voices:
  `female`, `libri_female_1088`, `libri_female_6147`, `libri_male_6209`, `libri_male_8088`.
  Yiddish is one of the checkpoint's declared languages and its latent statistics were
  exported from `stats_yiddish.pt`. Its character vocab covers the Yiddish inventory outright.
  Two extra options are accepted: `n_steps` (default 8) and `cfg_scale` (default 4.0).
- **`piper_yi`** — *the fallback.* The Piper ONNX voice committed beside `app.py`, 61 MB,
  **22.05 kHz**, single speaker, and much cheaper to load. It ignores `n_steps` and
  `cfg_scale`.

Sample rate therefore depends on the loaded runtime; `GET /v1/models/state` reports it and
nothing should hard-code it.

`n_steps` is how many flow-matching steps the vector estimator runs — more steps sound better
and take longer. Measured on this machine for about 1.2 s of audio: **4 steps 121 ms, 8 steps
178 ms, 16 steps 295 ms**, on top of a one-off ~1.5 s session build at startup. `cfg_scale` is
classifier-free guidance strength; guidance costs a second model pass per step, so
`cfg_scale = 1.0` disables it and roughly halves the work.

## Where the labels' authority comes from

Fixed chain, highest tier first; a lower tier never overrides a higher one.

1. **Native-speaker verdicts** — **509 gold words**, byte-identity enforced by a test gate in
   the source repo. Nothing may move a gold primary.
2. **Corpus audio** — a phone recognizer over episode audio, and only at graphemes the
   spelling genuinely leaves open (`א`, `פ`, `יי`, `וי`, shuruk-`ו`). Elsewhere the letter
   decides, so an audio deviation is a process, not evidence. Survivors ship at MED.
3. **Published pointing** (Sefaria) — LOW confidence, always queued.
4. **The v5 model's contextual guess** — LOW confidence, always queued.

When audio contradicts gold, the conflict becomes a question for the native reviewer, never a
silent flip. The G2P loads its knowledge from **7 generated lexicon tables**, and
`yiddish_labels.verify()` runs at import to assert every one of them loaded — an incomplete
deployment would otherwise emit plausible-looking IPA with zero native verdicts, silently
(`פעקל` is `pɛkl` with the tables, `fɛkl` without). The labels themselves are **citation
forms, not surface forms**: `האט` stays `hut` even where fast speech reduces it.

The pointing model is **phonikud-yi v5**, finetuned on labels repaired under that chain over a
**1.83 M-token** corpus of contemporary Hasidic Yiddish.

Because the chain runs inside the engine, the API's pipeline is `text → IPA` in one pass. The
nikud you see in a response is that pass's *display* output: it is not fed back into the G2P,
which would let a tier-4 model guess re-decide readings the gold lexicon had already settled.

## Limitations

- **The voices are not Yiddish readers.** BlueTTS 2.5 knows the phones — every unit of the
  Yiddish inventory is in its vocab, so nothing is folded or dropped — but all five bundled
  speakers are Hebrew or English readers. Expect a foreign accent, mostly in vowel colour and
  rhythm. A natively-read Yiddish voice would need new recordings and is future work.
- **No accuracy number is claimed, here or anywhere in this Space.** The source project
  records that no measured word-error or accuracy figure exists for this stack; the figures
  quoted informally there are estimates, not measurements. Any number presented as measured
  would be a fabrication. The per-token table exists so you can audit readings yourself
  instead of trusting an aggregate.
- **The Piper fallback is Hebrew-trained.** `model.onnx` is a Piper voice trained on Modern
  Hebrew speech and driven with Yiddish IPA. It is not a Yiddish voice: expect Hebrew-inflected
  vowel quality, prosody and rhythm, and `ʧ`/`ʤ` folded to `tʃ`/`dʒ` because its
  `phoneme_id_map` lacks them. Every fold is reported in `unsupported`.
- **One dialect.** Hasidic Unterland/Central Yiddish only. Litvish, Poylish and YIVO-standard
  readings are wrong here by design (`hut`, not `hot`).
- **LOW-confidence tokens are the review queue.** Unsettled words get the engine's best answer
  rather than silence, tagged LOW so they stay visible. Check the token table before trusting
  a word.
- **Non-Yiddish tokens are quarantined, not read.** Digits, Latin text and URLs are dropped
  from the spoken string — there is no Yiddish numeral reader yet — and so are single-letter
  geresh abbreviations (`ד'`, `ס'`).
- **Long text is chunked, on every path.** BlueTTS 2.5 renders one utterance per call and
  refuses text past 240 encoder tokens — the point where its duration predictor stops
  lengthening the utterance and starts cramming — so text is split per sentence at 200
  characters and the pieces are concatenated with a 60 ms gap. Streaming emits those same
  chunks one by one; `stream` changes delivery, never the audio. A request is capped at 4000
  characters, about 3.5 minutes of speech.
- **No voice cloning.** `voice_reference` is `false` for both runtimes. Blue's bundle ships
  frozen style vectors and no autoencoder encoder, so there is no way to make a sixth voice
  from a recording here.
- **Cold start is slow.** The first request after a restart waits on the engine and acoustic
  downloads; `GET /health` reports `engine_loaded`, `runtime_loaded` and any warm-up error, so
  you can poll for readiness and tell warming from broken.

## Credits

- **Phonikud** — Overcoming Phonetic Underspecification for Hebrew Text-To-Speech
  ([arXiv 2506.12311](https://arxiv.org/abs/2506.12311)), the method this stack adapts.
- **phonikud-yi** — the Yiddish G2P engine, gold lexicon and v5 pointing model:
  [notmax123/phonikud-yi-engine](https://huggingface.co/notmax123/phonikud-yi-engine).
- **BlueTTS 2.5** — the default acoustic model:
  [notmax123/BlueTTS2.5-onnx](https://huggingface.co/notmax123/BlueTTS2.5-onnx).
- **Piper** — the fallback ONNX synthesis runtime, via `piper-onnx`.
- The gold lexicon exists because a native speaker sat down and ruled on it word by word.
