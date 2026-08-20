# Yiddish Phonikud TTS — HTTP API

Hasidic (Unterland/Central) Yiddish text-to-speech. Hebrew-script Yiddish goes through the
[`notmax123/phonikud-yi-engine`](https://huggingface.co/notmax123/phonikud-yi-engine) stack
(G2P over a closed phone inventory, with a v5 pointing model for display), and the resulting
IPA drives the acoustic runtime,
[**blue-yi**](https://huggingface.co/notmax123/blue-yi), at 44.1 kHz.

> **What the voices are.** blue-yi declares Yiddish among its languages and its latent
> statistics were exported from `stats_yiddish.pt`, and its character vocab covers the entire
> closed Yiddish inventory — `ʦ ʧ ʤ ɡ ŋ ˈ` and the `aː` length mark included — so **every phone
> reaches the model unfolded and none is dropped**. (That is a claim about phones. A handful of
> punctuation characters the engine passes through — `[ ] ׃ „ ‚ ‹ › | < > { }` — are outside the
> character vocab and are removed on the way in; they carry no phonetic content and are not
> reported.) The caveat is narrower than "wrong
> language": all four bundled voices are English readers, so a foreign accent is
> likely, mostly in vowel colour and rhythm.
>
> **No accuracy figure is published anywhere in this API or its docs.** The source project
> records that no measured word-error or accuracy number exists for this stack. Verified
> engineering figures are fair game (509 native-verified gold words, 7 lexicon tables, a
> 1.83 M-token corpus, 44.1 kHz, the timings below); a quality percentage is not.

The API shape is ported from MamboTTS (`mambotts-server`): same field names, same error
envelope, same code strings, same streaming frame protocol. An existing MamboTTS client should
recognise every payload.

- Base URL in the Space: `https://<space-host>/`
- Base URL locally: `http://localhost:7860` (`python app.py --host 0.0.0.0 --port 7860`)
- Interactive docs: `/docs` (Swagger) and `/redoc`
- Content type: `application/json` in and out, except `POST /v1/audio/speech` (audio bytes) and
  `POST /generate` (form in, JSON out).

Cold start: the first startup downloads the ~1.23 GB engine snapshot and the ~281 MB acoustic
bundle. `app.py` does both on a background thread, so `GET /` and `GET /health` answer
immediately while the warm-up runs. Set `PHONIKUD_YI_ENGINE_DIR` to an unpacked engine bundle to
skip the engine download entirely.

---

## Contents

- [Errors](#errors)
- [Input modes](#input-modes)
- [Runtimes, voices and sample rates](#runtimes-voices-and-sample-rates)
- [Endpoints](#endpoints)
  - [`GET /health`](#get-health)
  - [`GET /v1/models/sources`](#get-v1modelssources)
  - [`POST /v1/models/load`](#post-v1modelsload)
  - [`GET /v1/models/state`](#get-v1modelsstate)
  - [`GET /v1/voices`](#get-v1voices)
  - [`GET /v1/languages`](#get-v1languages)
  - [`GET /v1/phonemes/inventory`](#get-v1phonemesinventory)
  - [`POST /v1/audio/diacritize`](#post-v1audiodiacritize)
  - [`POST /v1/audio/phonemize`](#post-v1audiophonemize)
  - [`POST /v1/audio/speech`](#post-v1audiospeech)
  - [`GET /` and `POST /generate` (UI)](#ui-endpoints)
- [Streaming frame format](#streaming-frame-format)
- [Phone inventory](#phone-inventory)
- [Verifying a deployment](#verifying-a-deployment)

---

## Errors

Every non-2xx response uses one envelope:

```json
{ "error": { "code": "invalid_request", "message": "only wav response_format is supported" } }
```

| `code` | HTTP | Meaning | Typical cause |
| --- | --- | --- | --- |
| `invalid_request` | 400 | The request itself is wrong. | Empty `input`; `input` longer than 4000 characters; input that phonemizes to nothing (digits, a URL, a lone geresh word — the G2P quarantines those, so there is nothing to speak); `response_format` other than `wav`; `speed` outside `[0.5, 2.0]`; `n_steps` outside `[1, 32]`; `cfg_scale` outside `[1.0, 8.0]`; `seed` outside `[0, 2^31-1]`; an unknown `runtime` id; an unknown `voice` name; an unknown `mode` on `POST /generate`; a single unsplittable run of words longer than the acoustic model can render; a malformed JSON body. |
| `no_model` | 503 | No TTS runtime is loaded and none could be loaded. | A synthesis request arrived before warm-up finished, or the runtime's files are missing. |
| `not_available` | 503 | The runtime exists in the catalog but cannot serve here, **or** the box is at its concurrency limit. | A runtime with `available: false`, one whose required model files are absent, or 2 syntheses already in flight with no slot free within 30 s (`PHONIKUD_YI_MAX_CONCURRENCY`). |
| `not_found` | 404 | No such route. | A typo in the path. |
| `method_not_allowed` | 405 | Route exists, wrong verb. | `GET /v1/audio/speech`. |
| `internal_error` | 500 | Unexpected failure inside the pipeline. | G2P or synthesis raised; engine snapshot incomplete (a missing lexicon table is fatal by design — see [`GET /health`](#get-health)). |

Body-validation failures are translated into the same envelope with
`code: "invalid_request"`; FastAPI's own `{"detail": ...}` shape is never returned.

Mid-stream failures cannot use the envelope — the status line is already sent. They arrive as a
kind-3 frame carrying the message as UTF-8 text; see [Streaming frame format](#streaming-frame-format).

---

## Input modes

The pipeline, in order:

1. **G2P** – `yiddish_labels.text_to_ipa` over the seven lexicon tables → IPA. It runs the
   engine's full authority chain internally (native gold verdicts > corpus audio > published
   pointing > model guess) and reads any diacritics present in the input as evidence.
2. **validate** – every unit is checked against the closed inventory. Caller-supplied IPA that
   fails is a `400`, because there it can only be a caller error; an off-inventory unit in
   *engine* output is reported in `unsupported` and still spoken, because refusing would turn a
   G2P quirk into an outage.
3. **fold** – rewrite units the loaded voice's vocabulary lacks. A no-op on blue-yi, whose
   vocabulary lacks none of them, so nothing this build ships ever folds. A **phone** that
   cannot be folded is dropped
   and reported in `unsupported` / `X-Dropped-Units`. Punctuation the vocabulary cannot spell is
   dropped silently and never reported: it is structure, not a phone, so nothing audible is lost.
4. **chunk** – long text is split per sentence (200 characters, never inside a multiword lexicon
   entry) because the acoustic model renders one utterance per call. This happens on the
   streaming and non-streaming paths alike.
5. **synthesize** – the serving runtime, one chunk at a time, joined with a 60 ms gap.

There is **no mark-stripping stage**. An earlier version of this document claimed the G2P was
"trained on undotted orthography" so that "nikud would miss every lexicon entry"; that was
false. The engine handles all three spelling systems (unpointed Hasidic, pointed YIVO, and
fully pointed nikud), its lexicon keys are point-stripped, and points steer real decisions:

```
א → a          אָ → u            (which vowel the alef is)
פאר → far      פּאר → par        (which consonant the pe is)
מלך → mˈajləx  מֶלֶךְ → mˈɛləx     (merged vs Whole-Hebrew loshn-koydesh reading)
```

Pointing is therefore passed to the G2P verbatim in every mode.

The v5 pointing model is **display only**. Its output is what `nikud` returns, and it is *not*
fed back into the G2P: doing so would let a tier-4 model guess re-decide readings the gold
lexicon had already settled, which is how a released voice ends up mixing dialects
(`hut`/`hot`/`hat` for `האט`).

`SpeechBody` picks the entry point with two flags. `input_is_phonemes` wins over
`input_is_nikud` when both are set.

| Mode | Flags | Runs | Skips |
| --- | --- | --- | --- |
| **text** (default) | both flags `false` | 1 → 4, plus the pointing model for the displayed `nikud` | nothing |
| **nikud** | `input_is_nikud: true` | 1 → 4 | the pointing model. Your pointing is echoed back as `nikud` **and** read by the G2P, so hand-correcting a point changes the pronunciation. |
| **phonemes** | `input_is_phonemes: true` | 2 → 4 | the G2P and the pointing model. Validation and folding are *never* skipped: off-inventory IPA is a 400, and units the voice lacks are still folded. |

`POST /v1/audio/phonemize` takes the same `input_is_nikud` flag, so a client can show the trace
for hand-pointed text before speaking it.

Practical consequences:

- **nikud** mode is the way to override a reading the G2P got wrong without writing IPA: point
  the word the way you want it read and send it back.
- **phonemes** mode is the only way to bypass the engine entirely, so it is also the only mode
  that works before the engine snapshot has finished downloading.

---

## Runtimes, voices and sample rates

One runtime ships. Its sample rate is still not something to hard-code — read it from
`GET /v1/models/state`, which is also where a future second runtime would show up.

| | `blue_yi` (default, and the only one) |
| --- | --- |
| Model | blue-yi ONNX, flow-matching |
| Sample rate | **44100 Hz** |
| On disk | ~281 MB, fetched from `notmax123/blue-yi` |
| Voices | `Berl`, `Hershl`, `Rukhl`, `Sheyndl` |
| Yiddish phones | complete: nothing folded, nothing dropped |
| Extra options | `n_steps` (8), `cfg_scale` (4.0), `seed` |

`n_steps` is how many flow-matching steps the vector estimator runs — more steps sound
marginally better and cost linearly more CPU. Measured on this hardware for about 1.2 s of
audio: **4 steps 121 ms, 8 steps 178 ms, 16 steps 295 ms**, plus a one-off ~1.5 s ONNX session
build at warm-up. `cfg_scale` is classifier-free guidance strength; guidance needs a second
model pass per step, so `cfg_scale: 1.0` skips it and roughly halves the cost.

`POST /v1/models/load` sets the runtime for the process and `SpeechBody.runtime` sets it for
one request; with one runtime in the catalog both are only ever used to load `blue_yi`
explicitly, and any other id is a `400 invalid_request`.

```bash
curl -s -X POST http://localhost:7860/v1/models/load \
  -H 'Content-Type: application/json' -d '{"runtime": "blue_yi"}'
```

---

## Endpoints

### `GET /health`

Liveness and warm-up progress. **Never returns a non-200 status** — a cold Space is alive even
though it is still downloading, and an orchestrator restarting it for that would never let it
finish.

Response — `HealthResponse`:

| Field | Type | Notes |
| --- | --- | --- |
| `status` | `string` | `warming` (the warm-up thread is still running), `ready` (engine **and** a runtime in memory), or `error` (warm-up finished and failed). |
| `engine_loaded` | `bool` | The snapshot is on disk and `yiddish_labels` imported, which means all seven lexicon tables verified. |
| `runtime_loaded` | `bool` | A TTS runtime is in memory. The warm-up loads the default runtime, so this becomes true on an idle Space without any request. |
| `runtime` | `string` | Loaded runtime id, `""` if none. |
| `engine_error` | `string` | The engine warm-up failure verbatim, or `""`. |
| `runtime_error` | `string` | The runtime warm-up failure verbatim, or `""`. |
| `warming` | `bool` | The warm-up thread is still running — lets a client tell `warming` from `error` without matching strings. |
| `version` | `string` | `yiddish_phonikud.__version__`. |

```bash
curl -s http://localhost:7860/health
```

```json
{"status":"ready","engine_loaded":true,"runtime_loaded":true,"runtime":"blue_yi","engine_error":"","runtime_error":"","warming":false,"version":"0.1.0"}
```

A deployment that would rather keep an idle box small can set `PHONIKUD_YI_WARM_RUNTIME=0`;
the acoustic runtime is then loaded by the first synthesis request instead of at warm-up, and
`ready` means "engine resident, runtime on demand".

`status: "error"` never clears on its own — alert on it instead of polling. A non-empty
`engine_error` is the case worth watching: the import runs `yiddish_labels.verify()`, which
raises when any of the seven tables is missing, deliberately fatal because `yiddish_g2p`
otherwise swallows a missing table and returns plausible-but-wrong Yiddish (`פעקל` as `fɛkl`
instead of `pɛkl`, `יארצייט` as `jˈarʦajt` instead of `jˈurʦajt`) with zero native verdicts and
no warning.

---

### `GET /v1/models/sources`

The runtime catalog: what exists, what is installed on this machine, what this build can
actually load. Port of MamboTTS `sources.rs`.

Response — `ModelSourcesResponse`:

| Field | Type | Notes |
| --- | --- | --- |
| `runtimes[]` | `ModelSource` | One per registry entry, in display order (default first). |
| `runtimes[].id` | `string` | Pass to `POST /v1/models/load` and `SpeechBody.runtime`. |
| `runtimes[].name`, `.version`, `.size`, `.description` | `string` | Display metadata; `description` carries the runtime's caveats. |
| `runtimes[].files[]` | `{name, url}` | Files required to install it. |
| `runtimes[].directory` | `string` | Directory the files live in under the model root (`"."` = repo root; for a Hub-hosted bundle it is only a hint for a manual install). |
| `runtimes[].installed` | `bool` | Every required file is present here. |
| `runtimes[].available` | `bool` | `false` = declared in the catalog but not implemented in this build; loading it returns 503 `not_available`. |
| `runtimes[].capabilities` | `{yiddish, streaming, voice_reference, fixed_voices}` | All booleans. `voice_reference` is `false`: blue-yi's bundle ships frozen style vectors and no autoencoder encoder, so no fifth voice can be made from a recording. |
| `engine_repo` | `string` | `notmax123/phonikud-yi-engine`. |
| `default_paths[]` | `string` | Filesystem locations searched for model files. |

```bash
curl -s http://localhost:7860/v1/models/sources \
  | jq '.runtimes[] | {id, size, installed, available, fixed: .capabilities.fixed_voices}'
```

```json
{"id":"blue_yi","size":"~281 MB","installed":true,"available":true,"fixed":true}
```

One object, because the catalog has one entry.

---

### `POST /v1/models/load`

Loads (or swaps to) a runtime. Idempotent: loading the runtime that is already loaded is a
no-op that still returns 200. Model construction is serialized behind a lock, but the lock is
never held across synthesis, so in-flight requests are unaffected.

Request — `LoadBody`: `runtime` (`string`, default `"blue_yi"`) — an id from
`GET /v1/models/sources`.

Response — `LoadResponse`: `status` (always `"loaded"`), `runtime`, `model`.

```bash
curl -s -X POST http://localhost:7860/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"runtime": "blue_yi"}'
```

```json
{"status":"loaded","runtime":"blue_yi","model":"blue-yi (Yiddish IPA)"}
```

Errors: `400 invalid_request` (unknown id), `503 not_available` (`available: false`, or required
files missing), `500 internal_error` (the model failed to construct).

---

### `GET /v1/models/state`

What the process holds in memory right now. Safe before anything is loaded.

Response — `StateResponse`: `loaded` (`bool`), `runtime`, `model`, `path` (`string`),
`sample_rate` (`int`, `0` when unloaded — 44100 for `blue_yi`).

`path` is whatever the runtime loaded *from*: `blue_yi` reports the snapshot **directory**
holding its four graphs, not a file. Do not assume an `.onnx` suffix.

```bash
curl -s http://localhost:7860/v1/models/state
```

```json
{"loaded":true,"runtime":"blue_yi","model":"blue-yi (Yiddish IPA)","path":"/root/.cache/huggingface/hub/models--notmax123--blue-yi/snapshots/468da64b4a51795a7594a3637727dbaf876b6df2","sample_rate":44100}
```

---

### `GET /v1/voices`

Voices offered by the **loaded** runtime, and the exact set `SpeechBody.voice` accepts. An
unknown name is a 400 that lists the valid ones, never a silent fallback to the default.

Response — `VoicesResponse`: `runtime` (`string`), `voices` (`string[]`).

```bash
curl -s http://localhost:7860/v1/voices
```

```json
{"runtime":"blue_yi","voices":["Berl","Hershl","Rukhl","Sheyndl"]}
```

The four names are blue-yi's saved styles; the demo UI's voice picker is populated from
exactly this response. Returns 503 `no_model` when nothing is loaded yet — check `/health` to
tell warming from broken.

---

### `GET /v1/languages`

Response — `LanguagesResponse`: `languages` (`string[]`) and `items` (`{code, name}[]`, for UI
menus). This build is Yiddish-only; there is no language parameter anywhere in the API.

```bash
curl -s http://localhost:7860/v1/languages
```

```json
{"languages":["yi"],"items":[{"code":"yi","name":"Yiddish (Hasidic)"}]}
```

---

### `GET /v1/phonemes/inventory`

The closed phone inventory, plus what the loaded voice cannot say.

Response — `PhonemeInventoryResponse`:

| Field | Type | Notes |
| --- | --- | --- |
| `vowels[]` | `string` | Vowel and diphthong units, `aː` and `oʊ` included. |
| `consonants[]` | `string` | Consonant units. |
| `marks[]` | `string` | `["ˈ"]`. `ː` is **not** a unit: `aː` is one unit and a bare `ː` is a violation. |
| `inventory[]` | `string` | Every unit, sorted — the full closed set. |
| `runtime_vocab_missing[]` | `string` | Inventory units absent from the loaded voice's vocabulary. **Always `[]` for the runtime this build ships**: blue-yi's vocab covers the whole inventory, so nothing is ever folded. It is also `[]` when nothing is loaded, where it says nothing at all — check `/v1/models/state` to tell the two apart. |

```bash
curl -s http://localhost:7860/v1/phonemes/inventory | jq '{marks, runtime_vocab_missing}'
```

```json
{"marks":["ˈ"],"runtime_vocab_missing":[]}
```

---

### `POST /v1/audio/diacritize`

Pointing only, without the G2P — the v5 model's output, for display or hand-correction.

Request — `PhonemizeBody`: `input` (`string`, required) — Yiddish in Hebrew script.

Response — `DiacritizeResponse`: `nikud` (`string`).

```bash
curl -s -X POST http://localhost:7860/v1/audio/diacritize \
  -H 'Content-Type: application/json' \
  -d '{"input": "מיט א פאר יאר צוריק"}'
```

```json
{"nikud":"מִיט אַ פּאָר יאָר צוּרִיק"}
```

Note `אַ פּאָר` ("a few") and not `פֿאַר` ("for"): the v5 model resolves that homograph. This string
is a display artefact — the IPA for the same sentence is produced from the raw input, not from
this output.

---

### `POST /v1/audio/phonemize`

The full linguistic trace: pointing, IPA, and a per-token record of how each reading was
decided. `phonemes` and `tokens` come from **one** pass, so they cannot contradict each other,
and `phonemes` is exactly the string `/v1/audio/speech` would speak for the same request.

Request — `PhonemizeBody`:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `input` | `string` | — | Required. Pointed or unpointed; pointing is honoured, never stripped. |
| `input_is_nikud` | `bool` | `false` | Echo your own pointing back as `nikud` and skip the v5 model. It does not change how the G2P treats the pointing — points in `input` are always read. |

Response — `PhonemizeResponse`:

| Field | Type | Notes |
| --- | --- | --- |
| `nikud` | `string` | The v5 pointing, or your own when `input_is_nikud` is set. **Display only** — `phonemes` is not derived from it. |
| `phonemes` | `string` | IPA over the closed inventory, stress `ˈ` immediately before the stressed vowel. |
| `tokens[]` | `TokenRowDTO` | The G2P trace, in input order, from the same pass. |
| `unsupported[]` | `string` | Units outside the closed inventory, deduped, in order. Empty is the expected case; anything here is a G2P bug worth reporting. |

`TokenRowDTO`:

| Field | Type | Notes |
| --- | --- | --- |
| `word` | `string` | The token as spelled in the input. **A multiword lexicon entry is one row whose `word` is the joined spelling** — do not assume one row per whitespace token. |
| `nikud` | `string` | The same token pointed, for display. `""` when the pointed text could not be aligned to this token (a blank beats a confidently wrong pointing). |
| `ipa` | `string` | The token's reading. On a `fallback` row this is a flagged approximation that is **excluded** from `phonemes`. |
| `route` | `string` | `lexicon` — a whole-token table hit (abbreviation table, multiword entry, the native-verdict gold lexicon, the legacy merged-LK / high-frequency / loan lists, or an audio-confirmed correction). `rule` — the Germanic or loshn-koydesh rule path derived it. `fallback` — **quarantine**: the engine judged its own output unfit to emit (a vowel-less LK consonant string, an unlexiconed unpointed LK word, an out-of-inventory token such as a number or a URL), so only the token's punctuation reaches `phonemes`. |
| `confidence` | `string` | `HIGH` a lexicon hit; `MED` an unambiguous rule application; `LOW` a defaulted ambiguous `א`/`פ`, an evidence-rescued or LK-fallback reading, a corpus-mined collocation, or an inventory violation. **`LOW` is the human-verification queue, not an error.** |
| `layer` | `string` | `G` Germanic, `L` loshn-koydesh, `E` loanword, `N` proper name, `A` abbreviation/acronym, `X` unclassified. |
| `reason` | `string` | Short engine note naming the evidence or the defect: `alef-default`, `pe-default`, `mwe-mined`, `sefaria-pointed`, `audio-homograph`, `lk-fallback`, `bad-phone`, … Empty for a plain lexicon hit or a clean rule application. |

```bash
curl -s -X POST http://localhost:7860/v1/audio/phonemize \
  -H 'Content-Type: application/json' \
  -d '{"input": "מיט א פאר יאר צוריק"}'
```

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
record whose `word` is the joined spelling and whose `nikud` consumes both pointed tokens. That
entry is why the reading is `pˈur` ("a few years") and not `far` ("for") — the homograph is
resolved by the multiword table, not by the single word.

Three more real rows, to show the other routes (all four IPA strings below are engine output):

```json
{"word":"שטיקלעך","nikud":"שְׁטִיקְלֶעךְ","ipa":"ʃtˈikləx","route":"rule","confidence":"MED","layer":"G","reason":""}
{"word":"וואלד","nikud":"וואלְד","ipa":"vald","route":"rule","confidence":"LOW","layer":"G","reason":"alef-default"}
{"word":"בעל-הבית","nikud":"בַּעַל-הַבַּיִת","ipa":"bˈaləbus","route":"lexicon","confidence":"LOW","layer":"L","reason":"mwe-mined"}
{"word":"5","nikud":"5","ipa":"5","route":"fallback","confidence":"LOW","layer":"G","reason":"bad-phone"}
```

The last one is quarantine in action: for `דער טעלעפאן נומער איז 5`, `phonemes` is
`dɛr tˈɛləfan nˈimər iz` — the digit has a row in the table but is not spoken, because there is
no Yiddish numeral reader in the engine.

---

### `POST /v1/audio/speech`

Synthesis. Returns one `audio/wav` body, or a framed binary stream when `stream: true`.

Request — `SpeechBody`:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `input` | `string` | — | Required, non-empty, ≤ 4000 chars (`dto.MAX_INPUT_CHARS`). Yiddish text, pointed Yiddish, or IPA depending on the flags. Longer input is `400 invalid_request`: 4000 characters is already ~3.5 minutes of audio, and the limit is what one CPU-basic box can finish. |
| `runtime` | `string` | `""` | Empty uses the loaded runtime, loading `blue_yi` if none is loaded yet. A named runtime serves *this request only* and does not become the process default. |
| `voice` | `string` | `""` | Empty picks the runtime's default voice. Validated against `GET /v1/voices`; an unknown name is a 400 listing the valid ones. |
| `response_format` | `string` | `"wav"` | Only `wav` is supported; anything else is `400 invalid_request`. |
| `input_is_phonemes` | `bool` | `false` | Treat `input` as IPA — see [Input modes](#input-modes). |
| `input_is_nikud` | `bool` | `false` | Declare `input` as pointed Yiddish: the pointing is read by the G2P and the v5 model is not run. Ignored when `input_is_phonemes` is set. |
| `speed` | `float` | `1.0` | `0.5`–`2.0`. Above `1.0` is faster. |
| `stream` | `bool` | `false` | Framed stream instead of one WAV body. |
| `n_steps` | `int \| null` | `null` → 8 | Flow-matching steps, `1`–`32`. |
| `cfg_scale` | `float \| null` | `null` → 4.0 | Guidance strength, `1.0`–`8.0`; `1.0` disables guidance and halves the work. |
| `seed` | `int \| null` | `null` | Sampler noise seed, for reproducible output. Omit for a fresh draw. |

Non-streaming response: `200`, `Content-Type: audio/wav`, body = a complete mono 16-bit PCM WAV
with a 44-byte header, **at the serving runtime's sample rate** — 44100 Hz on `blue_yi`.

Three response headers describe what actually happened, on both the single-body and the
streaming response:

| Header | Example | Notes |
| --- | --- | --- |
| `X-Runtime` | `blue_yi` | The runtime that served the request. |
| `X-Sample-Rate` | `44100` | Same rate as the WAV header — handy before parsing the body. |
| `X-Dropped-Units` | (empty) | Inventory **phones** outside the closed set, or that the voice could not render even after folding: space-separated and **percent-encoded UTF-8** (header values are latin-1 and `ˈ`/`ʧ`/`ʤ` are not). Decode with one `urllib.parse.unquote`. Empty is the normal case: blue-yi can say every phone the engine emits. The field to read for "this voice lacks that phone" is `runtime_vocab_missing` from `GET /v1/phonemes/inventory`. Punctuation is never reported: it is structure, not a phone. |

```bash
curl -sD - -o out.wav -X POST http://localhost:7860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "מענטש", "voice": "female"}' | grep -i '^x-'
```

```
x-runtime: blue_yi
x-sample-rate: 44100
x-dropped-units:
```

Default runtime, named voice:

```bash
curl -s -X POST http://localhost:7860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "מיט א פאר יאר צוריק", "voice": "Berl"}' \
  -o out.wav
```

Cheaper and reproducible (fewer steps, guidance off, fixed seed):

```bash
curl -s -X POST http://localhost:7860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "וואס האט ער געזאגט", "voice": "female",
       "n_steps": 4, "cfg_scale": 1.0, "seed": 1234}' \
  -o out.wav
```

Hand-pointed input, so your pointing decides the reading (`מֶלֶךְ` → `mˈɛləx`, where the
unpointed `מלך` reads `mˈajləx`):

```bash
curl -s -X POST http://localhost:7860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "דער מֶלֶךְ האט געזאגט", "input_is_nikud": true}' \
  -o out.wav
```

Hand-written IPA, no engine involved:

```bash
curl -s -X POST http://localhost:7860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "mit a pˈur jur ʦirˈik", "input_is_phonemes": true, "speed": 1.1}' \
  -o out.wav
```

Streaming (see the next section for how to decode it):

```bash
curl -sN -X POST http://localhost:7860/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input": "וואס האט ער געזאגט? מיט א פאר יאר צוריק איז דאס געווען אנדערש.", "stream": true}' \
  -o stream.bin
```

Errors: `400 invalid_request` (empty `input`, `input` over 4000 characters, input that
phonemizes to nothing, `response_format != "wav"`, out-of-range `speed` / `n_steps` /
`cfg_scale` / `seed`, unknown `runtime`, unknown `voice`, off-inventory IPA in phoneme mode, one
unsplittable run of words too long to render), `503 not_available` (unimplemented runtime /
missing files, or the concurrency limit), `503 no_model` (nothing loadable), `500 internal_error`
(pipeline failure). Once streaming has begun, errors arrive as a kind-3 frame instead.

A per-request `runtime` does **not** change what the process has loaded: it resolves through a
per-id instance cache, so `GET /v1/voices`, `GET /v1/models/state` and the default sample rate
other callers see are untouched. `POST /v1/models/load` is the only way to change process state.

---

### UI endpoints

Both are `include_in_schema=False` — they serve the demo page, not the API — but `POST /generate`
is documented here because `static/script.js` talks to it.

#### `GET /`

Renders `templates/index.html`. No parameters. The page reads `/v1/models/state` and
`/v1/voices` on load to populate its voice picker, so it can never offer a voice the server
does not have. There is no runtime picker: this build ships one runtime.

#### `POST /generate`

`multipart/form-data` (or `application/x-www-form-urlencoded`) in, JSON out. One call returns
the pointing, the IPA, the token table and the audio, from the same pipeline `/v1/audio/speech`
uses — the two cannot produce different phonemes for the same input.

| Form field | Type | Default | Notes |
| --- | --- | --- | --- |
| `mode` | `string` | `"text"` | `text` \| `nikud` \| `phonemes`. The legacy value `diacritics` is accepted as an alias for `nikud`. |
| `text` | `string` | `""` | Used by `text` and `nikud` modes. In `nikud` mode the pointing is read by the G2P. Capped at the same 4000 characters as `SpeechBody.input`. |
| `phonemes` | `string` | `""` | Used by `phonemes` mode. Same 4000-character cap. |
| `voice` | `string` | `""` | Voice name; empty uses the runtime default. An unknown name is a 400. |
| `speed` | `float` | `1.0` | Same bounds as `SpeechBody.speed`: `0.5`–`2.0`. |
| `n_steps` | `int` | unset → 8 | `1`–`32`. blue-yi only. |
| `cfg_scale` | `float` | unset → 4.0 | `1.0`–`8.0`. blue-yi only. |
| `seed` | `int` | unset | Sampler seed, `0`–`2147483647`. blue-yi only. |

There is deliberately **no** `runtime` form field: `/generate` always uses the resident runtime
(loading the default if none is). The demo page switches runtime with `POST /v1/models/load`
first, so a switch is an explicit, reportable act rather than a side effect of a form post.

Response:

| Field | Type | Notes |
| --- | --- | --- |
| `nikud` | `string \| null` | `null` in `phonemes` mode. |
| `diacritics` | `string \| null` | Legacy alias of `nikud`, same value. |
| `phonemes` | `string` | The IPA sent to the voice's front end (before folding). |
| `audio` | `string` | `data:audio/wav;base64,…` — playable straight from an `<audio src>`. |
| `tokens[]` | `TokenRowDTO` | Same shape as `/v1/audio/phonemize`. `[]` in `phonemes` mode, where there is no G2P trace to show. A trace *failure* is not degraded to `[]`: the token table comes from the same engine pass as `phonemes`, so if it raises the request fails with `500 internal_error` and no audio — an engine that cannot build a table cannot be trusted to have produced the transcription either. |
| `unsupported[]` | `string` | Units outside the inventory, unioned with units the voice could not render. Empty on `blue_yi` for engine-produced IPA. |
| `runtime` | `string` | The runtime that served this request. |
| `voice` | `string` | The voice name used (`""` = the runtime's default). |
| `sample_rate` | `int` | Sample rate of the WAV in `audio`. |

```bash
curl -s -X POST http://localhost:7860/generate \
  -F 'mode=text' -F 'text=מיט א פאר יאר צוריק' -F 'voice=female' \
  | jq 'del(.audio, .tokens)'
```

```json
{
  "nikud": "מִיט אַ פּאָר יאָר צוּרִיק",
  "diacritics": "מִיט אַ פּאָר יאָר צוּרִיק",
  "phonemes": "mit a pˈur jur ʦirˈik",
  "unsupported": [],
  "runtime": "blue_yi",
  "voice": "female",
  "sample_rate": 44100
}
```

(`audio` and `tokens` were dropped for readability; `tokens` holds the same rows as
`/v1/audio/phonemize` above.)

Errors use the same `ErrorBody` envelope and codes as `/v1`, and every form field is
range-checked before any engine call — the input length included — so a caller mistake is a 400
with a message rather than a 500 with a traceback. Long text is chunked and joined here exactly
as it is on `/v1/audio/speech`.

---

## Streaming frame format

`POST /v1/audio/speech` with `stream: true` responds `200` with
`Content-Type: application/octet-stream` and a body that is a concatenation of frames.
Byte-identical to MamboTTS `mambotts-server/src/server/handlers/speech.rs`.

```
┌────────────┬───────────────────────────┬─────────────────┐
│ kind : u8  │ length : u32 big-endian   │ payload         │
│ 1 byte     │ 4 bytes                   │ `length` bytes  │
└────────────┴───────────────────────────┴─────────────────┘
                5-byte header
```

- The length is **big-endian** (network byte order), not little-endian, and counts the payload
  only — never the 5 header bytes.
- Frames are back to back with no padding, no delimiter and no trailer.
- Nothing about the total is announced up front: read until EOF or until a kind-3 frame.
- Every WAV payload carries the runtime's own sample rate in its header, so a decoder should
  read the rate from the chunk rather than assume 44100.

| `kind` | Name | Payload | Meaning |
| --- | --- | --- | --- |
| `1` | chunk | a complete WAV file | One text chunk, self-contained: its own 44-byte header, playable the moment it arrives. Chunks arrive in order. |
| `2` | final | a complete WAV file | The whole utterance: every chunk concatenated with a 60 ms silence gap per seam (`audio.CHUNK_GAP_SECONDS`), sent once after the last chunk, and byte-identical to what the non-streaming path returns for the same request. For download/save. Do **not** play it after the chunks — you would hear the audio twice. Its PCM is therefore slightly longer than the chunk frames added up. |
| `3` | error | UTF-8 text | A failure after the response started. Terminal: no further frames follow. There is no JSON envelope here, just the message. |

Chunking is done by `audio.chunk_text()` (≤ `audio.MAX_CHUNK_CHARS`, 200 characters): it splits
at whitespace only, preferring sentence-final punctuation, then clause commas, then any
whitespace. It never cuts inside a word, never at a maqaf `־`, and never inside a
whitespace-spelled multiword lexicon entry, so a streamed utterance and a non-streamed one say
the same thing.

Chunking is a **requirement, not an optimisation**, for blue-yi, and it therefore happens on
**both** paths, not only this one: the acoustic model renders exactly one utterance per call and
refuses text longer than 240 encoder tokens (the point past which its duration predictor stops
lengthening the utterance and starts cramming the same words into less time). The non-streaming
path chunks the identical way and joins the pieces with a 60 ms gap; `stream` is a delivery
choice, never a correctness one. The chunk budget does **not** scale with `speed`: the cap is on
the length of the text, so `speed=0.5` renders the same 200 characters over twice as long, which
is a legitimate request the model handles.

Residual case: a single unsplittable run of more than ~237 characters without whitespace still
exceeds the cap and comes back as `400 invalid_request` (or a kind-3 frame on a stream), naming
the token count.

An empty text yields no chunk frames — never a zero-sample kind-1 frame, which can interrupt a
client's playback queue.

### Python client

Reads frames off the response and plays each chunk as it arrives.

```python
"""pip install requests sounddevice soundfile"""
import struct
from io import BytesIO

import requests
import sounddevice as sd
import soundfile as sf

BASE = "http://localhost:7860"
KIND_CHUNK, KIND_FINAL, KIND_ERROR = 1, 2, 3


def read_exactly(stream, n: int) -> bytes:
    """A socket read returns *at most* n bytes; frames need exactly n."""
    buf = bytearray()
    while len(buf) < n:
        part = stream.read(n - len(buf))
        if not part:
            return bytes(buf)  # EOF: a short read here means a truncated frame
        buf += part
    return bytes(buf)


def frames(stream):
    """Yield (kind, payload) until EOF."""
    while True:
        header = read_exactly(stream, 5)
        if len(header) < 5:
            return  # clean EOF, or a truncated header
        kind = header[0]
        (length,) = struct.unpack(">I", header[1:5])  # ">I" = big-endian u32
        payload = read_exactly(stream, length)
        if len(payload) < length:
            raise IOError(f"truncated frame: kind {kind}, {len(payload)}/{length} bytes")
        yield kind, payload


def speak(text: str, *, voice: str = "", save_to: str | None = None) -> None:
    body = {"input": text, "stream": True}
    if voice:
        body["voice"] = voice
    # stream=True defers reading the body; decode_content unwraps any gzip so the
    # frame lengths refer to the bytes we actually see.
    with requests.post(f"{BASE}/v1/audio/speech", json=body, stream=True) as resp:
        if resp.status_code != 200:            # pre-stream failure: JSON envelope
            raise RuntimeError(resp.json()["error"]["message"])
        resp.raw.decode_content = True
        for kind, payload in frames(resp.raw):
            if kind == KIND_CHUNK:
                # The rate comes from the WAV header (44100 on blue_yi), not a constant.
                samples, rate = sf.read(BytesIO(payload), dtype="float32")
                sd.play(samples, rate)
                sd.wait()                       # keeps chunks from overlapping
            elif kind == KIND_FINAL:
                if save_to:                     # already played as chunks
                    with open(save_to, "wb") as fh:
                        fh.write(payload)
            elif kind == KIND_ERROR:
                raise RuntimeError(payload.decode("utf-8", "replace"))


if __name__ == "__main__":
    speak(
        "וואס האט ער געזאגט? מיט א פאר יאר צוריק איז דאס געווען אנדערש.",
        voice="Berl",
        save_to="out.wav",
    )
```

---

## Phone inventory

Closed set, from the G2P spec §1 (`Phonikud-yi/docs/yiddish_phoneme_set.md`). **Nothing outside
it is ever emitted**; `GET /v1/phonemes/inventory` serves it, and `POST /v1/audio/phonemize`
reports any leak in `unsupported`. Every IPA string in the tables below is engine output for the
Yiddish word beside it.

### Vowels

| Unit | Description | Example | IPA |
| --- | --- | --- | --- |
| `a` | open central, short | מאכן | `maxn` |
| `aː` | same quality, long (class 34, flattened *ay*) | היינט | `haːnt` |
| `ɛ` | open-mid front (stressed ע) | קען | `kɛn` |
| `ə` | schwa — every unstressed ɛ | אבער | `ˈɔbər` |
| `i` | close front (also the native ו-vowel) | גוט | `ɡit` |
| `u` | close back (class 12/13 א, LK kometz) | וואס | `vus` |
| `ɔ` | open-mid back (class 41 א) | דארט | `dɔrt` |
| `ej` | class-25 lengthened e | וועג | `vejɡ` |
| `aj` | default יי | צוויי | `ʦvaj` |
| `ɔj` | default וי | שוין | `ʃɔjn` |
| `oʊ` | class-54 וי (lexical list) | הויז | `hoʊz` |

### Consonants

| Unit | Note | Example | IPA |
| --- | --- | --- | --- |
| `b d f h j k l m n p r s t v z` | as written | ברודער | `brˈidər` |
| `ɡ` | **U+0261 script g**, not ASCII `g` | גוט | `ɡit` |
| `x` | ch (כ/ח) | מאכן | `maxn` |
| `ʃ` | sh | שוין | `ʃɔjn` |
| `ʒ` | zh | זשארגאן | `ʒˈarɡan` |
| `ʦ` | **U+02A6**, ts (צ) | צוויי | `ʦvaj` |
| `ʧ` | **U+02A7**, tsh (טש) | טשאלנט, מענטש | `ʧalnt`, `mɛnʧ` |
| `ʤ` | **U+02A4**, dzh (דזש) | דזשאב, באדזשעט | `ʤab`, `baʤˈɛt` |
| `ŋ` | velar nasal | — | declared in the set, but the engine does not emit it: nasal place assimilation is deliberately not modelled, so זינגען is `zˈinɡən`, not `zˈiŋən` |

### Marks

| Unit | Note |
| --- | --- |
| `ˈ` | primary stress, placed **immediately before the stressed vowel**; at most one per word; never on monosyllables |

`ː` is deliberately **not** an inventory member: the spec allows it only in `aː`, so `aː` is
carried as a single unit and a `ː` after any other vowel is reported as an off-inventory unit
(`ɔːbər` → `unsupported: ["ː"]`) instead of being quietly accepted.

The affricates are **single codepoints**, not the sequences `ts` / `tʃ` / `dʒ`. Copy them, never
retype them: `"ɡ" == "g"` is `False`, and a mixed-up script-g turns every /ɡ/ into a different
phone. Punctuation is not a phone and not a violation: the characters the engine splices around
a token (`. , ! ? ; : ( ) [ ] … -` and friends) pass validation untouched.

### What blue-yi covers, and what folding would do

blue-yi's `vocab.json` maps every unit of the inventory, `ʦ` (155), `ʧ` (184), `ʤ` (182),
`ɡ` (66), `ŋ` (44), `ˈ` (120) and `ː` (122) included — the affricates are single ids there, not
`t`+`s` pairs — so `runtime_vocab_missing` is empty and `fold_to_vocab` is a no-op. Two traps
worth knowing if you write IPA by hand: ASCII `'` (U+0027) is also in that vocab as id 5, so it
does not raise, it simply means *apostrophe* instead of *stress*; and ASCII `g` (154) is a
different embedding from `ɡ` U+0261 (66).

Folding is therefore machinery no shipped runtime exercises. It is documented because it runs
on every request and because `runtime_vocab_missing` and `unsupported` are defined in terms of
it — a runtime with a narrower vocabulary would put it to work.

Before synthesis, `phones.fold_to_vocab(ipa, runtime.vocab())` rewrites every unit the vocab
lacks. A candidate is usable when **every character in it** is in the vocab — the vocab is keyed
by single characters, so `tʃ` is not an entry, it is the entries `t` and `ʃ` used back to back.
Candidates are tried in order and the first usable one wins:

| Unit | Candidates | Result on `blue_yi` |
| --- | --- | --- |
| `ʧ` | `ʧ`, `tʃ` | untouched |
| `ʤ` | `ʤ`, `dʒ` | untouched |
| `ʦ` | `ʦ`, `ts` | untouched |
| `ɡ` | `ɡ`, `g` | untouched |
| `aː` | `aː`, `a` | untouched |
| `oʊ` | `oʊ`, `o` | untouched |
| `ə` | `ə`, `e` | untouched |
| `ŋ` | `ŋ`, `n` | untouched |
| `x` | `x`, `χ`, `k` | untouched |

A fold would spell an affricate out as stop + fricative — a real change to the output, but the
honest degradation, since without it a word with a tsh or dzh would simply **lose a consonant**.
On this runtime the question does not arise.

A unit with no usable candidate is dropped from the string and **always reported**: it appears
in `unsupported` on `/generate` and `/v1/audio/phonemize`, and the runtime returns it to the
caller alongside the audio rather than stashing it on itself. Silence is never an acceptable
answer to a phone the voice cannot say.

---

## Verifying a deployment

```bash
python3 scripts/selftest_space.py     # must print ALL CHECKS PASSED
```

It checks the registry, the inventory and folding, WAV packing and the frame header, the
engine's canary readings (including all seven lexicon tables non-empty), an end-to-end
synthesis, and that every route documented above is actually exposed. Set
`PHONIKUD_YI_ENGINE_DIR` to an unpacked engine bundle to run it without the 1.23 GB download.
