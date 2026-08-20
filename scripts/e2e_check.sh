#!/usr/bin/env bash
# End-to-end smoke test against a running Space. Every request below was run by
# hand during integration; this file is that sequence, verbatim and repeatable.
#
#   scripts/e2e_check.sh [PORT]
#
# It starts its own server on PORT (default 7863) unless one is already
# answering there, exercises every documented endpoint on the one runtime this
# build ships (blue-yi at 44.1 kHz), decodes the streaming frame headers from
# the raw bytes, and verifies each WAV header with
# the standard library's `wave` module — no dependency beyond requirements.txt,
# which deliberately does not ship soundfile. Exits non-zero on the first
# failure.
#
# PHONIKUD_YI_ENGINE_DIR points at an unpacked engine bundle and skips the
# ~1.23 GB snapshot download; BLUE25_MODEL_DIR does the same for the ~282 MB
# blue-yi bundle. Both are auto-detected below when they are already on
# this machine. Without them the first request pays for the fetch.
set -euo pipefail

PORT="${1:-7863}"
BASE="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
OUT="${OUT_DIR:-/tmp}"
SAMPLE='מיט א פאר יאר צוריק'
PARAGRAPH='דער בעל־הבית האט געזאגט אז מען וועט מאכן א גרויסע שמחה. די קינדער שפילן זיך אין דרויסן, און די מאמע רופט זיי אריין. וואס האט ער געזאגט? מיט א פאר יאר צוריק איז דאס געווען אנדערש; היינט איז אלעס אנדערש. א דאנק פאר די גוטע נייעס, איך האב געהערט אז ער וועט קומען מארגן.'
# Two of blue-yi's four fixed voices, one male one female: the pair whose F0
# separation the selftest asserts, so different audio here is meaningful.
VOICE_A='Berl'
VOICE_B='Rukhl'

[ -x "$PY" ] || { echo "no venv at ${ROOT}/.venv — python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 2; }

# Local bundles, so a repeat run never touches the network.
LOCAL_ENGINE="$(cd "${ROOT}/.." 2>/dev/null && pwd)/Phonikud-yi/dist/phonikud-yi-engine"
LOCAL_BLUE="${HOME}/.cache/huggingface/hub/models--notmax123--blue-yi/snapshots/34ee8856b85043b68cfbcaf0b3acad4c20326f88"
if [ -z "${PHONIKUD_YI_ENGINE_DIR:-}" ] && [ -f "${LOCAL_ENGINE}/yiddish_labels.py" ]; then
  export PHONIKUD_YI_ENGINE_DIR="$LOCAL_ENGINE"
fi
if [ -z "${BLUE25_MODEL_DIR:-}" ] && [ -f "${LOCAL_BLUE}/tts.json" ]; then
  export BLUE25_MODEL_DIR="$LOCAL_BLUE"
fi
echo "engine: ${PHONIKUD_YI_ENGINE_DIR:-<download>}"
echo "blue  : ${BLUE25_MODEL_DIR:-<download>}"

started=""
if ! curl -sf "${BASE}/health" >/dev/null 2>&1; then
  echo "== starting server on ${PORT} =="
  (cd "$ROOT" && "$PY" app.py --port "$PORT" > "${OUT}/yi_server.log" 2>&1 &)
  started=1
  for _ in $(seq 1 120); do sleep 1; curl -sf "${BASE}/health" >/dev/null 2>&1 && break; done
  curl -sf "${BASE}/health" >/dev/null || { echo "server never came up; see ${OUT}/yi_server.log"; exit 1; }
fi
cleanup() { [ -n "$started" ] && pkill -f "app.py --port ${PORT}" || true; }
trap cleanup EXIT

# --- helpers ---------------------------------------------------------------
# Status code is checked explicitly everywhere: a 500 that returns valid JSON
# still has to fail this script.
jpost() { # jpost PATH JSON EXPECTED_STATUS
  local code
  code=$(curl -s -o "${OUT}/body.json" -w '%{http_code}' -X POST "${BASE}$1" \
            -H 'content-type: application/json' --data-binary "$2")
  head -c 800 "${OUT}/body.json"; echo
  [ "$code" = "$3" ] || { echo "!! $1 expected HTTP $3, got $code"; exit 1; }
}
jget() { # jget PATH EXPECTED_STATUS
  local code
  code=$(curl -s -o "${OUT}/body.json" -w '%{http_code}' "${BASE}$1")
  head -c 800 "${OUT}/body.json"; echo
  [ "$code" = "$2" ] || { echo "!! $1 expected HTTP $2, got $code"; exit 1; }
}
jassert() { # jassert PYTHON_EXPR_OVER_d  DESCRIPTION   (d = parsed body.json)
  "$PY" - "$1" "$2" "${OUT}/body.json" <<'PY'
import json, sys
expr, why, path = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(path, encoding="utf-8"))
assert eval(expr, {"d": d}), f"!! {why}: {json.dumps(d, ensure_ascii=False)[:400]}"
print(f"ok  {why}")
PY
}
# json_body TEXT [EXTRA_JSON_OBJECT] -> a request body on stdout. Built with
# json.dumps rather than string concatenation so the Yiddish text is escaped
# correctly whatever the shell does to it.
json_body() {
  "$PY" - "$1" "${2-}" <<'PY'
import json, sys
body = {"input": sys.argv[1]}
if len(sys.argv) > 2 and sys.argv[2].strip():
    body.update(json.loads(sys.argv[2]))
print(json.dumps(body, ensure_ascii=False))
PY
}
wav_report() { # wav_report FILE EXPECTED_RATE
  "$PY" - "$1" "$2" <<'PY'
import sys, wave
import numpy as np
path, want = sys.argv[1], int(sys.argv[2])
with wave.open(path, "rb") as fh:
    assert fh.getsampwidth() == 2, "!! not 16-bit PCM"
    channels, rate, frames = fh.getnchannels(), fh.getframerate(), fh.getnframes()
    pcm = fh.readframes(frames)
data = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
if channels > 1:
    data = data.reshape(-1, channels)
dur = len(data) / rate
peak = abs(data).max()
print(f"ok  {path}: {len(data)} frames @ {rate} Hz = {dur:.2f} s, peak {peak:.3f}")
assert data.ndim == 1, "not mono"
assert rate == want, f"!! expected {want} Hz in the WAV header, got {rate}"
assert 0.3 < dur < 60, f"!! implausible duration {dur:.2f} s"
assert peak > 0.01, "!! silence"
PY
}

echo "== GET /health =="
jget /health 200

echo "== GET /v1/models/sources (blue_yi first, default, available) =="
jget /v1/models/sources 200
jassert 'd["runtimes"][0]["id"] == "blue_yi" and d["runtimes"][0]["available"]' 'blue_yi heads the catalog and is available'
jassert 'all(r["available"] for r in d["runtimes"])' 'every catalog runtime is available in this build'
jassert '[r for r in d["runtimes"] if r["id"]=="blue_yi"][0]["capabilities"]["fixed_voices"] is True' 'blue_yi advertises fixed_voices'

echo "== POST /v1/models/load nope (400 invalid_request) =="
jpost /v1/models/load '{"runtime":"nope"}' 400
jassert 'd["error"]["code"] == "invalid_request"' 'unknown runtime is an invalid_request, not a crash'

echo "== POST /v1/models/load blue_yi (the default runtime; builds 4 ONNX sessions) =="
jpost /v1/models/load '{"runtime":"blue_yi"}' 200
jassert 'd["runtime"] == "blue_yi"' 'blue_yi loaded'

echo "== GET /v1/models/state (must report blue at 44100) =="
jget /v1/models/state 200
jassert 'd["loaded"] is True and d["runtime"] == "blue_yi" and d["sample_rate"] == 44100' 'state reports blue_yi @ 44100 Hz'

echo "== GET /v1/voices (blue-yi has four fixed voices) =="
jget /v1/voices 200
jassert 'd["runtime"] == "blue_yi" and len(d["voices"]) == 4' 'four voices'
jassert 'sorted(d["voices"]) == ["Berl","Hershl","Rukhl","Sheyndl"]' 'the four voices are the bundled ones'

echo "== GET /v1/languages =="
jget /v1/languages 200

echo "== GET /v1/phonemes/inventory (blue folds nothing: runtime_vocab_missing must be empty) =="
jget /v1/phonemes/inventory 200
jassert 'd["runtime_vocab_missing"] == []' 'blue-yi vocab covers the whole Yiddish inventory'

echo "== POST /v1/audio/diacritize =="
json_body "$SAMPLE" > "${OUT}/req.json"
jpost /v1/audio/diacritize "$(cat "${OUT}/req.json")" 200

echo "== POST /v1/audio/phonemize =="
jpost /v1/audio/phonemize "$(cat "${OUT}/req.json")" 200
jassert 'd["phonemes"] == "mit a pˈur jur ʦirˈik"' 'the canary phonemization is unchanged'
jassert 'd["unsupported"] == []' 'nothing unsupported under blue-yi'
echo "== POST /v1/audio/phonemize (blank input -> 400) =="
jpost /v1/audio/phonemize '{"input":"  "}' 400
jassert 'd["error"]["code"] == "invalid_request"' 'blank input is an ErrorBody, not FastAPI detail'

echo "== POST /v1/audio/speech with ${VOICE_A} (non-streaming WAV @ 44.1 kHz) =="
json_body "$SAMPLE" "{\"voice\":\"${VOICE_A}\"}" > "${OUT}/req_a.json"
code=$(curl -s -o "${OUT}/yi_blue_a.wav" -w '%{http_code}' -D "${OUT}/speech.headers" \
         -X POST "${BASE}/v1/audio/speech" -H 'content-type: application/json' \
         --data-binary "@${OUT}/req_a.json")
[ "$code" = "200" ] || { echo "!! speech returned $code"; head -c 400 "${OUT}/yi_blue_a.wav"; exit 1; }
grep -i '^content-disposition' "${OUT}/speech.headers"
wav_report "${OUT}/yi_blue_a.wav" 44100

echo "== POST /v1/audio/speech with ${VOICE_B} (must be a different voice, same text) =="
json_body "$SAMPLE" "{\"voice\":\"${VOICE_B}\"}" > "${OUT}/req_b.json"
code=$(curl -s -o "${OUT}/yi_blue_b.wav" -w '%{http_code}' \
         -X POST "${BASE}/v1/audio/speech" -H 'content-type: application/json' \
         --data-binary "@${OUT}/req_b.json")
[ "$code" = "200" ] || { echo "!! speech returned $code"; head -c 400 "${OUT}/yi_blue_b.wav"; exit 1; }
wav_report "${OUT}/yi_blue_b.wav" 44100
# Same text, different voice: the bytes must differ, and the female
# voice must actually be higher pitched. A `voice` field that is read but never
# reaches the style tensors would pass a byte-difference test on noise alone,
# so pitch is checked too — the same invariant the selftest guards.
"$PY" - "${OUT}/yi_blue_a.wav" "${OUT}/yi_blue_b.wav" <<'PY'
import sys
import wave

import numpy as np


def read_wav(path):
    """(mono float64 in [-1, 1], sample rate) from a 16-bit PCM WAV."""
    with wave.open(path, "rb") as fh:
        assert fh.getsampwidth() == 2, "!! not 16-bit PCM"
        rate, channels = fh.getframerate(), fh.getnchannels()
        pcm = fh.readframes(fh.getnframes())
    data = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def median_f0(path, fmin=60.0, fmax=400.0):
    x, sr = read_wav(path)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    win, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = int(sr / fmax), int(sr / fmin)
    f0s = []
    for i in range(0, x.size - win, hop):
        f = x[i:i + win]
        if np.sqrt(np.mean(f * f)) < 0.02:
            continue
        f = f - f.mean()
        ac = np.correlate(f, f, "full")[win - 1:]
        ac /= ac[0] + 1e-12
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] > 0.3:
            f0s.append(sr / lag)
    assert f0s, f"!! no voiced frames in {path}"
    return float(np.median(f0s))

a_path, b_path = sys.argv[1], sys.argv[2]
a, _ = read_wav(a_path)
b, _ = read_wav(b_path)
assert a.shape != b.shape or not np.array_equal(a, b), "!! two voices produced identical audio"
fa, fb = median_f0(a_path), median_f0(b_path)
print(f"ok  {a_path} F0 {fa:.1f} Hz (male) vs {b_path} F0 {fb:.1f} Hz (female)")
assert fb > fa * 1.2, f"!! the female voice is not higher pitched ({fb:.1f} vs {fa:.1f} Hz)"
PY

echo "== POST /v1/audio/speech (response_format mp3 -> 400) =="
jpost /v1/audio/speech '{"input":"מיט א פאר יאר צוריק","response_format":"mp3"}' 400

echo "== POST /v1/audio/speech (unknown voice -> 400, never a silent fallback) =="
jpost /v1/audio/speech '{"input":"מיט א פאר יאר צוריק","voice":"no_such_voice"}' 400

echo "== POST /v1/audio/speech (IPA input bypasses G2P) =="
code=$(curl -s -o "${OUT}/yi_phonemes.wav" -w '%{http_code}' -X POST "${BASE}/v1/audio/speech" \
         -H 'content-type: application/json' \
         -d '{"input":"mit a pˈur jur ʦirˈik","input_is_phonemes":true}')
[ "$code" = "200" ] || { echo "!! phoneme input returned $code"; exit 1; }
wav_report "${OUT}/yi_phonemes.wav" 44100

echo "== POST /v1/audio/speech stream=true (frame headers decoded from raw bytes) =="
json_body "$PARAGRAPH" '{"stream":true,"voice":"'"${VOICE_A}"'"}' > "${OUT}/req_stream.json"
curl -s -o "${OUT}/yi_stream.bin" -D "${OUT}/stream.headers" -X POST "${BASE}/v1/audio/speech" \
  -H 'content-type: application/json' --data-binary "@${OUT}/req_stream.json"
grep -i '^content-type' "${OUT}/stream.headers"
# audio.CHUNK_GAP_SECONDS: the non-streaming path and the kind-2 final frame
# join chunks with this much silence per seam (RECIPE G13), so the final WAV is
# legitimately longer than the chunk frames added up.
"$PY" - "${OUT}/yi_stream.bin" 44100 0.06 <<'PY'
import struct, sys
raw = open(sys.argv[1], "rb").read()
want_rate = int(sys.argv[2])
gap_bytes = round(float(sys.argv[3]) * want_rate) * 2  # 16-bit mono
off, chunk_pcm, final_pcm, kinds = 0, 0, None, []
while off < len(raw):
    kind = raw[off]                                  # [kind:u8]
    n = struct.unpack(">I", raw[off + 1:off + 5])[0] # [len:u32 big-endian]
    body = raw[off + 5:off + 5 + n]
    assert len(body) == n, "!! truncated frame"
    assert kind in (1, 2), f"!! unexpected frame kind {kind}: {body[:200]!r}"
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE", "!! chunk is not a WAV"
    channels = struct.unpack("<H", body[22:24])[0]
    rate = struct.unpack("<I", body[24:28])[0]
    print(f"kind={kind} len={n} channels={channels} rate={rate}")
    assert rate == want_rate, f"!! frame WAV header says {rate} Hz, expected {want_rate}"
    assert channels == 1, "!! frame WAV is not mono"
    kinds.append(kind)
    if kind == 1:
        chunk_pcm += n - 44
    elif kind == 2:
        final_pcm = n - 44
    off += 5 + n
assert off == len(raw), "!! trailing bytes"
assert final_pcm is not None, "!! no kind 2 final frame"
assert kinds.count(1) >= 2, f"!! expected several chunk frames, got {kinds}"
assert kinds[-1] == 2, "!! the final frame is not kind 2"
seams = kinds.count(1) - 1
expected = chunk_pcm + seams * gap_bytes
assert expected == final_pcm, (
    f"!! final WAV {final_pcm}B != chunks {chunk_pcm}B + {seams} x {gap_bytes}B of gap"
)
print(f"ok  {kinds.count(1)} chunk frames, {chunk_pcm} PCM bytes + {seams} seam gap(s) "
      f"== the {final_pcm}B kind 2 concatenated WAV @ {want_rate} Hz")
PY

echo "== POST /v1/audio/speech stream=true with off-inventory IPA (pre-flight 400 or in-band kind 3) =="
code=$(curl -s -o "${OUT}/yi_err.bin" -w '%{http_code}' -X POST "${BASE}/v1/audio/speech" \
  -H 'content-type: application/json' \
  -d '{"input":"θejl","input_is_phonemes":true,"stream":true}')
# Two legal shapes, and which one you get depends only on WHEN the bad phone is
# noticed. Validation before the response starts can still use the ErrorBody
# envelope and a real status code; once bytes are on the wire the status line is
# already sent, so the only way to report a failure is the in-band kind 3 frame.
# Both are accepted here; a 200 that just stops, or a bare FastAPI "detail", is
# not.
"$PY" - "${OUT}/yi_err.bin" "$code" <<'PY'
import json, struct, sys
raw = open(sys.argv[1], "rb").read()
code = sys.argv[2]
if code == "400":
    body = json.loads(raw.decode("utf-8"))
    assert "detail" not in body, f"!! FastAPI detail leaked: {body}"
    assert body["error"]["code"] == "invalid_request", f"!! wrong error code: {body}"
    assert "inventory" in body["error"]["message"], f"!! unhelpful message: {body}"
    print(f"ok  pre-flight 400 ErrorBody: {body['error']['message']}")
else:
    assert code == "200", f"!! unexpected status {code}: {raw[:200]!r}"
    kind, n = raw[0], struct.unpack(">I", raw[1:5])[0]
    message = raw[5:5 + n].decode("utf-8")
    print(f"ok  in-band kind={kind} len={n} message={message}")
    assert kind == 3, f"!! expected a kind 3 error frame, got kind {kind}"
    assert "inventory" in message, f"!! unhelpful message: {message}"
PY

echo "== POST /v1/audio/speech, a whole PARAGRAPH, stream=false (the default path chunks too) =="
# The regression this guards: chunking used to happen only when stream=true, so
# an ordinary paragraph on the default path came back 500 internal_error with a
# duration message. Only the 19-character SAMPLE was ever sent down this path,
# which is why CI passed.
json_body "$PARAGRAPH" "{\"voice\":\"${VOICE_A}\"}" > "${OUT}/req_para.json"
code=$(curl -s -o "${OUT}/yi_para.wav" -w '%{http_code}' -X POST "${BASE}/v1/audio/speech" \
         -H 'content-type: application/json' --data-binary "@${OUT}/req_para.json")
[ "$code" = "200" ] || { echo "!! non-streaming paragraph returned $code"; head -c 400 "${OUT}/yi_para.wav"; exit 1; }
wav_report "${OUT}/yi_para.wav" 44100

echo "== the same paragraph at speed 0.5 (the documented lower bound) =="
# And this one: the chunk budget used to ignore `speed` while the guard sat on
# predicted duration, so speed 0.5 produced no audio at all — a lone kind-3
# frame under HTTP 200. The cap is on text length now, so slow speech renders.
json_body "$PARAGRAPH" '{"speed":0.5}' > "${OUT}/req_slow.json"
code=$(curl -s -o "${OUT}/yi_slow.wav" -w '%{http_code}' -X POST "${BASE}/v1/audio/speech" \
         -H 'content-type: application/json' --data-binary "@${OUT}/req_slow.json")
[ "$code" = "200" ] || { echo "!! speed 0.5 returned $code"; head -c 400 "${OUT}/yi_slow.wav"; exit 1; }
wav_report "${OUT}/yi_slow.wav" 44100

echo "== POST /generate with the whole paragraph (the UI's Generate button) =="
curl -s -X POST "${BASE}/generate" -F mode=text -F "text=${PARAGRAPH}" -F phonemes= \
  | "$PY" -c 'import json,sys;d=json.load(sys.stdin);assert not d.get("error"),d;assert d["audio"].startswith("data:audio/wav;base64,");print("ok  /generate spoke a paragraph:",len(d["audio"]),"b64 chars,",len(d["tokens"]),"token rows, unsupported",d["unsupported"])'

echo "== POST /v1/audio/speech with text that phonemizes to nothing (400, not 500) =="
jpost /v1/audio/speech '{"input":"12345"}' 400
jassert 'd["error"]["code"] == "invalid_request"' 'quarantined-to-empty input is a 400 invalid_request'

echo "== POST /v1/audio/speech over the input cap (400) =="
"$PY" -c 'import json;print(json.dumps({"input":"א"*4001}))' > "${OUT}/req_big.json"
jpost /v1/audio/speech "$(cat "${OUT}/req_big.json")" 400
jassert 'd["error"]["code"] == "invalid_request"' '4001 characters is refused before any graph runs'

echo "== POST /v1/audio/speech with one unsplittable over-long run (400, not 500) =="
"$PY" -c 'import json;print(json.dumps({"input":"א"*400}))' > "${OUT}/req_word.json"
jpost /v1/audio/speech "$(cat "${OUT}/req_word.json")" 400
jassert '"text tokens" in d["error"]["message"]' 'the too-long refusal names the token count'

echo "== punctuation the vocab cannot spell is NOT reported as a dropped phone =="
json_body 'צי, אויב [ניט] פארוואס? אויב… יא!' > "${OUT}/req_punct.json"
code=$(curl -s -o "${OUT}/yi_punct.wav" -w '%{http_code}' -D "${OUT}/punct.headers" \
         -X POST "${BASE}/v1/audio/speech" -H 'content-type: application/json' \
         --data-binary "@${OUT}/req_punct.json")
[ "$code" = "200" ] || { echo "!! bracketed text returned $code"; exit 1; }
"$PY" - "${OUT}/punct.headers" <<'DROPPED'
import sys
value = ""
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    if line.lower().startswith("x-dropped-units:"):
        value = line.split(":", 1)[1].strip()
assert value == "", f"!! punctuation reported as a dropped phone: {value!r}"
print("ok  X-Dropped-Units is empty on bracketed Yiddish")
DROPPED
wav_report "${OUT}/yi_punct.wav" 44100

echo "== a per-request runtime must NOT change what the process has loaded =="
# One runtime ships, so the swap this used to make (blue -> the fallback -> blue)
# has nowhere to go. What is still worth proving is the plumbing around it: a
# per-request `runtime` naming the resident runtime is served normally, one
# naming anything else is a 400 rather than a crash or a silent fallback, and
# neither leaves the process pointing somewhere new.
jget /v1/models/state 200
jassert 'd["runtime"] == "blue_yi"' 'blue_yi is resident before the pinned request'
json_body "$SAMPLE" '{"runtime":"blue_yi"}' > "${OUT}/req_pin.json"
code=$(curl -s -o "${OUT}/yi_pinned.wav" -w '%{http_code}' -X POST "${BASE}/v1/audio/speech" \
         -H 'content-type: application/json' --data-binary "@${OUT}/req_pin.json")
[ "$code" = "200" ] || { echo "!! pinned-runtime request returned $code"; exit 1; }
wav_report "${OUT}/yi_pinned.wav" 44100
json_body "$SAMPLE" '{"runtime":"no_such_runtime"}' > "${OUT}/req_gone.json"
jpost /v1/audio/speech "$(cat "${OUT}/req_gone.json")" 400
jassert 'd["error"]["code"] == "invalid_request"' 'a runtime this build does not have is a 400, not a fallback'
jget /v1/models/state 200
jassert 'd["runtime"] == "blue_yi" and d["sample_rate"] == 44100' 'the pinned requests left blue_yi resident'
jget /v1/voices 200
jassert 'len(d["voices"]) == 4' 'and left the four blue-yi voices on offer'
json_body "$SAMPLE" '{"voice":"Hershl"}' > "${OUT}/req_after.json"
after_code=$(curl -s -o "${OUT}/yi_after.wav" -w '%{http_code}' -X POST "${BASE}/v1/audio/speech" \
         -H 'content-type: application/json' --data-binary "@${OUT}/req_after.json")
[ "$after_code" = "200" ] || { echo "!! a blue-yi voice failed after the pinned requests: $after_code"; exit 1; }
wav_report "${OUT}/yi_after.wav" 44100

echo "== POST /v1/models/load blue_yi (an explicit load; state must follow) =="
jpost /v1/models/load '{"runtime":"blue_yi"}' 200
jassert 'd["runtime"] == "blue_yi"' 'the explicit load reports blue_yi'
jget /v1/models/state 200
jassert 'd["loaded"] is True and d["runtime"] == "blue_yi" and d["sample_rate"] == 44100' 'state follows the explicit load'

echo "== GET / (demo UI) =="
code=$(curl -s -o "${OUT}/index.html" -w '%{http_code}' "${BASE}/")
[ "$code" = "200" ] || { echo "!! / returned $code"; exit 1; }
grep -q "voice-select" "${OUT}/index.html" || { echo "!! the voice picker is missing from the page"; exit 1; }
echo "ok  $(wc -c < "${OUT}/index.html") bytes"

echo "== GET /docs + /openapi.json =="
curl -s "${BASE}/docs" | grep -qi swagger || { echo "!! /docs is not the OpenAPI UI"; exit 1; }
curl -s "${BASE}/openapi.json" | "$PY" -c 'import json,sys;p=sorted(json.load(sys.stdin)["paths"]);print(p);assert "/v1/audio/speech" in p'

echo "== GET /nope (404 must use the ErrorBody envelope, not FastAPI detail) =="
jget /nope 404
jassert '"error" in d and "detail" not in d' '404 uses ErrorBody{error:{code,message}}'

echo "== POST /generate (legacy form endpoint, mode=text) =="
curl -s -X POST "${BASE}/generate" -F mode=text -F "text=${SAMPLE}" -F phonemes= \
  | "$PY" -c 'import json,sys;d=json.load(sys.stdin);assert d["audio"].startswith("data:audio/wav;base64,");print(d["nikud"],"|",d["phonemes"],"|",len(d["tokens"]),"tokens")'

echo "== POST /generate (legacy mode=diacritics alias) =="
curl -s -X POST "${BASE}/generate" -F mode=diacritics -F 'text=מִיט אַ פּאָר יאָר צוּרִיק' \
  | "$PY" -c 'import json,sys;d=json.load(sys.stdin);assert d["nikud"]==d["diacritics"] and d["audio"];print("ok",d["phonemes"])'

echo
echo "ALL E2E CHECKS PASSED"
