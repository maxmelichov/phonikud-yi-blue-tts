"""BlueTTS 2.5 adapter: Yiddish IPA in, 44.1 kHz float32 out.

Pure numpy + onnxruntime port of the reference gradio app's ``BlueTTS`` class
(notmax123/BlueV3 ``app.py``, lines ~930-1400), adapted to the
``notmax123/BlueTTS2.5-onnx`` bundle. No torch, no bluecodec, no librosa.

Why this bundle is the right acoustic model for Yiddish, and where it is not:

* Its ``vocab.json`` is char-level and carries the whole Yiddish closed
  inventory natively, affricate ligatures included (``ʦ`` U+02A6 = 155,
  ``ʧ`` U+02A7 = 184, ``ʤ`` U+02A4 = 182, ``ɡ`` U+0261 = 66, ``ŋ`` = 44,
  ``ˈ`` = 120, ``ː`` = 122). Nothing has to be folded, unlike the Piper voice.
* ``stats.npz`` was exported from ``stats_yiddish.pt`` and ``yi`` is one of the
  checkpoint's declared languages, so the latent statistics are the Yiddish
  ones — but every offered voice is a Hebrew or English reader, so expect
  a foreign accent even though every phone is rendered.
* The autoencoder *encoder* was never exported, so ``duration_predictor.onnx``
  (wants ``z_ref``) and ``reference_encoder.onnx`` are unusable here and this
  runtime never opens them. Voice cloning from a wav is impossible from this
  bundle; the five saved styles are all there is.

Scope: this class synthesizes **exactly one utterance per call**, and it
refuses text longer than ``MAX_TEXT_TOKENS`` with ``UtteranceTooLongError``.
The refusal is on the *token count of the text*, checked before any graph runs,
and that is the only place it can be correct — see MAX_TEXT_TOKENS for the
measurements. Splitting long text is the caller's job: every synthesis path in
the API layer (streaming and non-streaming alike, plus the demo UI) goes through
``audio.chunk_text`` at ``audio.MAX_CHUNK_CHARS``, which is derived from this
cap. The division of labour is: caller chunks, this class renders one chunk.

Every implementation detail below (the 144->24 latent fold, denormalizing
before the fold, the mandatory edge trim, the external CFG branch) was measured
against real audio on this bundle; see BLUE25_RECIPE.md. Do not "simplify" the
maths — four of the plausible variants produce audio-shaped output that is not
speech.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import phones
from ..registry import BLUE_MODEL_DIR_ENV, BLUE_REPO_ID, BLUE_REVISION

log = logging.getLogger(__name__)

__all__ = [
    "BLUE_MODEL_DIR_ENV",
    "MAX_TEXT_TOKENS",
    "BlueYiddish",
    "DEFAULT_VOICE",
    "UtteranceTooLongError",
    "available_cpus",
    "model_dir",
    "soft_limit",
]

# BLUE_MODEL_DIR_ENV ("BLUE25_MODEL_DIR") is the same override contract as
# engine.ENGINE_DIR_ENV: point it at an unpacked snapshot and nothing is
# downloaded, so local development never hits the network. It is defined in the
# registry, which needs it too (is_installed), and re-exported here.

# --- app-level heuristics, NOT model parameters -----------------------------
# Both of these are hardcoded in the reference app (app.py:113 and app.py:117)
# and have no counterpart in tts.json or any exported graph. The pace blend
# pulls the predicted duration halfway toward a flat 0.0625 s per *character*
# token, so its effect depends on how many diacritics the transcription style
# emits. Keeping the reference values keeps this runtime's timing identical to
# the reference app's; pace_blend=0.0 would give the model's own duration.
DURATION_PACE_DPT_REF = 0.0625  # seconds per text token
DEFAULT_PACE_BLEND = 0.5

# Defaults. n_steps/cfg_scale are the reference's; speed is not — the reference
# ships 1.2, which measured 0.975 s for a 20-phoneme Yiddish utterance against
# 1.46 s at 1.0. 1.0 is the natural rate for Yiddish here.
DEFAULT_N_STEPS = 8
DEFAULT_CFG_SCALE = 4.0
DEFAULT_SPEED = 1.0

# The male LibriTTS reader: of the five bundled styles it is the one whose
# synthesized F0 tracked the published per-voice figure most closely
# (128 Hz documented, 123-131 Hz measured over three utterances) and it is the
# reference CLI's own default.
# Public voice names. The bundle ships LibriTTS-derived style files whose stems
# ("libri_male_6209") name a speaker id in a corpus, which is meaningless to
# anyone choosing a voice, so each is presented under a Yiddish given name.
#
# The mapping is by style-file stem, and the underlying stems remain accepted as
# aliases so any caller or script written against the raw ids keeps working.
# Genders follow the model card's documented F0: 6209 ~128 Hz and 8088 male,
# 1088 ~204 Hz and 6147 ~211 Hz female.
VOICE_NAMES: dict[str, str] = {
    "Berl": "libri_male_6209",
    "Hershl": "libri_male_8088",
    "Sheyndl": "libri_female_1088",
    "Rukhl": "libri_female_6147",
}

#: Style files present in the bundle but deliberately not offered. `female` is
#: withheld at the project owner's request; its style file stays in the snapshot
#: (it is part of the published bundle) and is simply never listed or resolved.
WITHHELD_VOICES: frozenset[str] = frozenset({"female"})

DEFAULT_VOICE = "Berl"

# app.py:1845 rejects any voice JSON whose style_ttl std exceeds this as
# incompatible with the checkpoint. All five 2.5 voices measure 0.055-0.064.
_STYLE_TTL_STD_LIMIT = 0.3

# The one honest place to refuse a too-long request: the LENGTH OF THE TEXT,
# checked before the duration predictor runs.
#
# The bundle README says duration "saturates near 16.5 s". Measured on this
# export, the raw predictor's ceiling is lower and the saturation is gradual, so
# a threshold on predicted seconds cannot express it. Seconds per text token for
# libri_male_6209 (identical to within 1 % on the other four voices):
#
#     T_text     28    79   134   184   222   244   272   310   360   1410
#     s/token  .0434 .0464 .0462 .0450 .0433 .0422 .0408 .0382 .0351 .0106
#     total s   1.22  3.66  6.19  8.28  9.62 10.31 11.09 11.83 12.64 14.99
#
# The rate plateaus at ~0.0464 s/token around T_text 80-135 and then decays
# monotonically: the model stops making the utterance longer while the text
# keeps growing, so the same words are crammed into less and less time. Total
# predicted duration asymptotes at ~15.0 s, i.e. BELOW any 16 s threshold —
# which is why the old check on predicted seconds was unreachable: an 879-char
# IPA string sailed through it and rendered at 3x natural rate.
#
# 240 tokens is the largest length at which the rate is still within 10 % of the
# plateau (0.0422 / 0.0464 = 0.91 at T_text 244), i.e. the cram is inaudible.
# Two more things the cap buys, both of which a seconds-based check could not:
#   * it is evaluated BEFORE duration_predictor_style runs, whose activations
#     grow QUADRATICALLY with T_text. Peak RSS above a 0.44 GB baseline, one
#     predictor call per process: 240 chars +0.02 GB, 550 +0.13, 1 100 +0.57,
#     2 200 +2.23 — 4x per doubling, so ~9 GB at 4 400 characters and past 16 GB
#     before the pydantic limit that used to be 20 000. A single unauthenticated
#     request could therefore OOM-kill the container, and the old check could
#     not prevent it because it ran *after* the allocation. This one rejects
#     first;
#   * it is independent of ``speed`` and ``pace_blend``. A long duration
#     produced by speed<1 or by the pace blend's linear 0.0625*T_text term is a
#     legitimate request, not saturation, and refusing it (the old check did,
#     at speed 0.8 on 265 characters) refused audio that renders correctly.
#     T_lat is a symbolic dimension in every graph; there is no ceiling there.
MAX_TEXT_TOKENS = 240

_ONNX_FILES = (
    "duration_predictor_style.onnx",
    "text_encoder.onnx",
    "vector_estimator.onnx",
    "vocoder.onnx",
)

# app.py:142 — the reference wraps text in <lang> tags during preprocessing and
# replaces each tag with a single space before encoding.
_LANG_TAG_RE = re.compile(r"</?[^>]+>")
_ENDS_IN_PUNCT_RE = re.compile(r"[.!?;:,'\"')\]}…]$")

# Peak limiting, not RMS normalisation: the five voices differ by ~6 dB in
# natural loudness and that difference is speaker identity, not an error.
# Output legitimately exceeds +-1.0 inside speech (1.32 measured at the default
# cfg_scale=4.0, 1.41 at cfg 6.0) and every PCM16 writer clips that silently.
#
# The limiter is a STATIC soft knee, not the recipe's `0.95/peak` gain, and the
# difference matters as soon as an utterance is assembled from more than one
# call. `0.95/peak` is a per-signal gain, so two sentences of one paragraph get
# two different gains purely because of where each one's loudest sample landed:
# measured on libri_male_6209 at cfg 4.0, `mit a pˈur jur ʦirˈik` peaked at
# 0.746 (gain 1.0) between two neighbours that peaked at 1.23 and 0.66, and the
# assembled audio stepped 2.9 dB at a sentence boundary — audible pumping, and
# a per-voice loudness that depends on how the text happened to be chunked.
#
# A static curve has no such dependence: every chunk of every request gets the
# same transfer function, so relative loudness is preserved exactly. It costs
# almost nothing in fidelity because the overshoot is sparse — measured over
# three utterances at cfg 4.0 and 6.0, 0.0-0.35 % of samples exceed the knee and
# 0.0-0.14 % exceed the ceiling — and tanh is C1 at the knee, so there is no
# corner to buzz. Below the knee the waveform is bit-identical to the vocoder's.
_LIMIT_KNEE = 0.75
_LIMIT_CEILING = 0.95


class _IncompatibleStyle(RuntimeError):
    """A voice JSON whose style tensors do not belong to this checkpoint."""


class UtteranceTooLongError(ValueError):
    """The text is longer than the checkpoint can render without cramming.

    A caller error, not a server fault: the API layer maps it to 400 and every
    synthesis path chunks first so it is only reachable for a single
    unsplittable run of more than ``MAX_TEXT_TOKENS`` tokens.
    """


def available_cpus() -> int:
    """How many CPUs this process may really use, capped at 8.

    ``os.cpu_count()`` reports the HOST's cores, not the cgroup quota a
    container is held to: on a 2-vCPU Hugging Face Space it commonly answers
    8-16, and four ONNX sessions each opening that many intra-op threads put
    dozens of workers on two cores. The quota is read from cgroup v2
    (``cpu.max``) and v1 (``cpu.cfs_quota_us``/``cpu.cfs_period_us``); on a
    laptop neither exists and ``os.cpu_count()`` is the right answer.
    ``PHONIKUD_YI_THREADS`` overrides everything, for a deployment that knows
    better.
    """
    override = os.environ.get("PHONIKUD_YI_THREADS", "").strip()
    if override.isdigit() and int(override) > 0:
        return int(override)

    host = os.cpu_count() or 1
    quota: float | None = None
    try:
        text = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if text[0] != "max":
            quota = int(text[0]) / int(text[1])
    except (OSError, ValueError, IndexError):
        try:
            micro = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if micro > 0 and period > 0:
                quota = micro / period
        except (OSError, ValueError):
            quota = None
    if quota is not None and quota >= 1:
        host = min(host, int(quota))
    return max(1, min(8, host))


def soft_limit(
    samples: np.ndarray,
    knee: float = _LIMIT_KNEE,
    ceiling: float = _LIMIT_CEILING,
) -> np.ndarray:
    """Static soft-knee peak limiter: |x| <= `knee` is untouched, output <= `ceiling`.

    ``|x| > knee`` is compressed as ``knee + (ceiling - knee) * tanh(t)`` with
    ``t = (|x| - knee) / (ceiling - knee)``. tanh has unit slope at 0, so the
    curve is continuous and C1 at the knee, monotone above it, and asymptotic to
    ``ceiling`` — nothing can clip in PCM16, and nothing below the knee is
    changed by a single bit.

    Static is the whole point: the transfer function does not depend on the
    signal, so a waveform assembled from several ``synthesize()`` calls has one
    consistent level. See the note beside ``_LIMIT_KNEE`` for the 2.9 dB step
    the signal-dependent ``0.95/peak`` gain produced at chunk boundaries.
    """
    wav = np.asarray(samples, dtype=np.float32)
    if wav.size == 0:
        return wav
    magnitude = np.abs(wav)
    over = magnitude > knee
    if not over.any():
        return wav
    span = ceiling - knee
    out = wav.copy()
    excess = (magnitude[over] - knee) / span
    out[over] = np.sign(wav[over]) * (knee + span * np.tanh(excess))
    log.debug(
        "soft-limited peak %.3f -> %.3f (%.3f%% of samples above the knee)",
        float(magnitude.max()), float(np.abs(out).max()),
        100.0 * float(over.mean()),
    )
    return out


def model_dir() -> Path:
    """Where the BlueTTS 2.5 bundle lives, downloading it on first use.

    ``BLUE25_MODEL_DIR`` points at an already-unpacked snapshot and suppresses
    the download entirely. Otherwise ~282 MB is fetched from
    ``notmax123/BlueTTS2.5-onnx``. ``HF_HOME`` and ``HF_TOKEN`` are honoured
    from the environment, the token explicitly so a private revision keeps
    working.
    """
    override = os.environ.get(BLUE_MODEL_DIR_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not (path / "tts.json").is_file():
            raise FileNotFoundError(
                f"{BLUE_MODEL_DIR_ENV}={override} does not contain tts.json; "
                "point it at an unpacked BlueTTS2.5-onnx snapshot, or unset it "
                f"to download {BLUE_REPO_ID} from Hugging Face."
            )
        return path

    from huggingface_hub import snapshot_download  # imported late: network dependency

    log.info("downloading %s@%s (~282 MB on a cold cache)", BLUE_REPO_ID, BLUE_REVISION)
    return Path(
        snapshot_download(
            repo_id=BLUE_REPO_ID,
            revision=BLUE_REVISION,
            token=os.environ.get("HF_TOKEN"),
        )
    ).resolve()


@dataclass(frozen=True)
class _Style:
    """One saved voice: the two style tensors the graphs consume."""

    ttl: np.ndarray  # float32 [1, 50, 256] -> text_encoder, vector_estimator
    dp: np.ndarray   # float32 [1, 8, 16]   -> duration_predictor_style


def _style_tensor(payload: dict, key: str) -> np.ndarray:
    """Parse ``style_ttl``/``style_dp`` out of a voice JSON.

    ``data`` is a *nested* list matching ``dims`` (``len(data) == 1``, not
    12800), so it has to be flattened before it is reshaped — app.py:855 does
    the same ``.flatten().reshape(dims)`` dance for the same reason.
    """
    block = payload[key]
    flat = np.array(block["data"], dtype=np.float32).reshape(-1)
    return flat.reshape(tuple(int(d) for d in block["dims"]))


class BlueYiddish:
    """Runtime implementation over the BlueTTS 2.5 ONNX bundle.

    Cheap metadata (tts.json, vocab.json, stats.npz, uncond.npz, the voice
    directory listing) is read in the constructor; the four ONNX sessions cost
    ~1.5 s to build and are therefore created lazily on first use, once, under
    a lock that is released before any inference runs. Call ``prepare()`` to
    pay that cost at startup instead of on the first request.
    """

    def __init__(
        self,
        directory: Path | str | None = None,
        runtime_id: str = "blue_yi",
        model_name: str = "BlueTTS 2.5 (Yiddish IPA)",
    ) -> None:
        self.id = runtime_id
        self.model_name = model_name
        self.model_path = Path(directory) if directory is not None else model_dir()

        cfg = json.loads((self.model_path / "tts.json").read_text(encoding="utf-8"))
        self.sample_rate = int(cfg["ae"]["sample_rate"])                    # 44100
        self._base_chunk_size = int(cfg["ae"]["base_chunk_size"])           # 512
        self._compress_factor = int(cfg["ttl"]["chunk_compress_factor"])    # 6
        self._latent_dim = int(cfg["ttl"]["latent_dim"])                    # 24
        # Samples per latent frame, and therefore the output length quantum.
        self._frame_len = self._base_chunk_size * self._compress_factor     # 3072
        # The compressed channel count the flow operates on.
        self._compressed_dim = self._latent_dim * self._compress_factor     # 144

        vocab = json.loads((self.model_path / "vocab.json").read_text(encoding="utf-8"))
        self._pad_id = int(vocab.get("pad_id", 0))
        self._char_to_id: dict[str, int] = {
            str(k): int(v) for k, v in vocab["char_to_id"].items()
        }

        stats = np.load(self.model_path / "stats.npz")
        self._mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, -1, 1)
        self._std = np.asarray(stats["std"], dtype=np.float32).reshape(1, -1, 1)
        # npz wins over tts.json's ttl.normalizer.scale (app.py:795); they agree
        # at 0.25 in this bundle, but the npz is welded to these graphs.
        if "normalizer_scale" in stats.files:
            self._normalizer_scale = float(np.asarray(stats["normalizer_scale"]).reshape(-1)[0])
        else:
            self._normalizer_scale = float(cfg["ttl"]["normalizer"]["scale"])
        if self._normalizer_scale == 0.0:
            raise ValueError(f"stats.npz normalizer_scale is 0 in {self.model_path}")

        uncond = np.load(self.model_path / "uncond.npz")
        self._u_text = np.asarray(uncond["u_text"], dtype=np.float32)  # [1,256,1]
        self._u_ref = np.asarray(uncond["u_ref"], dtype=np.float32)    # [1,50,256]
        # u_text carries a single text position, so its mask is [1,1,1].
        self._u_text_mask = np.ones((1, 1, 1), dtype=np.float32)

        # Stat the four graphs now, in the constructor, rather than letting the
        # lazy session build discover a missing file on the first request. The
        # loader turns FileNotFoundError into RuntimeNotAvailable(MISSING_FILES),
        # i.e. a 503 `not_available` at load time and a red `/health`; without
        # this an interrupted 282 MB snapshot_download left `/health` saying
        # "ready" and every synthesis returning 500.
        for filename in _ONNX_FILES:
            if not (self.model_path / filename).is_file():
                raise FileNotFoundError(
                    f"{filename} is missing from {self.model_path}; the "
                    f"{BLUE_REPO_ID} snapshot is incomplete"
                )

        self._voice_files: dict[str, Path] = {
            path.stem: path
            for path in sorted((self.model_path / "voices").glob("*.json"))
            if path.stem not in WITHHELD_VOICES
        }
        if not self._voice_files:
            raise FileNotFoundError(f"no voices/*.json under {self.model_path}")

        self._styles: dict[str, _Style] = {}
        self._voices: list[str] | None = None
        self._style_lock = threading.Lock()

        # ONNX state, all filled by _ensure_sessions().
        self._sessions_lock = threading.Lock()
        self._dp = None
        self._text_enc = None
        self._vf = None
        self._vocoder = None
        self._vf_inputs: frozenset[str] = frozenset()
        self._vf_cfg_is_baked = False
        self._vocoder_input = ""
        #: The vocoder's declared channel count, or None when the graph declares
        #: that axis symbolically. None means "unknown", never "no fold".
        self._vocoder_channels: int | None = None

    # ------------------------------------------------------------------
    # Protocol surface
    # ------------------------------------------------------------------
    def voices(self) -> list[str]:
        """The saved voices, sorted; ``DEFAULT_VOICE`` is used when none is asked for.

        Incompatible styles are excluded, not merely deprioritised: app.py:1845
        skips any voice whose ``style_ttl.std()`` exceeds 0.3 because such a
        style belongs to a different checkpoint and renders as noise. All five
        bundled 2.5 voices pass (0.055-0.064).
        """
        if self._voices is None:
            with self._style_lock:
                if self._voices is None:
                    usable = []
                    for name in self._voice_files:
                        try:
                            self._load_style_locked(name)
                        except _IncompatibleStyle as exc:
                            log.warning("skipping voice %s: %s", name, exc)
                            continue
                        usable.append(name)
                    # Report the public names, not the style-file stems.
                    by_stem = {stem: name for name, stem in VOICE_NAMES.items()}
                    self._voices = sorted(by_stem.get(stem, stem) for stem in usable)
        return list(self._voices)

    def vocab(self) -> set[str]:
        """Every character the model has an embedding for (``vocab.json``)."""
        return set(self._char_to_id)

    def prepare(self) -> None:
        """Build the ONNX sessions now (~1.5 s) so no request pays for it."""
        self._ensure_sessions()

    def synthesize(
        self,
        ipa: str,
        voice: str = "",
        speed: float = DEFAULT_SPEED,
        *,
        n_steps: int = DEFAULT_N_STEPS,
        cfg_scale: float = DEFAULT_CFG_SCALE,
        pace_blend: float = DEFAULT_PACE_BLEND,
        seed: int | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Render ONE utterance. Returns (float32 mono in [-1,1], dropped units).

        ``dropped`` lists every *phone-bearing* character that had no embedding
        and was therefore omitted — the vocab's own note says unknown symbols
        vanish silently, so they are counted, logged and reported instead.
        Punctuation the vocab cannot spell is omitted too but is NOT reported:
        it is structure, not a phone (see ``phones.NON_PHONE``). For Yiddish IPA from
        phonikud-yi this list is always empty; anything in it means something
        upstream emitted off-inventory text (the classic case being ASCII ``'``
        U+0027 for stress instead of ``ˈ`` U+02C8, or ASCII ``g`` for ``ɡ``
        U+0261 — both are *in* vocab, so they are not dropped, they are simply
        the wrong phone; only genuinely unmapped characters show up here).

        ``speed`` > 1.0 is faster. Deterministic for a given ``seed``; with
        ``seed=None`` each call draws fresh noise. Raises
        ``UtteranceTooLongError`` when the text is longer than
        ``MAX_TEXT_TOKENS`` tokens — the caller chunks, this class renders one
        chunk.
        """
        if speed <= 0:
            raise ValueError("speed must be greater than 0")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        name = (voice or "").strip() or DEFAULT_VOICE
        style = self._style(name)
        self._ensure_sessions()

        started = time.perf_counter()
        text_ids, text_mask, dropped = self._encode(ipa)
        t_text = int(text_ids.shape[1])
        if t_text > MAX_TEXT_TOKENS:
            # BEFORE any graph runs: duration_predictor_style's activations grow
            # quadratically with T_text, so this is also the memory guard.
            raise UtteranceTooLongError(
                f"{t_text} text tokens exceeds this checkpoint's usable "
                f"{MAX_TEXT_TOKENS} (about {MAX_TEXT_TOKENS - 3} characters of "
                "IPA): past that the duration predictor stops lengthening the "
                "utterance and the same words are crammed into less time. Split "
                "the text into sentences and synthesize one at a time."
            )
        if dropped:
            log.warning(
                "%d character(s) had no embedding and were omitted: %s",
                len(dropped),
                " ".join(f"{ch!r}(U+{ord(ch):04X})" for ch in dropped),
            )

        duration = self._duration(text_ids, text_mask, style, speed, pace_blend)
        xt, latent_mask = self._noisy_latent(duration, seed)
        text_emb, *_ = self._text_enc.run(
            None,
            {"text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask},
        )

        vf_started = time.perf_counter()
        xt = self._flow(xt, text_emb, style, latent_mask, text_mask, n_steps, cfg_scale)
        vf_elapsed = time.perf_counter() - vf_started

        wav = self._decode(xt)
        elapsed = time.perf_counter() - started
        log.debug(
            "blue_yi voice=%s T_text=%d T_lat=%d dur=%.3fs steps=%d cfg=%.2f "
            "audio=%.3fs total=%.0fms vf_loop=%.0fms (%.0f%%)",
            name, text_ids.shape[1], int(xt.shape[2]), float(duration[0]), n_steps,
            cfg_scale, wav.size / self.sample_rate, elapsed * 1e3, vf_elapsed * 1e3,
            100.0 * vf_elapsed / elapsed if elapsed else 0.0,
        )
        return wav, dropped

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def _ensure_sessions(self) -> None:
        if self._vf is not None:
            return
        with self._sessions_lock:
            if self._vf is not None:
                return
            import onnxruntime as ort  # heavy; and only needed once we synthesize

            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            # Batch is fixed at 1 in every graph, so all the parallelism there
            # is to have is intra-op. app.py:953-960 uses exactly these.
            opts.intra_op_num_threads = available_cpus()
            opts.inter_op_num_threads = 1

            started = time.perf_counter()
            built = {}
            for filename in _ONNX_FILES:
                path = self.model_path / filename
                if not path.is_file():
                    raise FileNotFoundError(f"{filename} is missing from {self.model_path}")
                built[filename] = ort.InferenceSession(
                    str(path), sess_options=opts, providers=["CPUExecutionProvider"]
                )

            self._dp = built["duration_predictor_style.onnx"]
            self._text_enc = built["text_encoder.onnx"]
            vf = built["vector_estimator.onnx"]
            vocoder = built["vocoder.onnx"]

            self._vf_inputs = frozenset(i.name for i in vf.get_inputs())
            vocoder_input = vocoder.get_inputs()[0]
            self._vocoder_input = vocoder_input.name
            channels = vocoder_input.shape[1]
            self._vocoder_channels = int(channels) if isinstance(channels, int) else None
            # Fail here, at session build time, rather than with an obscure ORT
            # shape error inside a request. This class can feed 24 channels
            # (folded) or 144 (compressed) and nothing else; a bundle that wants
            # some third width is a different checkpoint.
            if self._vocoder_channels not in (
                None, self._latent_dim, self._compressed_dim
            ):
                raise RuntimeError(
                    f"vocoder.onnx declares {self._vocoder_channels} input "
                    f"channels; this code can feed {self._latent_dim} "
                    f"(folded) or {self._compressed_dim} (compressed). "
                    f"{BLUE_REPO_ID}@{BLUE_REVISION} is not a BlueTTS 2.5 "
                    "shaped bundle."
                )

            # Both questions are answered by the graphs themselves, never by a
            # repo-name string: BlueV3-onnx bakes guidance into the vector
            # estimator (it has a cfg_scale input) and its vocoder eats the
            # compressed 144-channel latent; 2.5 has no cfg_scale input and its
            # vocoder wants 24 channels. Applying external CFG to a graph that
            # already bakes it in over-amplifies (app.py:970-973), so a future
            # bundle swap must not silently take the wrong branch.
            self._vf_cfg_is_baked = "cfg_scale" in self._vf_inputs
            # `self._vf` is the flag the lock-free fast path at the top of this
            # method tests, so it is published LAST, after every session and
            # every attribute derived from one. Assigning it before
            # `self._vocoder` (as this did) let a second thread pass the
            # initialization check in the one-bytecode window between the two
            # statements and reach `self._vocoder.run(...)` on None.
            self._vocoder = vocoder
            self._vf = vf
            log.info(
                "blue_yi sessions built in %.2fs from %s (cfg_baked=%s, "
                "vocoder_channels=%s, sample_rate=%d)",
                time.perf_counter() - started, self.model_path,
                self._vf_cfg_is_baked, self._vocoder_channels or "dynamic",
                self.sample_rate,
            )

    # ------------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------------
    def _style(self, name: str) -> _Style:
        # A public name ("Berl") or the raw style-file stem it maps to; the stems
        # stay accepted so callers written against the bundle's own ids work.
        name = VOICE_NAMES.get(name, name)
        cached = self._styles.get(name)
        if cached is not None:
            return cached
        if name not in self._voice_files:
            raise ValueError(
                f"unknown voice `{name}`; this bundle has "
                f"{', '.join(self.voices())}"
            )
        with self._style_lock:
            try:
                return self._load_style_locked(name)
            except _IncompatibleStyle as exc:
                raise ValueError(f"voice `{name}` is not usable: {exc}") from exc

    def _load_style_locked(self, name: str) -> _Style:
        cached = self._styles.get(name)
        if cached is not None:
            return cached
        payload = json.loads(self._voice_files[name].read_text(encoding="utf-8"))
        ttl = _style_tensor(payload, "style_ttl")
        dp = _style_tensor(payload, "style_dp")
        std = float(ttl.std())
        if std > _STYLE_TTL_STD_LIMIT:
            raise _IncompatibleStyle(
                f"style_ttl std {std:.3f} exceeds {_STYLE_TTL_STD_LIMIT} — the "
                "style was exported from a different checkpoint"
            )
        style = _Style(ttl=ttl, dp=dp)
        self._styles[name] = style
        return style

    # ------------------------------------------------------------------
    # Text -> ids
    # ------------------------------------------------------------------
    def _encode(self, ipa: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """IPA string -> (int64 [1,T_text], float32 [1,1,T_text], dropped).

        Reproduces every preprocessing step that changes the token stream
        (app.py:700 + app.py's tag stripping in _encode):

        1. NFC — the vocab holds precomposed IPA, NFD would split characters
           into base + combining mark and change the ids.
        2. collapse whitespace and strip.
        3. append ``"."`` when the text does not already end in punctuation.
           ``.`` is a real token: it lengthens T_text, which lengthens the
           predicted duration, and it adds the utterance-final fall.
        4. one leading and one trailing space (id 3), which is what the
           reference's ``<yi>…</yi>`` wrapping leaves behind once each tag is
           replaced by a single space.

        Drop any of these and T_text is short and the utterance sounds clipped.
        """
        text = unicodedata.normalize("NFC", ipa)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            raise ValueError("nothing to synthesize: the IPA string is empty")
        if not _ENDS_IN_PUNCT_RE.search(text):
            text += "."
        prepared = _LANG_TAG_RE.sub(" ", f"<yi>{text}</yi>")

        ids: list[int] = []
        dropped: list[str] = []
        seen: set[str] = set()
        for ch in prepared:
            mapped = self._char_to_id.get(ch)
            if mapped is not None:
                ids.append(mapped)
                continue
            # No embedding. Omit the character rather than substitute PAD: PAD
            # in the middle of a sequence is padding, i.e. it means "nothing is
            # here", so writing it there asks the model to read a hole.
            if ch in phones.NON_PHONE:
                # Punctuation the engine passes through and this vocab cannot
                # spell ([ ] ׃ „ ‚ ‹ › | < > { } are all missing here). It
                # carries no phonetic content, so it is not a casualty and is
                # not reported -- reporting it lit the UI's "the audio does not
                # match the IPA at [ ]" strip on ordinary bracketed Yiddish.
                continue
            if ch not in seen:
                seen.add(ch)
                dropped.append(ch)
        if not ids:
            raise ValueError("nothing to synthesize: no character had an embedding")
        text_ids = np.array([ids], dtype=np.int64)
        # Batch 1, so every position is real: the mask is all ones (app.py:765).
        text_mask = np.ones((1, 1, text_ids.shape[1]), dtype=np.float32)
        return text_ids, text_mask, dropped

    # ------------------------------------------------------------------
    # Duration -> noise
    # ------------------------------------------------------------------
    def _duration(
        self,
        text_ids: np.ndarray,
        text_mask: np.ndarray,
        style: _Style,
        speed: float,
        pace_blend: float,
    ) -> np.ndarray:
        predicted, *_ = self._dp.run(
            None,
            {"text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask},
        )
        seconds = np.asarray(predicted, dtype=np.float64).reshape(-1)

        blend = min(max(float(pace_blend), 0.0), 1.0)
        if blend > 0.0:
            # app.py:500 — pull seconds-per-token toward a stable reference.
            tokens = np.maximum(
                np.asarray(text_mask, dtype=np.float64).sum(axis=(1, 2)), 1.0
            ).reshape(-1)
            per_token = seconds / tokens
            seconds = ((1.0 - blend) * per_token + blend * DURATION_PACE_DPT_REF) * tokens
        # No length check here, deliberately. Saturation is a property of the
        # TEXT (see MAX_TEXT_TOKENS, checked in synthesize before this runs); a
        # long duration produced by speed<1 or by the pace blend's linear
        # 0.0625*T_text term is a legitimate request and renders correctly.
        return (seconds / max(float(speed), 1e-6)).astype(np.float32)

    def _noisy_latent(self, duration: np.ndarray, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
        """app.py:1003 — Gaussian latent of the length the duration implies.

        Output length is quantized to whole latent frames, so the rendered
        audio is ``frame_len * (T_lat - 2)`` samples after the edge trim.
        """
        samples = float(duration.max()) * self.sample_rate
        t_lat = max(int(math.ceil(samples / self._frame_len)), 1)
        rng = np.random.default_rng(seed)  # seed=None -> fresh entropy per call
        xt = rng.standard_normal((1, self._compressed_dim, t_lat)).astype(np.float32)
        # Batch 1: every latent frame is in use, so the mask is all ones and
        # multiplying by it is a no-op kept for shape parity with the reference.
        latent_mask = np.ones((1, 1, t_lat), dtype=np.float32)
        return xt * latent_mask, latent_mask

    # ------------------------------------------------------------------
    # Flow matching
    # ------------------------------------------------------------------
    def _flow(
        self,
        xt: np.ndarray,
        text_emb: np.ndarray,
        style: _Style,
        latent_mask: np.ndarray,
        text_mask: np.ndarray,
        n_steps: int,
        cfg_scale: float,
    ) -> np.ndarray:
        """Euler integration with classifier-free guidance.

        The graph bakes the Euler step in: it returns ``x + v/total_step``, not
        ``v``. Blending the two graph *outputs* is therefore identical to
        blending velocities, because both passes are fed the same ``xt`` and the
        shared ``x`` cancels in the difference:

            out_u + s*(out_c - out_u) = x + (1/N)*(v_u + s*(v_c - v_u))

        (verified numerically to 1.3e-15). ``current_step``/``total_step`` are
        float32[1] — int64 fails ORT's type check.
        """
        total_step = np.array([n_steps], dtype=np.float32)
        scale = float(cfg_scale)
        # cfg_scale == 1.0 is the identity blend, so the uncond pass would be
        # computed and discarded: skip it (measured 104 ms vs 178 ms).
        external_cfg = not self._vf_cfg_is_baked and scale != 1.0

        for step in range(n_steps):
            current_step = np.array([step], dtype=np.float32)
            cond = {
                "noisy_latent": xt,
                "text_emb": text_emb,
                "style_ttl": style.ttl,
                "latent_mask": latent_mask,
                "text_mask": text_mask,
                "current_step": current_step,
                "total_step": total_step,
            }
            if self._vf_cfg_is_baked:
                # BlueV3-style graph: guidance happens inside, one pass only.
                cond["cfg_scale"] = np.array([scale], dtype=np.float32)
                xt, *_ = self._vf.run(None, cond)
                continue
            out_cond, *_ = self._vf.run(None, cond)
            if not external_cfg:
                xt = out_cond
                continue
            out_uncond, *_ = self._vf.run(
                None,
                {
                    "noisy_latent": xt,  # the same xt as the cond pass, on purpose
                    "text_emb": self._u_text,
                    "style_ttl": self._u_ref,
                    "latent_mask": latent_mask,
                    "text_mask": self._u_text_mask,
                    "current_step": current_step,
                    "total_step": total_step,
                },
            )
            xt = out_uncond + scale * (out_cond - out_uncond)
        return xt

    # ------------------------------------------------------------------
    # Latent -> waveform
    # ------------------------------------------------------------------
    def _decode(self, xt: np.ndarray) -> np.ndarray:
        # Denormalize BEFORE the fold. Skipping this still yields speech-like
        # audio with roughly the right F0 but the wrong spectral balance and an
        # inflated level — the failure most likely to ship unnoticed. For this
        # checkpoint the answer is unconditionally "denormalize"; do not port
        # the reference's _vocoder_stats_are_identity() probe, which needs the
        # `onnx` package and defaults to True when it cannot answer.
        z = (xt / self._normalizer_scale) * self._std + self._mean

        # Fold unless the graph explicitly says it wants the compressed latent.
        # The default is deliberately "fold": this class only supports
        # 2.5-shaped bundles, where the fold is non-negotiable (RECIPE §7), and
        # the channel axis can legitimately come back symbolic — as T_lat,
        # T_text and T_ref already do in vector_estimator. Treating an unknown
        # dimension as "no fold" (the previous `int(channels) if isinstance(...)
        # else 0` did) would silently hand a [1,144,T] latent to a 24-channel
        # vocoder. A width that is neither 24 nor 144 was already rejected when
        # the session was built.
        if self._vocoder_channels != self._compressed_dim:
            z = self._decompress(z)

        wav, *_ = self._vocoder.run(None, {self._vocoder_input: z.astype(np.float32)})
        wav = np.asarray(wav, dtype=np.float32)

        # Mandatory edge trim (app.py:1094). The final latent frame is often a
        # click ~25 dB above speech (per-frame peak 9.37 vs a 0.14 body RMS in
        # one measured case) and the leading frame is near-silent. Trim BEFORE
        # measuring level: peak-limiting first would attenuate the whole
        # utterance by ~20 dB.
        if wav.shape[-1] > 2 * self._frame_len:
            wav = wav[..., self._frame_len:-self._frame_len]
        wav = wav.reshape(-1)

        return np.ascontiguousarray(soft_limit(wav), dtype=np.float32)

    def _decompress(self, z: np.ndarray) -> np.ndarray:
        """[1,144,T] -> [1,24,6T] (numpy port of models/utils.py:16).

        Exactly one interleaving is correct: compressed channel
        ``c = orig_channel * 6 + phase`` and decompressed time
        ``t = t_lat * 6 + phase``, i.e. the channel axis is 24-major /
        phase-minor, so ``z_dec[0, ch, t*6 + p] == z[0, ch*6 + p, t]``. Four
        wrong interleavings were A/B tested against real audio: all of them
        produce float output of the correct length (a shape assertion catches
        nothing) and only this one produces speech.
        """
        batch, channels, t_lat = z.shape
        if channels != self._compressed_dim:
            raise RuntimeError(f"expected {self._compressed_dim} latent channels, got {channels}")
        return (
            z.reshape(batch, self._latent_dim, self._compress_factor, t_lat)
            .transpose(0, 1, 3, 2)
            .reshape(batch, self._latent_dim, t_lat * self._compress_factor)
        )
