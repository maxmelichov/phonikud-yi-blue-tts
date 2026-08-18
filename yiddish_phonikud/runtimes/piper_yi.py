"""Piper adapter: Yiddish IPA into the bundled Piper voice.

The checkpoint is Hebrew-trained (espeak voice "he", phoneme_type "raw") and
is driven here with Yiddish IPA, so every phone the Yiddish inventory has but
this voice's phoneme_id_map lacks -- notably ʧ and ʤ -- must be folded to a
sequence the model knows before synthesis. Folding is reported, never silent.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .. import phones

logger = logging.getLogger(__name__)

# Tuned in the original Hebrew Phonikud Space; they are not defaults and
# changing them audibly degrades the voice.
_BASE_LENGTH_SCALE = 1.20
_NOISE_SCALE = 0.640
_NOISE_W = 1.0

# The checkpoint renders quietly enough to sound broken in a browser, so the
# original app doubled the amplitude before writing the WAV.
_VOLUME_GAIN = 2.0


class PiperYiddish:
    """Runtime implementation over piper_onnx.Piper."""

    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        runtime_id: str = "piper_yi",
        model_name: str = "Piper Yiddish (Hebrew acoustic model)",
    ) -> None:
        from piper_onnx import Piper  # heavy: loads onnxruntime

        self.id = runtime_id
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        # piper_onnx takes plain path strings, not Path objects.
        self._piper = Piper(str(self.model_path), str(self.config_path))
        self._config = json.loads(self.config_path.read_text(encoding="utf-8"))
        # Fixed at construction and never written again. `synthesize` used to
        # assign `self.sample_rate = int(sample_rate)` from Piper's own return
        # value, i.e. it mutated a process-wide singleton mid-request: the
        # streaming path builds `X-Sample-Rate` from this attribute BEFORE the
        # first chunk and every frame's WAV header from it AFTER, and
        # /v1/models/state reads it concurrently, so the moment the config and
        # the checkpoint disagreed a stream would advertise one rate and carry
        # another. A disagreement is now logged once, at the first synthesis,
        # and the reported rate stays the one the caller was promised.
        self.sample_rate = int(self._config.get("audio", {}).get("sample_rate", 22050))
        self._rate_checked = False
        self._vocab: set[str] | None = None

    def voices(self) -> list[str]:
        # Single-speaker checkpoint: there is no speaker id to choose.
        return ["default"]

    def vocab(self) -> set[str]:
        """The symbols the model accepts, read straight from the config JSON."""
        if self._vocab is None:
            # Read from the parsed config rather than the Piper object, whose
            # internals are not part of piper_onnx's public surface.
            self._vocab = set(self._config.get("phoneme_id_map", {}))
        return self._vocab

    def synthesize(
        self, ipa: str, voice: str = "", speed: float = 1.0, **options
    ) -> tuple[np.ndarray, list[str]]:
        """Renders one utterance: (float32 mono in [-1, 1], dropped units).

        ``dropped`` is returned rather than stashed on the instance: the runtime
        is a process-wide singleton, so per-instance state let concurrent
        requests read each other's fold report.

        This checkpoint has no sampler and no speaker table, so ``voice`` (only
        "" or "default" exist) and Blue's ``n_steps`` / ``cfg_scale`` / ``seed``
        are accepted through ``**options`` and ignored — a caller must be able
        to send one request body to either runtime.
        """
        if speed <= 0:
            raise ValueError("speed must be greater than 0")
        if voice and voice not in self.voices():
            raise ValueError(
                f"unknown voice {voice!r}; this checkpoint offers "
                f"{', '.join(self.voices())}"
            )
        folded, dropped = phones.fold_to_vocab(ipa, self.vocab())
        if dropped:
            logger.warning(
                "dropped %d phone(s) this voice cannot render: %s",
                len(dropped),
                " ".join(dropped),
            )

        samples, sample_rate = self._piper.create(
            folded,
            is_phonemes=True,
            # length_scale stretches duration, so it is inverse to perceived
            # speed: speed=2.0 halves it and the voice talks twice as fast.
            length_scale=_BASE_LENGTH_SCALE / float(speed),
            noise_scale=_NOISE_SCALE,
            noise_w=_NOISE_W,
        )
        if not self._rate_checked:
            self._rate_checked = True
            if int(sample_rate) != self.sample_rate:
                logger.error(
                    "%s reports %d Hz but %s declares %d Hz; the API has "
                    "already advertised %d Hz, so re-export the voice or fix "
                    "the config rather than trusting the checkpoint here",
                    self.model_path.name, int(sample_rate),
                    self.config_path.name, self.sample_rate, self.sample_rate,
                )

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        audio = np.clip(audio * _VOLUME_GAIN, -1.0, 1.0)
        return audio.astype(np.float32, copy=False), dropped


__all__ = ["PiperYiddish"]
