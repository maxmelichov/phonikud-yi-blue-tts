"""Yiddish Phonikud TTS — a Hasidic (Unterland/Central) Yiddish TTS Space.

Yiddish text goes through the phonikud-yi engine (v5 diacritizer + G2P) to get
nikud and closed-inventory IPA, and the IPA drives an acoustic runtime. The
Space exposes a MamboTTS-shaped API under ``/v1`` alongside a small web UI.

The default runtime (``DEFAULT_RUNTIME_ID``) is blue-yi: a Yiddish-trained
flow-matching acoustic model at 44.1 kHz with four saved voices, whose
character vocabulary covers the closed Yiddish inventory outright.

``piper_yi`` is the lightweight 22.05 kHz fallback, and it is the one carrying
the stronger caveat: a Hebrew-trained Piper checkpoint (espeak voice ``he``,
single speaker) driven with Yiddish IPA, which additionally has to fold ʧ/ʤ to
tʃ/dʒ because its ``phoneme_id_map`` lacks them. See ``registry.PIPER_YI``.

Importing this package stays free of heavy dependencies: only the runtime
catalog comes along. FastAPI, onnxruntime, numpy and the engine bindings live
behind ``yiddish_phonikud.api``, ``.runtimes`` and ``.engine``, which are
imported explicitly by the code that needs them.

``registry.runtimes()`` is re-exported here as ``runtime_catalog``, NOT as
``runtimes``: ``yiddish_phonikud.runtimes`` is a subpackage, and binding a
function to that name makes ``from yiddish_phonikud import runtimes`` return
whichever of the two was bound last — a real bug that broke the UI router.
"""

from __future__ import annotations

from yiddish_phonikud.registry import (
    DEFAULT_RUNTIME_ID,
    ENGINE_REPO_ID,
    ENGINE_REVISION,
    REGISTRY,
    InstallKind,
    ModelFile,
    RuntimeCapabilities,
    RuntimeManifest,
    is_installed,
    runtime,
)
from yiddish_phonikud.registry import runtimes as runtime_catalog

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_RUNTIME_ID",
    "ENGINE_REPO_ID",
    "ENGINE_REVISION",
    "REGISTRY",
    "InstallKind",
    "ModelFile",
    "RuntimeCapabilities",
    "RuntimeManifest",
    "__version__",
    "is_installed",
    "runtime",
    "runtime_catalog",
]
