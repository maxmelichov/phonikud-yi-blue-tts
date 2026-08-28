"""The single source of truth for TTS runtime metadata.

Adding an engine starts here: define its model manifest and capabilities,
then implement the matching runtime adapter under ``yiddish_phonikud.runtimes``.

Ported from MamboTTS's ``mambotts-registry`` crate, with two additions:
``RuntimeManifest.available`` — False for a runtime that is advertised in the
catalog but not implemented in this build, so ``/v1/models/load`` answers with
a clear build-specific error instead of pretending the runtime is present, the
way MamboTTS declares its Qwen runtime ahead of the platforms that can run it —
and ``RuntimeManifest.hf_repo_id``, for a bundle that is downloaded from Hugging
Face at first use rather than committed to the Space. The one runtime in this
build is implemented; ``available`` exists for the next one that is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

DEFAULT_RUNTIME_ID = "blue_yi"

# The Yiddish G2P + diacritizer bundle. Not a TTS runtime: every runtime in
# this catalog is fed IPA produced by it, so it is downloaded unconditionally
# at startup rather than being listed as a per-runtime file.
ENGINE_REPO_ID = "notmax123/phonikud-yi-engine"
# PINNED to a commit for the same reason blue-yi is (below): the engine decides
# every phoneme the voice speaks, and tracking a branch means a cold cache can
# fetch a new bundle unvetted. Still the v6 pointing model -- v7 was trained
# and measured WORSE on the phonetic eval (vs_gold_rule 73.13 -> 72.76), so it
# was not shipped. This revision adds the vowel-rule work on top: וי defaults
# to ɔj (Weinreich 44) with oʊ kept as a closed û-class list so אויך is not
# collapsed with הויז, plus ברויכ־, the פויל/פוילן homograph, and the
# די + bare טויב context read. model_pointed_lk at 7,633 types.
ENGINE_REVISION = "faf402b770cee967a0c3ca2b55609565a4c73bc5"

# The blue-yi acoustic bundle. Like the engine it is fetched with
# huggingface_hub rather than committed to the Space, so its manifest carries
# ``hf_repo_id`` and its files never live under the repo root.
BLUE_REPO_ID = "notmax123/blue-yi"
# PINNED to a commit, not "main". stats.npz and vocab.json are welded to this
# export (RECIPE G7/G14): the 144->24 fold, the 0.25 normalizer scale, the
# per-voice style shapes and every character id were verified against these
# exact files. Tracking a branch means a cold cache can fetch a re-export at any
# time and run verified code against unverified graphs — silently, because a
# wrong-but-plausible latent still decodes to audio-shaped float32. Bump this
# deliberately, and re-run scripts/selftest_space.py's F0 block when you do.
BLUE_REVISION = "34ee8856b85043b68cfbcaf0b3acad4c20326f88"
# Local-override env var for that bundle, the counterpart of
# engine.ENGINE_DIR_ENV. It lives here rather than in runtimes/blue_yi.py so
# is_installed() can honour it without importing the adapter (which would pull
# in onnxruntime and import this module back).
BLUE_MODEL_DIR_ENV = "BLUE25_MODEL_DIR"


class InstallKind(str, Enum):
    FILES = "files"
    ARCHIVE = "archive"


@dataclass(frozen=True)
class RuntimeCapabilities:
    yiddish: bool
    streaming: bool
    voice_reference: bool
    fixed_voices: bool


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str


@dataclass(frozen=True)
class RuntimeManifest:
    id: str
    name: str
    version: str
    size: str
    description: str
    directory: str
    install_kind: InstallKind
    files: tuple[ModelFile, ...]
    required_files: tuple[str, ...]
    capabilities: RuntimeCapabilities
    # False = declared in the catalog but not implemented in this build.
    available: bool = True
    # Set when the files come from a Hugging Face repo instead of the Space
    # itself. ``directory`` is then only a hint for a manual local install:
    # is_installed() and the runtime adapter both look in the HF cache.
    hf_repo_id: str = ""


BLUE_MODEL_BASE_URL = "https://huggingface.co/notmax123/blue-yi/resolve/main"

# The file list from blue-yi's onnx/manifest.json plus the saved voices, which
# sit at the repo root rather than under onnx/. duration_predictor.onnx and
# reference_encoder.onnx are part of the bundle and are listed for completeness,
# but the adapter never opens them (both need z_ref, and the autoencoder encoder
# that would produce it was not exported), so they are absent from
# required_files.
BLUE_YI_FILES: tuple[ModelFile, ...] = tuple(
    ModelFile(name=name, url=f"{BLUE_MODEL_BASE_URL}/{name}")
    for name in (
        "README.md",
        "onnx/manifest.json",
        "onnx/tts.json",
        "onnx/vocab.json",
        "onnx/stats.npz",
        "onnx/uncond.npz",
        "onnx/duration_predictor.onnx",
        "onnx/duration_predictor_style.onnx",
        "onnx/reference_encoder.onnx",
        "onnx/text_encoder.onnx",
        "onnx/vector_estimator.onnx",
        "onnx/vocoder.onnx",
        "voices/libri_female_1088.json",
        "voices/libri_female_6147.json",
        "voices/libri_male_6209.json",
        "voices/libri_male_8088.json",
    )
)

# Exactly what runtimes/blue_yi.py opens: the four usable graphs, the two npz
# tensor files, the two JSON configs, and the voice styles it discovers.
BLUE_YI_REQUIRED_FILES: tuple[str, ...] = (
    "onnx/tts.json",
    "onnx/vocab.json",
    "onnx/stats.npz",
    "onnx/uncond.npz",
    "onnx/duration_predictor_style.onnx",
    "onnx/text_encoder.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
    "voices/libri_female_1088.json",
    "voices/libri_female_6147.json",
    "voices/libri_male_6209.json",
    "voices/libri_male_8088.json",
)

BLUE_YI = RuntimeManifest(
    id="blue_yi",
    name="blue-yi (Yiddish)",
    version="bluetts-2.5",
    # manifest.json's own file sizes sum to 280.1 MB; the five voice JSONs add
    # 1.4 MB, so a full snapshot_download of this repo is 281.5 MB on disk.
    size="~281 MB",
    description=(
        "The default runtime, and a Yiddish-trained model rather than a "
        "multilingual one borrowed for Yiddish: its manifest declares `yi` "
        "alone, its latent statistics come from stats_yiddish.pt, and it was "
        "trained on IPA from this same phonikud-yi engine (step 817,000). "
        "44.1 kHz output, four saved voices, and a character vocab that covers "
        "the Yiddish closed inventory outright — ʦ, ʧ, ʤ, ɡ and ŋ are single "
        "embeddings, so every phone reaches the model unfolded and none is "
        "dropped (a few punctuation characters the engine passes through are "
        "outside the vocab and are removed silently; they carry no sound). "
        "Two caveats: the four saved voices are LibriTTS-R readers of English, "
        "so speaker identity carries a foreign colour even though the phonology "
        "is Yiddish; and the duration predictor emits one total length with no "
        "monotonic alignment, so a sentence whose predicted total runs short "
        "can swallow a word."
    ),
    # Only a hint for a manual local install: the adapter loads the bundle from
    # the huggingface_hub cache (or BLUE25_MODEL_DIR), never from the repo root.
    directory="blue-yi",
    install_kind=InstallKind.FILES,
    files=BLUE_YI_FILES,
    required_files=BLUE_YI_REQUIRED_FILES,
    capabilities=RuntimeCapabilities(
        yiddish=True,
        streaming=True,
        # NOT a mistake and not a Blue limitation in general: BlueTTS can clone
        # from a reference wav, but doing so needs the autoencoder *encoder* to
        # turn that wav into z_ref, and export_onnx.py exported the decoder
        # only. This bundle therefore ships five frozen style vectors and no way
        # to make a sixth, so cloning is impossible here — voice_reference=False.
        voice_reference=False,
        fixed_voices=True,
    ),
    available=True,
    hf_repo_id=BLUE_REPO_ID,
)

# One runtime: blue-yi, the Yiddish-trained model.
REGISTRY: tuple[RuntimeManifest, ...] = (BLUE_YI,)


def runtimes() -> tuple[RuntimeManifest, ...]:
    """The whole catalog, in display order."""
    return REGISTRY


def runtime(runtime_id: str) -> RuntimeManifest | None:
    """The manifest with this id, or None. Ids are matched after trimming so a
    stray space from a form post or query string does not read as unknown."""
    wanted = runtime_id.strip()
    for manifest in REGISTRY:
        if manifest.id == wanted:
            return manifest
    return None


def is_installed(manifest: RuntimeManifest, root: Path) -> bool:
    """True when every required file is already on this machine.

    Repo-committed bundles are looked for under ``root/directory``;
    Hugging-Face-hosted ones (``hf_repo_id`` set) in the hub cache or the local
    override directory, never under ``root``. Says nothing about whether the
    runtime can be loaded — a manifest with ``available=False`` can be fully
    installed and still be unimplemented here.
    """
    if manifest.hf_repo_id:
        return _hf_files_cached(manifest)
    base = root / manifest.directory
    return all((base / name).is_file() for name in manifest.required_files)


def _hf_files_cached(manifest: RuntimeManifest) -> bool:
    """True when a Hugging-Face-hosted bundle is already on this machine.

    Checked without touching the network: either the local-override directory
    holds every required file, or every one of them is in the hub cache.
    """
    import os

    override = os.environ.get(BLUE_MODEL_DIR_ENV) if manifest.hf_repo_id == BLUE_REPO_ID else None
    if override:
        base = Path(override).expanduser()
        return all((base / name).is_file() for name in manifest.required_files)
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # huggingface_hub absent: nothing can be installed
        return False
    # Asked at the same revision the adapter downloads, so "installed" means
    # "the pinned commit is cached", not "some revision of this repo is".
    revision = BLUE_REVISION if manifest.hf_repo_id == BLUE_REPO_ID else None
    return all(
        isinstance(
            try_to_load_from_cache(manifest.hf_repo_id, name, revision=revision), str
        )
        for name in manifest.required_files
    )
