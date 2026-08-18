"""Runtime protocol, loader and loaded-state singleton.

Ported from mambotts-server/src/runtime/mod.rs: one narrow trait every
acoustic back end implements, plus a loader that turns a registry manifest
into a live instance. A runtime declared in the catalog but not implemented
in this build fails loudly the way `RuntimeParams::QwenHe` does when the
`qwen` feature is off -- it never pretends to be present. Both catalog
runtimes are implemented here: blue_yi (default, BlueTTS 2.5 from the
Hugging Face hub) and piper_yi (the committed lightweight fallback).
"""

from __future__ import annotations

import logging
import threading
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .. import registry

if TYPE_CHECKING:  # numpy is only needed for the annotation, not at import
    import numpy

logger = logging.getLogger(__name__)

# The Space repo root: model.onnx / model.config.json sit next to the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class RuntimeUnavailableReason(str, Enum):
    """Why a runtime could not be loaded, for the API's error codes."""

    NOT_IMPLEMENTED = "not_implemented"
    MISSING_FILES = "missing_files"


class RuntimeNotAvailable(RuntimeError):
    """A runtime exists in the catalog but cannot serve requests here."""

    def __init__(self, reason: RuntimeUnavailableReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@runtime_checkable
class Runtime(Protocol):
    """The whole contract an acoustic back end has to satisfy."""

    id: str
    sample_rate: int

    def voices(self) -> list[str]:
        """Named speakers, or ["default"] for a single-speaker checkpoint."""

    def vocab(self) -> set[str]:
        """Symbols the model accepts; phones.fold_to_vocab folds against it."""

    def synthesize(
        self, ipa: str, voice: str = "", speed: float = 1.0, **options
    ) -> "tuple[numpy.ndarray, list[str]]":
        """Renders one utterance: (float32 mono in [-1, 1], dropped units).

        ``dropped`` is every input unit the model could not render — folded
        away or encoded as PAD. It is *returned*, never stashed on the
        instance: the runtime is a process-wide singleton, so per-instance
        state would let concurrent requests read each other's report.

        ``speed`` > 1.0 is faster. ``**options`` is per-runtime and optional;
        blue_yi accepts ``n_steps``, ``cfg_scale``, ``pace_blend`` and
        ``seed``, piper_yi accepts none and ignores extras.
        """


# Construction can download or memory-map hundreds of megabytes, so it is
# serialized -- but the lock is released before any synthesize() call, which
# must stay concurrent.
_lock = threading.Lock()
_loaded: Runtime | None = None
# Every runtime instance this process has built, keyed by id. `_loaded` names
# which one is the process default; this dict is what makes a per-request
# `runtime` switch possible WITHOUT changing that default (see `instance`).
_instances: dict[str, Runtime] = {}


def _resolve_files(manifest: registry.RuntimeManifest, root: Path) -> tuple[dict[str, Path], list[str]]:
    """Locates every required file, tolerating both root and root/directory."""
    found: dict[str, Path] = {}
    missing: list[str] = []
    for name in manifest.required_files:
        for candidate in (root / manifest.directory / name, root / name):
            if candidate.is_file():
                found[name] = candidate
                break
        else:
            missing.append(name)
    return found, missing


def load_runtime(runtime_id: str, root: Path | None = None) -> Runtime:
    """Builds the runtime named by `runtime_id` from its registry manifest."""
    root = Path(root) if root is not None else _REPO_ROOT
    wanted = (runtime_id or registry.DEFAULT_RUNTIME_ID).strip() or registry.DEFAULT_RUNTIME_ID
    manifest = registry.runtime(wanted)
    if manifest is None:
        known = ", ".join(entry.id for entry in registry.runtimes())
        raise RuntimeNotAvailable(
            RuntimeUnavailableReason.NOT_IMPLEMENTED,
            f"unknown runtime `{wanted}`; this build has {known}",
        )
    if not manifest.available:
        # Same shape as MamboTTS's compiled-out branch: name the runtime, say
        # the build lacks it, and point at where it is declared.
        raise RuntimeNotAvailable(
            RuntimeUnavailableReason.NOT_IMPLEMENTED,
            f"this build does not bundle the {manifest.name} pipeline; "
            f"runtime `{manifest.id}` is declared in yiddish_phonikud/registry.py "
            f"(REGISTRY) with available=False and has no adapter here",
        )

    if manifest.id == "blue_yi":
        # hf_repo_id is set, so the bundle lives in the huggingface_hub cache
        # (or wherever BLUE25_MODEL_DIR points): there is nothing to resolve
        # under `root`, and the adapter raises its own missing-file errors.
        from .blue_yi import BlueYiddish  # imported late: pulls in onnxruntime

        try:
            return BlueYiddish(runtime_id=manifest.id, model_name=manifest.name)
        except FileNotFoundError as exc:
            raise RuntimeNotAvailable(
                RuntimeUnavailableReason.MISSING_FILES,
                f"runtime `{manifest.id}` could not load {manifest.hf_repo_id}: {exc}",
            ) from exc

    files, missing = _resolve_files(manifest, root)
    if missing:
        raise RuntimeNotAvailable(
            RuntimeUnavailableReason.MISSING_FILES,
            f"runtime `{manifest.id}` is missing {', '.join(missing)} "
            f"under {root}; download it from the registry entry's file list",
        )

    if manifest.id != "piper_yi":
        # available=True with no adapter would be the pretence qwen.rs avoids.
        raise RuntimeNotAvailable(
            RuntimeUnavailableReason.NOT_IMPLEMENTED,
            f"runtime `{manifest.id}` has no adapter in this build",
        )

    from .piper_yi import PiperYiddish  # imported late: pulls in onnxruntime

    return PiperYiddish(
        model_path=files["model.onnx"],
        config_path=files["model.config.json"],
        runtime_id=manifest.id,
        model_name=manifest.name,
    )


def loaded() -> Runtime | None:
    """The current runtime, or None if nothing has been loaded yet."""
    return _loaded


def _get_locked(runtime_id: str, root: Path | None) -> Runtime:
    """The instance for `runtime_id`, built and cached on first ask. Lock held."""
    existing = _instances.get(runtime_id)
    if existing is not None:
        return existing
    built = load_runtime(runtime_id, root)
    _instances[built.id] = built
    logger.info("built runtime %s at %sHz", built.id, built.sample_rate)
    return built


def instance(runtime_id: str, root: Path | None = None) -> Runtime:
    """The runtime named by `runtime_id`, WITHOUT changing what is loaded.

    This is what a per-request `runtime` field resolves through. Routing it
    through `load()` instead made one caller's choice process-global: a single
    `POST /v1/audio/speech {"runtime": "piper_yi"}` left `/v1/voices`,
    `/v1/models/state` and the default sample rate switched for everybody, and
    the next request naming a Blue voice failed with `unknown voice 'female' for
    runtime piper_yi`. Instances are cached per id, so alternating requests do
    not rebuild sessions either.
    """
    wanted = (runtime_id or registry.DEFAULT_RUNTIME_ID).strip() or registry.DEFAULT_RUNTIME_ID
    current = _loaded
    if current is not None and current.id == wanted:
        return current
    with _lock:
        return _get_locked(wanted, root)


def load_default() -> Runtime:
    """Loads DEFAULT_RUNTIME_ID once and reuses it afterwards."""
    global _loaded
    with _lock:
        if _loaded is None:
            _loaded = _get_locked(registry.DEFAULT_RUNTIME_ID, None)
            logger.info("loaded runtime %s at %sHz", _loaded.id, _loaded.sample_rate)
        return _loaded


def load(runtime_id: str, root: Path | None = None) -> Runtime:
    """Swaps the PROCESS-WIDE runtime for `runtime_id` (POST /v1/models/load).

    The only entry point that changes `loaded()`, and therefore the only one
    that changes what `/v1/voices`, `/v1/models/state` and a `runtime`-less
    request see.
    """
    global _loaded
    with _lock:
        current = _loaded
        if current is not None and current.id == runtime_id:
            return current
        _loaded = _get_locked(
            (runtime_id or registry.DEFAULT_RUNTIME_ID).strip()
            or registry.DEFAULT_RUNTIME_ID,
            root,
        )
        logger.info("loaded runtime %s at %sHz", _loaded.id, _loaded.sample_rate)
        return _loaded


def state() -> dict:
    """What /v1/models/state reports; safe to call before anything is loaded."""
    runtime = _loaded
    if runtime is None:
        return {
            "loaded": False,
            "runtime": "",
            "model": "",
            "path": "",
            "sample_rate": 0,
        }
    return {
        "loaded": True,
        "runtime": runtime.id,
        "model": getattr(runtime, "model_name", runtime.id),
        "path": str(getattr(runtime, "model_path", "")),
        "sample_rate": runtime.sample_rate,
    }


__all__ = [
    "Runtime",
    "RuntimeNotAvailable",
    "RuntimeUnavailableReason",
    "instance",
    "load",
    "load_default",
    "load_runtime",
    "loaded",
    "state",
]
