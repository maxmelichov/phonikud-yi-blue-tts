"""Runtime gold / וי-exception overlays, persisted for ABE101 edits.

The Space downloads a frozen engine snapshot. Native corrections and the וי
policy therefore live *on top* of that snapshot: we mutate ``GOLD_LEXICON``
(authority #1) and ``_WORD_LATIN`` (the Latin exception list that turns וי into
oʊ on the rule path) after ``engine.load()``.

Persistence, best durable option first:

1. Hugging Face dataset ``LEXICON_EDITS_DATASET`` (default
   ``notmax123/phonikud-yi-lexicon-edits``), written with the Space's ``HF_TOKEN``.
   Survives restarts and rebuilds.
2. ``LEXICON_EDITS_PATH`` or ``$HF_HOME/phonikud-yi-lexicon-edits.json`` as a
   local cache. This is wiped when the Space container is rebuilt unless a
   volume is mounted.

Seeded וי policy is applied first; ABE101's saved edits win over the seed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from . import phones, vav_yud

log = logging.getLogger(__name__)

EDITOR_USER_ENV = "LEXICON_EDITOR_USER"
EDITS_DATASET_ENV = "LEXICON_EDITS_DATASET"
EDITS_PATH_ENV = "LEXICON_EDITS_PATH"
DEFAULT_DATASET = "notmax123/phonikud-yi-lexicon-edits"
EDITS_FILENAME = "edits.jsonl"

_HEBREW_TOKEN = re.compile(r"^[\u0590-\u05FF'\"\-]+$")
_ALLOWED_LAYERS = frozenset("GLEANX")
_MAX_WORD = 40
_MAX_IPA = 80
_MAX_NOTE = 240
_TABLE_ATTRS: tuple[str, ...] = (
    "GOLD_LEXICON",
    "_AUDIO_PE",
    "_AUDIO_VOWEL",
    "_AUDIO_ENDORSED",
    "_HOMOGRAPH_LK",
    "_SEFARIA_POINTED",
    "_MODEL_POINTED",
    "_WORD_LATIN",
    "_ABBREVIATIONS",
    "_MULTIWORD",
)

_LOCK = threading.Lock()
_edits: list[dict[str, Any]] = []
_applied = False
_persist_note = "not yet applied"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _edits_path() -> Path:
    override = os.environ.get(EDITS_PATH_ENV)
    if override:
        return Path(override).expanduser()
    root = os.environ.get("HF_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "huggingface"
    return base / "phonikud-yi-lexicon-edits.json"


def _dataset_id() -> str:
    return (os.environ.get(EDITS_DATASET_ENV) or DEFAULT_DATASET).strip()


def persist_status() -> str:
    return _persist_note


def edits_snapshot() -> list[dict[str, Any]]:
    with _LOCK:
        return [dict(row) for row in _edits]


def _ipa_of(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("ipa_primary") or entry.get("ipa") or "")
    if isinstance(entry, tuple) and entry:
        return str(entry[0])
    if isinstance(entry, str):
        return entry
    return ""


def validate_word(word: str) -> str:
    text = (word or "").strip()
    if not text:
        raise ValueError("word is required")
    if len(text) > _MAX_WORD:
        raise ValueError(f"word is {len(text)} characters; the limit is {_MAX_WORD}")
    if " " in text or "\n" in text or "\t" in text:
        raise ValueError("word must be a single token, not a sentence")
    if not _HEBREW_TOKEN.match(text):
        raise ValueError("word must be Hebrew-script Yiddish (letters, geresh, maqaf)")
    return text


def validate_ipa(ipa: str) -> str:
    text = (ipa or "").strip()
    if not text:
        raise ValueError("ipa_primary is required")
    if len(text) > _MAX_IPA:
        raise ValueError(f"IPA is {len(text)} characters; the limit is {_MAX_IPA}")
    unknown = phones.validate(text)
    if unknown:
        raise ValueError(
            "IPA is outside the closed inventory: " + ", ".join(unknown)
        )
    return text


def validate_layer(layer: str) -> str:
    letter = (layer or "G").strip().upper()[:1] or "G"
    if letter not in _ALLOWED_LAYERS:
        raise ValueError(f"layer must be one of {sorted(_ALLOWED_LAYERS)}")
    return letter


def validate_variants(ipa_primary: str, variants: Any) -> list[str]:
    out: list[str] = [ipa_primary]
    if not variants:
        return out
    if isinstance(variants, str):
        pieces = [part.strip() for part in variants.replace(",", "|").split("|")]
    elif isinstance(variants, list):
        pieces = [str(part).strip() for part in variants]
    else:
        raise ValueError("variants must be a list of IPA strings")
    for piece in pieces:
        if not piece:
            continue
        validate_ipa(piece)
        if piece not in out:
            out.append(piece)
    return out


def _latin_for(g2p: ModuleType, word: str) -> str | None:
    bare = g2p._strip_points(g2p.normalize_surface(word))
    latin = getattr(g2p, "_WORD_LATIN", {}).get(bare)
    return str(latin) if latin else None


def find_in_engine(g2p: ModuleType, word: str) -> dict[str, Any] | None:
    """Locate ``word`` in any table the live engine holds."""
    key = g2p.lexicon_key(word)
    bare = g2p._strip_points(g2p.normalize_surface(word))
    for attr in _TABLE_ATTRS:
        table = getattr(g2p, attr, None)
        if not isinstance(table, dict):
            continue
        entry = table.get(key) or table.get(bare) or table.get(word)
        if entry is None:
            continue
        ipa = _ipa_of(entry)
        surface = word
        variants: list[str] = []
        layer = "G"
        note = ""
        if isinstance(entry, dict):
            surface = str(entry.get("word") or word)
            variants = [str(v) for v in (entry.get("variants") or []) if v]
            layer = str(entry.get("layer") or "G")
            note = str(entry.get("note") or "")
        return {
            "word": surface,
            "key": key,
            "table": attr,
            "ipa_primary": ipa,
            "variants": variants or ([ipa] if ipa else []),
            "layer": layer,
            "note": note,
            "latin": _latin_for(g2p, word),
            "vav_yud_class": vav_yud.classify_ipa(ipa),
            "has_vav_yud": vav_yud.has_vav_yud(surface) or vav_yud.has_vav_yud(word),
        }
    return None


def lookup(g2p: ModuleType, word: str) -> dict[str, Any]:
    surface = validate_word(word)
    found = find_in_engine(g2p, surface)
    flag = vav_yud.FLAGGED_UNCERTAIN.get(surface)
    payload = found or {
        "word": surface,
        "key": g2p.lexicon_key(surface),
        "table": None,
        "ipa_primary": "",
        "variants": [],
        "layer": "G",
        "note": "",
        "latin": _latin_for(g2p, surface),
        "vav_yud_class": None,
        "has_vav_yud": vav_yud.has_vav_yud(surface),
    }
    payload["found"] = found is not None
    payload["flagged"] = bool(flag)
    payload["flag_reason"] = flag or ""
    payload["existing"] = found is not None
    return payload


def _write_gold(g2p: ModuleType, word: str, ipa: str, variants: list[str],
                layer: str, note: str) -> None:
    key = g2p.lexicon_key(word)
    gold: dict[str, Any] = g2p.GOLD_LEXICON
    prev = gold.get(key) if isinstance(gold.get(key), dict) else {}
    gold[key] = {
        "word": word,
        "ipa_primary": ipa,
        "variants": variants,
        "layer": layer,
        "freq": int(prev.get("freq") or 0),
        "note": note,
    }


def _write_latin(g2p: ModuleType, word: str, latin: str | None,
                 target: vav_yud.VavYudClass | None) -> None:
    latin_map: dict[str, str] = g2p._WORD_LATIN
    bare = g2p._strip_points(g2p.normalize_surface(word))
    current = latin or latin_map.get(bare)
    if target and current:
        current = vav_yud.rewrite_latin(current, target)
    elif target == "oʊ" and word in vav_yud.OU_SEED:
        current = vav_yud.OU_SEED[word]["latin"]
    elif target == "ɔj" and current:
        current = vav_yud.rewrite_latin(current, "ɔj")
    if current:
        latin_map[bare] = current


def apply_seed(g2p: ModuleType) -> list[str]:
    """וי policy: default ɔj; oʊ exceptions; named ɔj fixes. Returns log lines."""
    lines: list[str] = []
    # 1. Named ɔj gold fixes (only when current primary is oʊ).
    for word, spec in vav_yud.OJ_GOLD_FIXES.items():
        found = find_in_engine(g2p, word)
        current = (found or {}).get("ipa_primary") or ""
        if "oʊ" not in current and found:
            lines.append(f"keep {word} {current} (already not oʊ)")
            # Still pin Latin so the rule path cannot resurrect oukh.
            _write_latin(g2p, word, spec.get("latin"), "ɔj")
            continue
        _write_gold(
            g2p, word, spec["ipa_primary"], list(spec["variants"]),
            "G", spec.get("note") or "",
        )
        _write_latin(g2p, word, spec.get("latin"), "ɔj")
        lines.append(f"fix {word} {current or '?'} -> {spec['ipa_primary']}")

    # 2. û-class stems: Latin exception (rule path). Gold only if already gold
    #    and currently wrong; new stems stay on _WORD_LATIN so we do not mint
    #    unverified gold rows.
    for word, spec in vav_yud.OU_SEED.items():
        found = find_in_engine(g2p, word)
        _write_latin(g2p, word, spec["latin"], "oʊ")
        if found and found.get("table") == "GOLD_LEXICON":
            ipa = found.get("ipa_primary") or ""
            if "ɔj" in ipa and "oʊ" not in ipa:
                # User named this stem as oʊ; gold currently has ɔj.
                variants = vav_yud.rewrite_ipa("|".join(found.get("variants") or [ipa]), "oʊ").split("|")
                _write_gold(
                    g2p, word, vav_yud.rewrite_ipa(ipa, "oʊ"),
                    [vav_yud.rewrite_ipa(v, "oʊ") for v in variants],
                    found.get("layer") or "G",
                    "וי û-class (Weinreich 54) seed",
                )
                lines.append(f"gold oʊ {word} {ipa} -> {spec['ipa']}")
            else:
                lines.append(f"gold keep oʊ {word} {ipa}")
        else:
            lines.append(f"latin oʊ {word} -> {spec['latin']}")

    # 3. Named ɔj words: only flip if some table currently stores oʊ.
    for word, want_ipa in vav_yud.OJ_NAMED.items():
        if word in vav_yud.OJ_GOLD_FIXES:
            continue
        found = find_in_engine(g2p, word)
        if not found:
            continue
        ipa = found.get("ipa_primary") or ""
        if "oʊ" not in ipa:
            continue
        new_ipa = vav_yud.rewrite_ipa(ipa, "ɔj") or want_ipa
        variants = [vav_yud.rewrite_ipa(v, "ɔj") for v in (found.get("variants") or [ipa])]
        _write_gold(
            g2p, word, new_ipa, variants or [new_ipa],
            found.get("layer") or "G",
            "וי default ɔj (named ɔj-class correction)",
        )
        _write_latin(g2p, word, None, "ɔj")
        lines.append(f"oj-class {word} {ipa} -> {new_ipa}")

    for word, reason in vav_yud.FLAGGED_UNCERTAIN.items():
        found = find_in_engine(g2p, word)
        ipa = (found or {}).get("ipa_primary") or ""
        lines.append(f"flag {word} {ipa or '(absent)'} — {reason}")

    # Sense-homographs: write both attested readings. Do not pin Latin ou/oy
    # (that would collapse the other sense) and do not invent a context model.
    for word, spec in vav_yud.SENSE_HOMOGRAPHS.items():
        ipa = spec["ipa_primary"]
        variants = list(spec.get("variants") or [ipa])
        if ipa not in variants:
            variants.insert(0, ipa)
        _write_gold(g2p, word, ipa, variants, "G", spec.get("note") or "")
        lines.append(f"homograph {word} {ipa} variants={variants}")
    return lines


def apply_edit_row(g2p: ModuleType, row: dict[str, Any]) -> None:
    word = validate_word(str(row.get("word") or ""))
    ipa = validate_ipa(str(row.get("ipa_primary") or ""))
    variants = validate_variants(ipa, row.get("variants"))
    layer = validate_layer(str(row.get("layer") or "G"))
    note = str(row.get("note") or "")[:_MAX_NOTE]
    target = row.get("vav_yud_class")
    if target in ("oʊ", "ɔj"):
        ipa = vav_yud.rewrite_ipa(ipa, target)
        variants = [vav_yud.rewrite_ipa(v, target) for v in variants]
        if ipa not in variants:
            variants.insert(0, ipa)
    _write_gold(g2p, word, ipa, variants, layer, note)
    latin = row.get("latin")
    _write_latin(g2p, word, str(latin) if latin else None,
                 target if target in ("oʊ", "ɔj") else vav_yud.classify_ipa(ipa))


def apply_to_engine(g2p: ModuleType) -> None:
    """Idempotent overlay: seed policy, then persisted ABE101 edits."""
    global _applied, _persist_note
    with _LOCK:
        seed_log = apply_seed(g2p)
        loaded, source = _load_persisted()
        _edits[:] = loaded
        for row in _edits:
            try:
                apply_edit_row(g2p, row)
            except Exception as exc:  # noqa: BLE001 - one bad row must not drop TTS
                log.warning("skipping persisted lexicon edit %r: %s", row.get("word"), exc)
        _applied = True
        _persist_note = source
        log.info(
            "lexicon overlay: %d seed notes, %d persisted edits (%s)",
            len(seed_log), len(_edits), source,
        )
        for line in seed_log:
            log.info("וי policy: %s", line)


def _load_persisted() -> tuple[list[dict[str, Any]], str]:
    rows, source = _load_dataset()
    if rows is not None:
        _edits_path().write_text(_dump_jsonl(rows), encoding="utf-8")
        return rows, source
    path = _edits_path()
    if path.is_file():
        return _parse_jsonl(path.read_text(encoding="utf-8")), f"local-file:{path}"
    return [], "empty (no dataset, no local file)"


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("word"):
            rows.append(row)
    return rows


def _dump_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _load_dataset() -> tuple[list[dict[str, Any]] | None, str]:
    repo = _dataset_id()
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=repo,
            filename=EDITS_FILENAME,
            repo_type="dataset",
            token=token,
        )
        rows = _parse_jsonl(Path(path).read_text(encoding="utf-8"))
        return rows, f"dataset:{repo}"
    except Exception as exc:  # noqa: BLE001 - missing dataset is expected at first
        log.info("lexicon edits dataset %s not loaded: %s", repo, exc)
        return None, f"dataset-miss:{repo}"


def _save_dataset(rows: list[dict[str, Any]]) -> str | None:
    repo = _dataset_id()
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        try:
            api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not ensure dataset %s: %s", repo, exc)
        tmp = _edits_path()
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(_dump_jsonl(rows), encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(tmp),
            path_in_repo=EDITS_FILENAME,
            repo_id=repo,
            repo_type="dataset",
            commit_message="lexicon edit from Space (ABE101)",
        )
        return f"dataset:{repo}"
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to persist lexicon edits to %s: %s", repo, exc)
        return None


def save_edit(g2p: ModuleType, *, word: str, ipa_primary: str,
              variants: Any = None, layer: str = "G", note: str = "",
              vav_yud_class: str | None = None, username: str,
              mode: str = "update") -> dict[str, Any]:
    """Validate, apply, persist. Caller must already have passed require_editor.

    ``mode="update"`` overwrites an existing type (and may still insert a new
    one — that is the Save-reading path). ``mode="add"`` refuses if the type
    is already in any engine table: no silent clobber; the client must use
    update. New וי types on add default to ɔj unless ``vav_yud_class`` is
    explicitly ``oʊ``.
    """
    if mode not in ("update", "add"):
        raise ValueError("mode must be 'update' or 'add'")
    surface = validate_word(word)
    ipa = validate_ipa(ipa_primary)
    vars_ = validate_variants(ipa, variants)
    layer_ = validate_layer(layer)
    note_ = (note or "")[:_MAX_NOTE]
    target = vav_yud_class if vav_yud_class in ("oʊ", "ɔj") else None
    existing = find_in_engine(g2p, surface)

    if mode == "add" and existing is not None:
        raise ValueError(
            f"{surface} already exists in {existing['table']} as "
            f"{existing.get('ipa_primary') or '(empty)'}; use update, not add"
        )

    if mode == "add" and target is None and vav_yud.has_vav_yud(surface):
        # Policy: new וי is ɔj unless ABE101 explicitly picks oʊ. Do not invent
        # etymology — oʊ is only the closed û-class exception they opt into.
        target = "ɔj"

    if target:
        if target == "oʊ" and not vav_yud.has_vav_yud(surface):
            raise ValueError("oʊ is only valid for a word that contains וי")
        if "oʊ" not in ipa and "ɔj" not in ipa:
            raise ValueError(
                "this IPA has no וי nucleus to reclassify; set ipa_primary explicitly"
            )
        ipa = vav_yud.rewrite_ipa(ipa, target)
        vars_ = [vav_yud.rewrite_ipa(v, target) for v in vars_]
        if ipa not in vars_:
            vars_.insert(0, ipa)

    if surface in vav_yud.FLAGGED_UNCERTAIN and target is None:
        # Saving a full IPA is an explicit human verdict; that is allowed.
        note_ = (note_ + " " if note_ else "") + "[was flagged uncertain]"

    if existing is None:
        # New entries must be a single attested-looking type, not free text.
        if not vav_yud.has_vav_yud(surface) and not ipa:
            raise ValueError("new entries need a closed-inventory IPA reading")
        note_ = note_ or "ABE101 new lexicon type"

    row = {
        "word": surface,
        "ipa_primary": ipa,
        "variants": vars_,
        "layer": layer_,
        "note": note_,
        "vav_yud_class": target or vav_yud.classify_ipa(ipa),
        "op": mode,
        "updated_by": username,
        "updated_at": _utc_now(),
    }
    with _LOCK:
        _edits[:] = [item for item in _edits if item.get("word") != surface]
        _edits.append(row)
        apply_edit_row(g2p, row)
        local = _edits_path()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(_dump_jsonl(_edits), encoding="utf-8")
        source = _save_dataset(_edits) or f"local-file:{local}"
        global _persist_note
        _persist_note = source
    log.info("lexicon %s by %s: %s -> %s (%s)", mode, username, surface, ipa, _persist_note)
    return {**row, "persisted": _persist_note, "was_existing": existing is not None}


def add_entry(g2p: ModuleType, *, word: str, ipa_primary: str,
              variants: Any = None, layer: str = "G", note: str = "",
              vav_yud_class: str | None = None, username: str) -> dict[str, Any]:
    """Insert a type that is not already in gold / overlay. Never overwrites."""
    return save_edit(
        g2p,
        word=word,
        ipa_primary=ipa_primary,
        variants=variants,
        layer=layer,
        note=note,
        vav_yud_class=vav_yud_class,
        username=username,
        mode="add",
    )
