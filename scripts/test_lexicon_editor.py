#!/usr/bin/env python3
"""Lexicon editor + וי policy checks that do not need the 1.23 GB engine."""

from __future__ import annotations

import os
import sys
import tempfile
import unicodedata as ud
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("SPACE_ID", None)
os.environ.pop("HF_TOKEN", None)

from yiddish_phonikud import auth, engine, lexicon_edits, vav_yud  # noqa: E402

_FINAL = str.maketrans({"ך": "כ", "ם": "מ", "ן": "נ", "ף": "פ", "ץ": "צ"})


def _strip_points(text: str) -> str:
    return "".join(c for c in ud.normalize("NFD", text) if ud.category(c) != "Mn")


def _lexicon_key(word: str) -> str:
    return _strip_points(ud.normalize("NFC", word)).translate(_FINAL)


def _fake_g2p(*, gold: dict | None = None) -> SimpleNamespace:
    """Minimal engine stand-in for add/update without loading 1.23 GB."""
    gold_map = dict(gold or {})
    return SimpleNamespace(
        GOLD_LEXICON=gold_map,
        _WORD_LATIN={},
        _AUDIO_PE={},
        _AUDIO_VOWEL={},
        _AUDIO_ENDORSED={},
        _HOMOGRAPH_LK={},
        _SEFARIA_POINTED={},
        _MODEL_POINTED={},
        _ABBREVIATIONS={},
        _MULTIWORD={},
        lexicon_key=_lexicon_key,
        _strip_points=_strip_points,
        normalize_surface=lambda w: ud.normalize("NFC", w or ""),
    )


def check(cond: bool, name: str, detail: str = "") -> None:
    if cond:
        print("  OK  ", name, detail)
    else:
        print("  FAIL", name, detail)
        raise SystemExit(1)


def test_vav_yud() -> None:
    print("וי policy")
    check(vav_yud.has_vav_yud("אויך"), "אויך has וי")
    check(not vav_yud.has_vav_yud("ווי"), "ווי is not the וי digraph")
    check(vav_yud.has_vav_yud("וואוינט"), "וואוינט still has וי")
    check(vav_yud.classify_ipa("ɔjx") == "ɔj", "classify ɔjx")
    check(vav_yud.classify_ipa("oʊf") == "oʊ", "classify oʊf")
    check(vav_yud.rewrite_ipa("oʊx", "ɔj") == "ɔjx", "rewrite oʊx -> ɔjx")
    check(vav_yud.rewrite_latin("oukh", "ɔj") == "oykh", "latin oukh -> oykh")
    check("אויך" in vav_yud.OJ_GOLD_FIXES, "אויך is a named ɔj gold fix")
    check("אנגעהויבן" in vav_yud.FLAGGED_UNCERTAIN, "אנגעהויבן is flagged, not flipped")
    check("קרוין" in vav_yud.OJ_NAMED, "קרוין (Romance crown) is ɔj")
    check("קרויט" in vav_yud.OU_SEED, "קרויט is û-class seed")
    check("טויב" not in vav_yud.OU_SEED, "טויב is a homograph, not a single oʊ seed")
    check(vav_yud.OU_CSV_PATH is None, "oʊ CSV not in repo; do not invent stems")
    toyv = vav_yud.SENSE_HOMOGRAPHS["טויב"]
    check(toyv["ipa_primary"] == "tɔjb", "טויב primary is deaf tɔjb")
    check("toʊb" in toyv["variants"] and "tɔjb" in toyv["variants"],
          "טויב keeps both dove and deaf")
    check(len(vav_yud.OU_SEED) == 16, "OU_SEED still the named list",
          str(len(vav_yud.OU_SEED)))
    invented = set(vav_yud.OU_SEED) - {
        "דרויסן", "ארויס", "קרויט", "שטוינט", "הויז", "מויז", "מויל",
        "הויט", "בויך", "טויזנט", "לויט", "פויער", "אויף", "דערויף",
        "דעראויף", "אויס",
    }
    check(not invented, "oʊ seed is only the user list", str(invented) or "exact")


def test_validation() -> None:
    print("validation")
    check(lexicon_edits.validate_word("אויך") == "אויך", "hebrew token ok")
    try:
        lexicon_edits.validate_word("hello")
        check(False, "latin rejected")
    except ValueError:
        check(True, "latin rejected")
    try:
        lexicon_edits.validate_word("מיט א פאר יאר")
        check(False, "sentence rejected")
    except ValueError:
        check(True, "sentence rejected")
    check(lexicon_edits.validate_ipa("ɔjx") == "ɔjx", "ɔjx in inventory")
    try:
        lexicon_edits.validate_ipa("θojo")
        check(False, "garbage IPA rejected")
    except ValueError:
        check(True, "garbage IPA rejected")


class _FakeRequest:
    def __init__(self, username: str | None) -> None:
        self.session = {}
        if username:
            self.session["oauth_info"] = {
                "access_token": "x",
                "expires_at": 9_999_999_999,
                "userinfo": {"preferred_username": username, "name": username,
                             "picture": "", "profile": "", "isPro": False},
                "scope": "openid profile",
                "state": None,
            }


def test_auth_gate() -> None:
    print("ABE101 gate")
    check(auth.editor_username() == "ABE101", "default editor is ABE101")
    os.environ["LEXICON_EDITOR_USER"] = "ABE101"
    signed_out = auth.require_editor(_FakeRequest(None))
    check(getattr(signed_out, "status_code", None) == 401, "unsigned -> 401")
    other = auth.require_editor(_FakeRequest("someoneelse"))
    check(getattr(other, "status_code", None) == 403, "other user -> 403")
    ok = auth.require_editor(_FakeRequest("ABE101"))
    check(ok == "ABE101", "ABE101 passes")
    # Confirm the error envelope, not FastAPI {detail: ...}.
    body = signed_out.body.decode() if hasattr(signed_out, "body") else ""
    check("forbidden" in body and "error" in body, "401 uses error envelope")


def test_create_app() -> None:
    print("create_app")
    from app import create_app
    app = create_app()

    def walk(routes: object) -> set[str]:
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

    paths = walk(app.routes)
    for need in ("/v1/lexicon/me", "/v1/lexicon/lookup", "/v1/lexicon/update",
                 "/v1/lexicon/add", "/v1/lexicon/edits", "/v1/lexicon/browse",
                 "/", "/health"):
        check(need in paths, f"route {need}")


def _with_temp_edits(fn):
    prev_path = os.environ.get("LEXICON_EDITS_PATH")
    prev_edits = list(lexicon_edits._edits)
    prev_note = lexicon_edits._persist_note
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["LEXICON_EDITS_PATH"] = str(Path(tmp) / "edits.json")
        lexicon_edits._edits.clear()
        try:
            fn()
        finally:
            lexicon_edits._edits[:] = prev_edits
            lexicon_edits._persist_note = prev_note
            if prev_path is None:
                os.environ.pop("LEXICON_EDITS_PATH", None)
            else:
                os.environ["LEXICON_EDITS_PATH"] = prev_path


def test_add_entry() -> None:
    print("add_entry")

    def body() -> None:
        existing_key = _lexicon_key("אויך")
        g2p = _fake_g2p(gold={
            existing_key: {
                "word": "אויך",
                "ipa_primary": "ɔjx",
                "variants": ["ɔjx"],
                "layer": "G",
                "note": "gold",
                "freq": 1,
            },
        })

        try:
            lexicon_edits.add_entry(
                g2p, word="אויך", ipa_primary="ɔjx", username="ABE101",
            )
            check(False, "add existing refused")
        except ValueError as exc:
            msg = str(exc)
            check(
                "already exists" in msg and "use update" in msg,
                "add existing -> use update",
                msg,
            )

        row = lexicon_edits.add_entry(
            g2p, word="שול", ipa_primary="ʃul", username="ABE101",
        )
        check(row["word"] == "שול" and row["ipa_primary"] == "ʃul", "ABE101 add non-וי")
        check(row["op"] == "add" and row["was_existing"] is False, "add op recorded")
        check(
            g2p.GOLD_LEXICON[_lexicon_key("שול")]["ipa_primary"] == "ʃul",
            "add writes gold overlay",
        )

        # New וי defaults to ɔj: IPA oʊ is rewritten unless oʊ is explicit.
        oy = lexicon_edits.add_entry(
            g2p, word="פלוין", ipa_primary="ploʊn", username="ABE101",
        )
        check(oy["ipa_primary"] == "plɔjn", "new וי defaults to ɔj", oy["ipa_primary"])
        check(oy["vav_yud_class"] == "ɔj", "new וי class is ɔj")

        ou = lexicon_edits.add_entry(
            g2p, word="פלויז", ipa_primary="plɔjz", vav_yud_class="oʊ",
            username="ABE101",
        )
        check(ou["ipa_primary"] == "ploʊz", "explicit oʊ is kept", ou["ipa_primary"])
        check(ou["vav_yud_class"] == "oʊ", "explicit oʊ class")

        try:
            lexicon_edits.add_entry(
                g2p, word="", ipa_primary="ʃul", username="ABE101",
            )
            check(False, "empty surface rejected")
        except ValueError:
            check(True, "empty surface rejected")

        try:
            lexicon_edits.add_entry(
                g2p, word="hello", ipa_primary="ʃul", username="ABE101",
            )
            check(False, "latin surface rejected on add")
        except ValueError:
            check(True, "latin surface rejected on add")

        try:
            lexicon_edits.add_entry(
                g2p, word="מיט א פאר", ipa_primary="ʃul", username="ABE101",
            )
            check(False, "sentence rejected on add")
        except ValueError:
            check(True, "sentence rejected on add")

        try:
            lexicon_edits.add_entry(
                g2p, word="שולע", ipa_primary="θojo", username="ABE101",
            )
            check(False, "garbage IPA rejected on add")
        except ValueError:
            check(True, "garbage IPA rejected on add")

        seed_g2p = _fake_g2p()
        log = lexicon_edits.apply_seed(seed_g2p)
        toyv_gold = seed_g2p.GOLD_LEXICON.get(_lexicon_key("טויב"), {})
        check(toyv_gold.get("ipa_primary") == "tɔjb", "seed writes טויב primary")
        check(
            "toʊb" in (toyv_gold.get("variants") or []),
            "seed keeps טויב dove variant",
        )
        check(any("homograph טויב" in line for line in log), "seed logs טויב homograph")

    _with_temp_edits(body)


def test_http_add() -> None:
    print("HTTP add gate")
    from app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)
    payload = {"word": "שול", "ipa_primary": "ʃul"}

    unsigned = client.post("/v1/lexicon/add", json=payload)
    check(unsigned.status_code == 401, "unsigned add 401", str(unsigned.status_code))
    check("forbidden" in unsigned.text, "unsigned add error envelope")

    orig_user = auth.logged_in_username
    orig_loaded = engine.is_loaded
    orig_g2p = engine._g2p

    def body() -> None:
        fake = _fake_g2p()
        engine.is_loaded = lambda: True  # type: ignore[method-assign]
        engine._g2p = fake
        try:
            auth.logged_in_username = lambda _req: "someoneelse"  # type: ignore[assignment]
            other = client.post("/v1/lexicon/add", json=payload)
            check(other.status_code == 403, "non-ABE101 add 403", str(other.status_code))

            auth.logged_in_username = lambda _req: "ABE101"  # type: ignore[assignment]
            bad = client.post(
                "/v1/lexicon/add",
                json={"word": "hello", "ipa_primary": "ʃul"},
            )
            check(bad.status_code == 400, "ABE101 invalid add 400", str(bad.status_code))

            ok = client.post("/v1/lexicon/add", json=payload)
            check(ok.status_code == 200, "ABE101 valid add 200", str(ok.status_code))
            data = ok.json()
            check(data.get("word") == "שול" and data.get("ipa_primary") == "ʃul",
                  "ABE101 add body")
        finally:
            engine.is_loaded = orig_loaded  # type: ignore[method-assign]
            engine._g2p = orig_g2p

    try:
        _with_temp_edits(body)
    finally:
        auth.logged_in_username = orig_user  # type: ignore[assignment]


def test_browse() -> None:
    print("browse")
    fake = _fake_g2p(gold={
        _lexicon_key("הויז"): {"word": "הויז", "ipa_primary": "hoʊz",
                               "variants": ["hoʊz"], "layer": "G", "freq": 40},
        _lexicon_key("די"): {"word": "די", "ipa_primary": "də",
                             "variants": ["də", "di"], "layer": "G", "freq": 4153},
    })
    # A weaker tier holding a word that CONTAINS the gold one, to pin the ranking.
    fake._MODEL_POINTED[_lexicon_key("הויזבחור")] = {
        "ipa": "hɔjjzbˈuxur", "pointed": "הוֹיזְבָחוּר", "why": "default-embedded"}
    fake._SEFARIA_POINTED[_lexicon_key("כזית")] = {
        "ipa": "kazˈajis", "pointed": "כַּזַּיִת", "variants": []}

    page = lexicon_edits.browse(fake, limit=10)
    check(page["total"] == 4, "browse counts every table", str(page["total"]))
    check([r["word"] for r in page["rows"]][0] == "די",
          "browse orders by corpus frequency")
    check({r["source"] for r in page["rows"]} == {"gold", "model", "sefaria"},
          "browse attributes each row to its table")
    tiers = {r["source"]: r["tier"] for r in page["rows"]}
    check(tiers["gold"] < tiers["sefaria"] < tiers["model"],
          "browse tier follows the authority chain", str(tiers))

    # Typing a whole word must surface that word, not the longer compound that
    # merely contains it -- freq order alone gets this wrong.
    hit = lexicon_edits.browse(fake, q="הויז")
    check(hit["matched"] == 2, "browse search matches substrings", str(hit["matched"]))
    check(hit["rows"][0]["word"] == "הויז", "browse ranks the exact word first",
          hit["rows"][0]["word"])

    check(lexicon_edits.browse(fake, q="hoʊz")["rows"][0]["word"] == "הויז",
          "browse searches IPA as well as spelling")
    check(lexicon_edits.browse(fake, source="gold")["matched"] == 2,
          "browse filters by source")
    check(lexicon_edits.browse(fake, only="vav_yud")["matched"] == 2,
          "browse filters to וי words")
    check(lexicon_edits.browse(fake, only="variants")["matched"] == 1,
          "browse filters to multi-reading words")

    paged = lexicon_edits.browse(fake, limit=1, offset=1)
    check(len(paged["rows"]) == 1 and paged["offset"] == 1, "browse paginates")
    check(lexicon_edits.browse(fake, limit=9999)["limit"] == 200,
          "browse caps the page size")

    def body() -> None:
        lexicon_edits.save_edit(fake, word="הויז", ipa_primary="hoʊz",
                                username="ABE101")
        after = lexicon_edits.browse(fake, q="הויז")
        edited = {r["word"] for r in after["rows"] if r["edited"]}
        check(edited == {"הויז"}, "browse marks edited rows", str(edited))
        check(lexicon_edits.browse(fake, only="edited")["matched"] == 1,
              "browse filters to edited rows")

    _with_temp_edits(body)


def test_http_browse() -> None:
    print("HTTP browse gate")
    from app import create_app
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    unsigned = client.get("/v1/lexicon/browse")
    check(unsigned.status_code == 401, "unsigned browse 401", str(unsigned.status_code))

    orig_user = auth.logged_in_username
    orig_loaded = engine.is_loaded
    orig_g2p = engine._g2p
    try:
        engine.is_loaded = lambda: True  # type: ignore[method-assign]
        engine._g2p = _fake_g2p(gold={_lexicon_key("די"): {
            "word": "די", "ipa_primary": "də", "variants": ["də"],
            "layer": "G", "freq": 4153}})
        auth.logged_in_username = lambda _req: "someoneelse"  # type: ignore[assignment]
        check(client.get("/v1/lexicon/browse").status_code == 403,
              "non-ABE101 browse 403")
        auth.logged_in_username = lambda _req: "ABE101"  # type: ignore[assignment]
        ok = client.get("/v1/lexicon/browse?limit=5")
        check(ok.status_code == 200, "ABE101 browse 200", str(ok.status_code))
        payload = ok.json()
        check(payload["rows"][0]["word"] == "די", "ABE101 browse body")
        check(any(s["slug"] == "gold" for s in payload["sources"]),
              "browse ships the source list the picker needs")
    finally:
        auth.logged_in_username = orig_user  # type: ignore[assignment]
        engine.is_loaded = orig_loaded  # type: ignore[method-assign]
        engine._g2p = orig_g2p


def main() -> None:
    test_vav_yud()
    test_validation()
    test_auth_gate()
    test_create_app()
    test_add_entry()
    test_browse()
    test_http_add()
    test_http_browse()
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
