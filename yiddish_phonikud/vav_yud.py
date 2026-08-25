"""וי is phonemic: default ɔj; oʊ only via a closed etymological exception set.

Spelling cannot decide. Weinreich vowel 54 (historical long *ū*, German *au* from
MHG *û*, English cognates often *ou/ow*) is oʊ. Everything else — historical *ō*
and *ou* (42/44) and all loshn-koydesh ḥolem — is ɔj.

The grapheme rule in the engine already emits Latin ``oy`` for וי, which
``latin_to_ipa`` maps to ɔj. oʊ is Latin ``ou``, and that spelling must come
from a lexicon / _WORD_LATIN exception, never from "Germanic וי → oʊ".

This module is the Space-side copy of that policy: the seed exception set (only
stems the user named), the named ɔj corrections, and the uncertain items that
must not be silently flipped. ABE101's editor extends the exception set; nobody
else can.

English cousin *ou/ow* → oʊ is a diagnostic gut-check, not an inference rule.
Do not look up English cognates at runtime.

The user mentioned a closed ~45-stem oʊ CSV. It is **not in this workspace**
(see ``OU_CSV_PATH``). Do not invent the rest of the list.
"""

from __future__ import annotations

import re
from typing import Literal

VavYudClass = Literal["oʊ", "ɔj"]

# True וי / ױ, not the ווי sequence (consonantal /v/ + yud).
_VAV_YUD = re.compile(r"(?<!ו)וי|ױ")

# Closed û-class CSV the user said they made (~45 stems). Empty until they
# attach the file; do not pad OU_SEED up to 45 from imagination.
OU_CSV_PATH: str | None = None

# û-class (vowel 54). Seeded ONLY from the explicit user list — not invented.
# IPA follows existing engine conventions (no stress on monosyllables).
# טויב is a sense-homograph (dove oʊ vs deaf ɔj), not a single-class seed.
OU_SEED: dict[str, dict[str, str]] = {
    "דרויסן": {"ipa": "droʊsn", "latin": "drousn"},
    "ארויס": {"ipa": "arˈoʊs", "latin": "arous"},
    "קרויט": {"ipa": "kroʊt", "latin": "krout"},
    "שטוינט": {"ipa": "ʃtoʊnt", "latin": "shtount"},
    "הויז": {"ipa": "hoʊz", "latin": "houz"},
    "מויז": {"ipa": "moʊz", "latin": "mouz"},
    "מויל": {"ipa": "moʊl", "latin": "moul"},
    "הויט": {"ipa": "hoʊt", "latin": "hout"},
    "בויך": {"ipa": "boʊx", "latin": "boukh"},
    "טויזנט": {"ipa": "toʊznt", "latin": "touznt"},
    "לויט": {"ipa": "loʊt", "latin": "lout"},
    "פויער": {"ipa": "pˈoʊər", "latin": "pouer"},
    "אויף": {"ipa": "oʊf", "latin": "ouf"},
    "דערויף": {"ipa": "dərˈoʊf", "latin": "derouf"},
    "דעראויף": {"ipa": "dərˈoʊf", "latin": "derouf"},
    "אויס": {"ipa": "oʊs", "latin": "ous"},
}

# Productive û-class prefixes. Already rewritten on the Latin string in the
# engine (_CLASS54_PREFIXES). Listed here so the editor can label them.
OU_PREFIXES: tuple[str, ...] = ("אויס", "ארויס", "ארויף", "אויף")

# Named ɔj-class words. Only applied when a table currently stores oʊ.
OJ_NAMED: dict[str, str] = {
    "לויז": "lɔjz",
    "שויס": "ʃɔjs",
    "קרוין": "krɔjn",
    "שוין": "ʃɔjn",
    "נויט": "nɔjt",
    "טויט": "tɔjt",
    "בלויז": "blɔjz",
    "וואוינט": "vɔjnt",
    "שטויסן": "ʃtɔjsn",
    "אנטלויפן": "antlɔjfn",
    "קויפן": "kɔjfn",
    "ברויט": "brɔjt",
    "בוים": "bɔjm",
    "אויג": "ɔjɡ",
    "אויך": "ɔjx",
    "לויפן": "lɔjfn",
    "גלויבן": "ɡlɔjbn",
    "רויט": "rɔjt",
    "גרויס": "ɡrɔjs",
    "גרויסע": "ɡrˈɔjsə",
    "בויגן": "bɔjɡn",
}

# Sense-ambiguous types: same spelling, two attested IPAs. No context model —
# the type-level primary is a known majority limitation; variants must not be
# dropped. Handled like חלה / מקדש / מדבר (primary + variants, not a collapse).
SENSE_HOMOGRAPHS: dict[str, dict] = {
    "טויב": {
        "ipa_primary": "tɔjb",
        "variants": ["tɔjb", "toʊb"],
        "pointed": "טוֹיב",
        "note": (
            "HOMOGRAPH: tɔjb=deaf (default וי ɔj); toʊb=dove/bird (û-class 54). "
            "Needs context; global majority is a limitation — keep both."
        ),
        "senses": {"deaf": "tɔjb", "dove": "toʊb"},
    },
}

# Heard as [ɔj] but not locked. Do not silently rewrite.
FLAGGED_UNCERTAIN: dict[str, str] = {
    "אנגעהויבן": (
        "Uncertain וי class ([ɔj] by ear). Do not silently flip to oʊ; "
        "ABE101 can set the class explicitly if a verdict lands."
    ),
}

# Gold / Latin entries that were on the old v3 oʊ-list and are now ɔj.
OJ_GOLD_FIXES: dict[str, dict] = {
    "אויך": {
        "ipa_primary": "ɔjx",
        "variants": ["ɔjx", "oʊx"],
        "latin": "oykh",
        "note": "וי *ō/ou (Weinreich 44), not û-class 54; default ɔj",
    },
}


def has_vav_yud(word: str) -> bool:
    """True when ``word`` contains the וי digraph (not ווי)."""
    return bool(_VAV_YUD.search(word or ""))


def classify_ipa(ipa: str) -> VavYudClass | None:
    """Which וי nucleus the IPA uses, if either."""
    has_ou = "oʊ" in (ipa or "")
    has_oj = "ɔj" in (ipa or "")
    if has_ou and not has_oj:
        return "oʊ"
    if has_oj and not has_ou:
        return "ɔj"
    return None


def rewrite_ipa(ipa: str, target: VavYudClass) -> str:
    """Swap oʊ ↔ ɔj. No-op when the string has neither nucleus."""
    if not ipa:
        return ipa
    if target == "oʊ":
        return ipa.replace("ɔj", "oʊ")
    return ipa.replace("oʊ", "ɔj")


def rewrite_latin(latin: str, target: VavYudClass) -> str:
    """Swap the engine's Latin ``ou`` / ``oy`` spellings to match ``target``."""
    if not latin:
        return latin
    if target == "oʊ":
        # oy → ou, but do not touch a ou that is already there.
        return latin.replace("oy", "ou")
    return latin.replace("ou", "oy")
