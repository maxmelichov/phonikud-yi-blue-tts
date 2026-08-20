/* Yiddish Phonikud TTS — demo front end.

   Reads the live server state (GET /v1/models/state, GET /v1/voices) to populate
   the voice picker, then posts a FormData to POST /generate and renders the
   nikud, the IPA, the per-token derivation table, and the audio.

   There is one runtime in this build, so nothing here picks one: no runtime
   picker, no POST /v1/models/load, and no per-runtime branching — every branch
   of that kind had exactly one reachable arm.

   Every response key touched here is one the API actually returns: the /generate
   payload (nikud, diacritics, phonemes, audio, tokens, unsupported) and TokenRowDTO
   (word, nikud, ipa, route, confidence, layer, reason). There is deliberately no
   handling for `ipa_primary` or `variants`: those are engine-internal names that no
   endpoint serves, and branching on them was dead code that could only ever be
   skipped. */

"use strict";

// Hebrew-script combining marks: U+05B0..U+05C7. Yiddish nikud lives entirely in
// this range, so its presence is a sufficient test for "this text is pointed".
const NIKUD_RE = /[ְ-ׇ]/;

const MODES = ["text", "nikud", "phonemes"];

const LOW_TOOLTIP =
  "LOW is the least certain tier: a defaulted ambiguous א/פ, a rescued loshn-koydesh " +
  "form, a corpus-mined collocation, or an inventory violation. It is the " +
  "human-verification queue, not an error.";

const ROUTE_TOOLTIP = {
  lexicon: "Read from a table: the native-verified gold lexicon, the abbreviation or multiword table, or a legacy merged-LK / high-frequency list.",
  rule: "No table knew the word; the Germanic or loshn-koydesh rule path derived it.",
  fallback:
    "Quarantined: the engine judged the output unfit to emit (a vowel-less loshn-koydesh skeleton, an unlexiconed unpointed LK word, or an out-of-inventory token such as a number or a URL). Only its punctuation reaches the spoken string.",
};

const el = (id) => document.getElementById(id);

/* ---------------- small helpers ---------------- */

function statusFor(mode) {
  return el(mode + "-status");
}

function setStatus(mode, text, kind) {
  const node = statusFor(mode);
  if (!node) return;
  node.textContent = text;
  node.className = "status-indicator" + (kind ? " status-" + kind : "");
  node.style.display = "flex";
}

function hideTransient() {
  document.querySelectorAll(".status-indicator").forEach((n) => {
    n.style.display = "none";
  });
  const alert = el("nikud-alert");
  if (alert) alert.style.display = "none";
}

function setBusy(busy) {
  MODES.forEach((mode) => {
    const button = el(mode + "-btn");
    if (button) button.disabled = busy;
  });
}

async function errorMessage(response) {
  // The API answers failures with {"error": {"code", "message"}}; show the message.
  try {
    const body = await response.json();
    if (body && body.error && body.error.message) return body.error.message;
    if (body && typeof body.detail === "string") return body.detail;
    if (body && body.message) return body.message;
  } catch (err) {
    /* not JSON — fall through to the status line */
  }
  return "Request failed (HTTP " + response.status + " " + response.statusText + ")";
}

async function getJSON(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

/* ---------------- voice picker ---------------- */

function fillVoiceSelect(voices, keep) {
  const select = el("voice-select");
  if (!select) return;
  const wanted = keep || select.value;
  if (!Array.isArray(voices) || !voices.length) {
    // Nothing to report (no runtime resident yet): keep whatever the server
    // rendered instead of replacing a real list with a placeholder.
    if (select.options.length) return;
  }
  select.textContent = "";
  const names = Array.isArray(voices) ? voices.filter(Boolean) : [];
  if (!names.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "runtime default";
    select.appendChild(option);
    select.disabled = true;
    return;
  }
  names.forEach((name) => {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  });
  select.disabled = names.length === 1;
  select.value = names.indexOf(wanted) >= 0 ? wanted : names[0];
}

async function refreshVoices(keepVoice) {
  // /v1/voices is 503 no_model until a runtime is resident, so ask the state
  // first and leave the list alone rather than invent one while warming.
  const state = await getJSON("/v1/models/state");
  let voices = [];
  if (state.loaded) {
    try {
      const payload = await getJSON("/v1/voices");
      voices = Array.isArray(payload.voices) ? payload.voices : [];
    } catch (err) {
      voices = [];
    }
  }
  fillVoiceSelect(voices, keepVoice);
}

async function initControls() {
  try {
    await refreshVoices();
  } catch (err) {
    /* /v1/models/state is cheap and never 503s; ignore a transient failure */
  }
}

/* ---------------- token table ---------------- */

function routeBadge(route) {
  const span = document.createElement("span");
  const value = (route || "").toLowerCase();
  span.className = "badge-route route-" + (value || "unknown");
  span.textContent = value || "—";
  const tip = ROUTE_TOOLTIP[value];
  if (tip) {
    span.title = tip;
    span.setAttribute("aria-label", value + " route. " + tip);
  }
  return span;
}

function confidenceCell(confidence) {
  const value = (confidence || "").toUpperCase();
  const span = document.createElement("span");
  span.textContent = value || "—";
  if (value === "HIGH") {
    span.className = "conf conf-high";
  } else if (value === "MED" || value === "MEDIUM") {
    span.className = "conf conf-med";
  } else if (value === "LOW") {
    span.className = "conf conf-low";
    // A tooltip, not a warning colour: LOW rows are review candidates, not failures.
    span.title = LOW_TOOLTIP;
    span.setAttribute("aria-label", "LOW confidence. " + LOW_TOOLTIP);
  } else {
    span.className = "conf";
  }
  return span;
}

function cell(row, className, text, dir) {
  const td = document.createElement("td");
  if (className) td.className = className;
  if (dir) td.dir = dir;
  td.textContent = text || "";
  row.appendChild(td);
  return td;
}

function renderTokens(tokens) {
  const tbody = el("token-tbody");
  const block = el("derivation-block");
  const count = el("token-count");
  if (!tbody || !block) return;

  tbody.textContent = "";
  const rows = Array.isArray(tokens) ? tokens : [];

  if (!rows.length) {
    block.style.display = "none";
    if (count) count.textContent = "";
    return;
  }

  rows.forEach((token) => {
    const tr = document.createElement("tr");

    cell(tr, "cell-word", token.word, "rtl");
    cell(tr, "cell-nikud", token.nikud, "rtl");
    cell(tr, "cell-ipa", token.ipa, "ltr");

    const routeTd = document.createElement("td");
    routeTd.appendChild(routeBadge(token.route));
    tr.appendChild(routeTd);

    const confTd = document.createElement("td");
    confTd.appendChild(confidenceCell(token.confidence));
    tr.appendChild(confTd);

    cell(tr, "cell-layer", token.layer);
    cell(tr, "cell-reason", token.reason);

    tbody.appendChild(tr);
  });

  block.style.display = "";
  if (count) {
    count.textContent = rows.length === 1 ? "1 token" : rows.length + " tokens";
  }
}

function renderUnsupported(unsupported) {
  const strip = el("unsupported-strip");
  const list = el("unsupported-list");
  if (!strip || !list) return;

  const phones = Array.isArray(unsupported) ? unsupported.filter(Boolean) : [];
  list.textContent = "";
  if (!phones.length) {
    strip.style.display = "none";
    return;
  }
  phones.forEach((phone) => {
    const chip = document.createElement("span");
    chip.className = "chip chip-warn mono";
    chip.textContent = phone;
    list.appendChild(chip);
  });
  strip.style.display = "flex";
}

/* ---------------- generate ---------------- */

function numberField(id, fallback) {
  const input = el(id);
  if (!input) return fallback;
  const value = parseFloat(input.value);
  return Number.isFinite(value) ? value : fallback;
}

async function generate(mode) {
  const audio = el("audio");
  const results = el("results");
  const alert = el("nikud-alert");

  if (alert) alert.style.display = "none";

  const textValue = mode === "phonemes" ? "" : (el(mode + "-input").value || "").trim();
  const phonemeValue = (el("phonemes-input").value || "").trim();

  if (mode === "phonemes") {
    if (!phonemeValue) {
      setStatus(mode, "Enter some IPA first.", "error");
      return;
    }
  } else if (!textValue) {
    setStatus(mode, "Enter some Yiddish text first.", "error");
    return;
  }

  // The Nikud tab is for pointing the text by hand; unpointed input there has
  // nothing for the pointing to change, so say so instead of generating twice.
  if (mode === "nikud" && !NIKUD_RE.test(textValue)) {
    if (alert) alert.style.display = "flex";
    const node = statusFor(mode);
    if (node) node.style.display = "none";
    return;
  }

  setStatus(mode, "Generating… the engine may need a moment on the first run.", "generating");
  setBusy(true);

  if (audio) {
    audio.pause();
    audio.removeAttribute("src");
  }

  const form = new FormData();
  form.append("mode", mode);
  form.append("text", textValue);
  form.append("phonemes", phonemeValue);
  // No `runtime` field: /generate always uses the resident runtime, which is the
  // only one this build has.
  form.append("voice", (el("voice-select") && el("voice-select").value) || "");
  form.append("speed", String(numberField("speed-input", 1.0)));
  form.append("n_steps", String(numberField("steps-input", 8)));
  form.append("cfg_scale", String(numberField("cfg-input", 4.0)));

  try {
    const response = await fetch("/generate", { method: "POST", body: form });
    if (!response.ok) {
      setStatus(mode, "✗ " + (await errorMessage(response)), "error");
      return;
    }

    const data = await response.json();

    // Resumable field filling: each stage's output lands in its own tab so the
    // user can edit it and re-run from there.
    const nikud = data.nikud || data.diacritics || "";
    if (nikud) el("nikud-input").value = nikud;
    if (data.phonemes) el("phonemes-input").value = data.phonemes;

    renderTokens(data.tokens);
    renderUnsupported(data.unsupported);

    if (results) results.style.display = "";

    if (data.audio && audio) {
      audio.src = data.audio;
      audio.load();
      audio.play().catch(() => {
        /* autoplay blocked — the controls are right there */
      });
    }

    setStatus(mode, "✓ Done", "ready");

    // The first generate on a cold Space is what loads the runtime, so the
    // voice list is re-read afterwards — before it, there was none to read.
    refreshVoices((el("voice-select") && el("voice-select").value) || "").catch(() => {});
  } catch (err) {
    setStatus(mode, "✗ " + (err && err.message ? err.message : "Network error"), "error");
    console.error(err);
  } finally {
    setBusy(false);
  }
}

/* ---------------- helper key insertion ---------------- */

function insertAtCursor(chars, target) {
  const input = el(target + "-input");
  if (!input) return;
  const start = input.selectionStart === null ? input.value.length : input.selectionStart;
  const end = input.selectionEnd === null ? input.value.length : input.selectionEnd;
  input.value = input.value.slice(0, start) + chars + input.value.slice(end);
  const caret = start + chars.length;
  input.focus();
  input.setSelectionRange(caret, caret);
}

/* ---------------- wiring ---------------- */

document.addEventListener("DOMContentLoaded", () => {
  MODES.forEach((mode) => {
    const button = el(mode + "-btn");
    if (button) {
      button.addEventListener("click", () => generate(mode));
    }
  });

  document.querySelectorAll(".helper-btn[data-insert]").forEach((button) => {
    button.addEventListener("click", () => {
      insertAtCursor(button.getAttribute("data-insert"), button.getAttribute("data-target"));
    });
  });

  // One-click sample: fills the Text tab and switches to it, since a sample is
  // unpointed and belongs to the full text -> nikud -> IPA pipeline.
  document.querySelectorAll(".sample-btn[data-sample]").forEach((button) => {
    button.addEventListener("click", () => {
      const input = el("text-input");
      if (!input) return;
      input.value = button.getAttribute("data-sample") || "";
      const tabButton = document.querySelector('[data-bs-toggle="tab"][data-mode="text"]');
      if (tabButton && window.bootstrap && window.bootstrap.Tab) {
        window.bootstrap.Tab.getOrCreateInstance(tabButton).show();
      }
      input.focus();
    });
  });

  // Stale status from one tab must not read as the state of another.
  document.querySelectorAll('[data-bs-toggle="tab"]').forEach((tab) => {
    tab.addEventListener("shown.bs.tab", hideTransient);
  });

  // Ctrl/Cmd+Enter generates from whichever tab is showing.
  document.addEventListener("keydown", (event) => {
    if (!(event.metaKey || event.ctrlKey) || event.key !== "Enter") return;
    const pane = document.querySelector(".tab-pane.active .btn-generate");
    if (pane && !pane.disabled) {
      event.preventDefault();
      pane.click();
    }
  });

  initControls();
});
