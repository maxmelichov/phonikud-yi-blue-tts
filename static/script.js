/* Yiddish Phonikud TTS — demo front end.

   Reads the live server state (GET /v1/models/sources, /v1/models/state, /v1/voices)
   to populate the runtime and voice pickers, then posts a FormData to POST /generate
   and renders the nikud, the IPA, the per-token derivation table, and the audio.

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

// Runtimes whose ids start with this take the flow-matching options. Kept as a
// prefix test rather than a hardcoded id so blue_yi_* variants keep working; the
// server is still the authority — it ignores options a runtime does not use.
const BLUE_PREFIX = "blue";

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
  const runtimeSelect = el("runtime-select");
  if (runtimeSelect) runtimeSelect.disabled = busy;
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

/* ---------------- runtime / voice pickers ---------------- */

// Last state read from GET /v1/models/state, so nothing on the page claims a
// runtime, voice list or sample rate the server did not report.
const serverState = { runtime: "", sampleRate: 0, voices: [], catalog: [] };

function selectedRuntime() {
  const select = el("runtime-select");
  return select && select.value ? select.value : serverState.runtime;
}

function isBlue(runtimeId) {
  return (runtimeId || "").startsWith(BLUE_PREFIX);
}

function updateCaveat(runtimeId) {
  const blue = el("caveat-blue");
  const piper = el("caveat-piper");
  if (!blue || !piper) return;
  const showBlue = isBlue(runtimeId);
  blue.style.display = showBlue ? "" : "none";
  piper.style.display = showBlue ? "none" : "";
}

function updateAdvancedNote(runtimeId) {
  const note = el("advanced-note");
  if (!note) return;
  note.textContent = isBlue(runtimeId)
    ? "The selected runtime (" + runtimeId + ") uses both values."
    : "The selected runtime (" +
      (runtimeId || "none") +
      ") ignores both values — they are sent anyway and dropped server-side.";
}

function updateRuntimeLine() {
  const line = el("runtime-line");
  const rate = el("panel-sample-rate");
  const voices = el("panel-voices");
  const summary = document.querySelector(".runtime-summary");

  if (line) {
    if (serverState.runtime) {
      const rateText = serverState.sampleRate ? " · " + serverState.sampleRate + " Hz" : "";
      const voiceText = serverState.voices.length
        ? " · " + serverState.voices.length + (serverState.voices.length === 1 ? " voice" : " voices")
        : "";
      line.textContent = "Loaded: " + serverState.runtime + rateText + voiceText;
    } else {
      line.textContent = "No runtime loaded yet — the first generate loads the default.";
    }
  }
  if (rate) rate.textContent = serverState.sampleRate ? serverState.sampleRate + " Hz" : "—";
  if (voices) voices.textContent = serverState.voices.length ? serverState.voices.join(", ") : "—";
  if (summary) {
    summary.textContent = serverState.runtime
      ? serverState.runtime + (serverState.sampleRate ? " · " + serverState.sampleRate + " Hz" : "")
      : "not loaded yet";
  }
}

function fillRuntimeSelect(catalog, current) {
  const select = el("runtime-select");
  if (!select) return;
  if (!catalog.length) {
    // The catalog fetch failed; the server-rendered options are still valid.
    return;
  }
  select.textContent = "";
  catalog.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.id;
    // `available: false` means the catalog declares it but this build cannot load
    // it; showing it disabled is honest, hiding it would misrepresent the catalog.
    option.textContent =
      (entry.name || entry.id) + (entry.available ? "" : " — not in this build");
    option.disabled = !entry.available;
    select.appendChild(option);
  });
  if (current && catalog.some((entry) => entry.id === current)) {
    select.value = current;
  } else {
    const firstAvailable = catalog.find((entry) => entry.available);
    if (firstAvailable) select.value = firstAvailable.id;
  }
}

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

async function refreshState(keepVoice) {
  const state = await getJSON("/v1/models/state");
  serverState.runtime = state.loaded ? state.runtime || "" : "";
  serverState.sampleRate = state.sample_rate || 0;
  serverState.voices = [];
  if (serverState.runtime) {
    try {
      const payload = await getJSON("/v1/voices");
      serverState.voices = Array.isArray(payload.voices) ? payload.voices : [];
    } catch (err) {
      // 503 no_model while warming: leave the list empty rather than invent one.
      serverState.voices = [];
    }
  }
  fillVoiceSelect(serverState.voices, keepVoice);
  updateRuntimeLine();
  return state;
}

async function initControls() {
  try {
    const sources = await getJSON("/v1/models/sources");
    serverState.catalog = Array.isArray(sources.runtimes) ? sources.runtimes : [];
  } catch (err) {
    serverState.catalog = [];
  }
  let loaded = "";
  try {
    const state = await refreshState();
    loaded = state.loaded ? state.runtime || "" : "";
  } catch (err) {
    /* /v1/models/state is cheap and never 503s; ignore a transient failure */
  }
  fillRuntimeSelect(serverState.catalog, loaded);
  const chosen = selectedRuntime();
  updateCaveat(chosen);
  updateAdvancedNote(chosen);
}

async function switchRuntime() {
  const runtimeId = selectedRuntime();
  updateCaveat(runtimeId);
  updateAdvancedNote(runtimeId);
  if (!runtimeId || runtimeId === serverState.runtime) return;

  const line = el("runtime-line");
  if (line) line.textContent = "Loading " + runtimeId + "…";
  setBusy(true);
  try {
    const response = await fetch("/v1/models/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ runtime: runtimeId }),
    });
    if (!response.ok) {
      const message = await errorMessage(response);
      if (line) line.textContent = "✗ " + message;
      // Snap the picker back to what is actually resident, so the page never
      // shows a runtime the server refused.
      const select = el("runtime-select");
      if (select && serverState.runtime) select.value = serverState.runtime;
      updateCaveat(selectedRuntime());
      updateAdvancedNote(selectedRuntime());
      return;
    }
    await refreshState();
    updateCaveat(serverState.runtime);
    updateAdvancedNote(serverState.runtime);
  } catch (err) {
    if (line) line.textContent = "✗ " + (err && err.message ? err.message : "Network error");
  } finally {
    setBusy(false);
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
  // No `runtime` field: /generate always uses the resident runtime, and the
  // picker switches it explicitly through POST /v1/models/load (see switchRuntime).
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

    // /generate reports which runtime, voice and sample rate actually served the
    // request; show those rather than what the pickers happened to say. The first
    // generate on a cold Space is also what loads the default runtime, so the
    // catalog state is re-read afterwards.
    if (data.runtime) {
      serverState.runtime = data.runtime;
      serverState.sampleRate = data.sample_rate || serverState.sampleRate;
      updateRuntimeLine();
      updateCaveat(data.runtime);
      updateAdvancedNote(data.runtime);
    }
    refreshState((el("voice-select") && el("voice-select").value) || "").catch(() => {});
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

  const runtimeSelect = el("runtime-select");
  if (runtimeSelect) runtimeSelect.addEventListener("change", () => switchRuntime());

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
