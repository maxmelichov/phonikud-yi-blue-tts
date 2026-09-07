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

/* Set by initLexiconEditor once /v1/lexicon/me says this visitor may edit.
   The results table is rendered by renderTokens, which runs long before the
   editor exists, so the two talk through these instead of a shared closure. */
/* The closed phone inventory, each letter carried by a word the reviewer knows.
   Every example is a real reading the engine already holds — gold for all but
   ʒ (audio-endorsed) and ʤ (the דזש digraph) — so nothing here is invented.
   ŋ has no lexicon word: the engine writes nɡ, and it stays for completeness. */
const IPA_PAD = [{"label":"Stress","phones":[{"p":"ˈ","w":"אבער","i":"ˈɔbər"}]},{"label":"Vowels","phones":[{"p":"a","w":"אז","i":"az"},{"p":"aː","w":"זיין","i":"zaːn"},{"p":"ɛ","w":"עס","i":"ɛs"},{"p":"ə","w":"די","i":"də"},{"p":"i","w":"איז","i":"iz"},{"p":"u","w":"וואס","i":"vus"},{"p":"ɔ","w":"אבער","i":"ˈɔbər"}]},{"label":"Diphthongs","phones":[{"p":"ej","w":"דעם","i":"dejm"},{"p":"aj","w":"זיי","i":"zaj"},{"p":"ɔj","w":"שוין","i":"ʃɔjn"},{"p":"oʊ","w":"אויף","i":"oʊf"}]},{"label":"Consonants","phones":[{"p":"b","w":"אבער","i":"ˈɔbər"},{"p":"d","w":"די","i":"də"},{"p":"f","w":"פון","i":"fin"},{"p":"ɡ","w":"געווען","i":"ɡəvˈejn"},{"p":"h","w":"האט","i":"hut"},{"p":"j","w":"יא","i":"ju"},{"p":"k","w":"קען","i":"kɛn"},{"p":"l","w":"זאל","i":"zul"},{"p":"m","w":"מיט","i":"mit"},{"p":"n","w":"און","i":"in"},{"p":"p","w":"עפעס","i":"ˈɛpəs"},{"p":"r","w":"פאר","i":"far"},{"p":"s","w":"וואס","i":"vus"},{"p":"t","w":"נישט","i":"niʃt"},{"p":"v","w":"וואס","i":"vus"},{"p":"z","w":"איז","i":"iz"},{"p":"x","w":"איך","i":"ix"},{"p":"ʃ","w":"נישט","i":"niʃt"},{"p":"ʒ","w":"השגחה","i":"haʒɡˈuxə"},{"p":"ʦ","w":"צו","i":"ʦi"},{"p":"ʧ","w":"מענטשן","i":"mɛnʧn"},{"p":"ʤ","w":"דזשאב","i":"ʤab"},{"p":"ŋ","w":"","i":""}]}];

/* The two-character phones, so Delete removes a letter and not half of one. */
const IPA_MULTI = ["aː", "ej", "aj", "ɔj", "oʊ"];

const lexBridge = { canEdit: false, editWord: null };

/* The last rows renderTokens drew. /v1/lexicon/me is a separate request, so an
   editor who generates before it answers would get a table with no Fix column
   and no way to get one short of generating again. Keeping the rows lets the
   verdict redraw them whenever it lands. */
let lastTokens = [];

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
  lastTokens = rows;

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

    if (lexBridge.canEdit) {
      const fixTd = document.createElement("td");
      fixTd.className = "tok-fix-col";
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-fix";
      btn.textContent = "Fix";
      btn.title = "Correct how " + token.word + " is pronounced";
      btn.setAttribute("aria-label", "Fix the pronunciation of " + token.word);
      btn.addEventListener("click", () => {
        if (lexBridge.editWord) lexBridge.editWord(token.word);
      });
      fixTd.appendChild(btn);
      tr.appendChild(fixTd);
    }

    tbody.appendChild(tr);
  });

  block.style.display = "";
  if (count) {
    count.textContent = rows.length === 1 ? "1 word" : rows.length + " words";
  }
  renderSummary(rows);
}

/* Plain-language verdict above the expert table: how much of the sentence the
   engine actually KNEW versus guessed. The table opens itself only when there
   is something worth looking at. */
function renderSummary(rows) {
  const summary = el("read-summary");
  const details = el("derivation-details");
  if (!summary) return;

  const low = rows.filter((t) => (t.confidence || "").toUpperCase() === "LOW");
  const skipped = rows.filter((t) => (t.route || "").toLowerCase() === "fallback");
  const known = rows.filter((t) => (t.route || "").toLowerCase() === "lexicon").length;

  summary.textContent = "";
  const bits = [];
  bits.push(chipText("chip-ok", known + " of " + rows.length + " words from the verified lexicon"));
  const guessed = rows.length - known - skipped.length;
  if (guessed > 0) {
    bits.push(chipText("chip-neutral", guessed + " read by rule"));
  }
  if (low.length) {
    bits.push(chipText("chip-warn", low.length + " uncertain — worth checking"));
  }
  if (skipped.length) {
    bits.push(chipText("chip-err", skipped.length + " not spoken (numbers, foreign words…)"));
  }
  bits.forEach((chip) => summary.appendChild(chip));

  if (details) details.open = low.length > 0 || skipped.length > 0;
  // Uncertain rows get a highlight so the open table points at them.
  const tbody = el("token-tbody");
  if (tbody) {
    Array.from(tbody.rows).forEach((tr, i) => {
      const token = rows[i] || {};
      const isLow = (token.confidence || "").toUpperCase() === "LOW";
      const isSkipped = (token.route || "").toLowerCase() === "fallback";
      tr.classList.toggle("row-low", isLow && !isSkipped);
      tr.classList.toggle("row-skipped", isSkipped);
    });
  }
}

function chipText(kind, text) {
  const span = document.createElement("span");
  span.className = "chip " + kind;
  span.textContent = text;
  return span;
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
      input.dispatchEvent(new Event("input"));
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

  // Live 0/4000 counter: the limit exists (dto.MAX_INPUT_CHARS) and hitting it
  // silently is confusing — show the number turning red instead.
  const textInput = el("text-input");
  const textCount = el("text-count");
  if (textInput && textCount) {
    const update = () => {
      const n = textInput.value.length;
      textCount.textContent = n + " / 4000";
      textCount.classList.toggle("char-count-max", n >= 4000);
    };
    textInput.addEventListener("input", update);
    update();
  }

  initControls();
  initLexiconEditor();
});

function initLexiconEditor() {
  const editor = el("lexicon-editor");
  if (!editor) return;

  const login = el("auth-login");
  const logout = el("auth-logout");
  const who = el("auth-who");
  const status = el("lex-status");
  const panel = el("lex-edit");

  // `editing` is null while adding, which is the only difference between the
  // two modes: /update takes whatever word is in the box, /add refuses a word
  // that already exists.
  const state = { editing: null };

  function setLexStatus(text, kind) {
    if (!status) return;
    status.textContent = text || "";
    status.className = "small lex-status" + (kind ? " status-" + kind : "");
  }

  fetch("/v1/lexicon/me", { credentials: "same-origin" })
    .then((r) => r.json())
    .then((me) => {
      if (me.username) {
        if (login) login.hidden = true;
        if (who) {
          who.hidden = false;
          who.textContent = me.can_edit
            ? "Signed in as " + me.username + " (editor)"
            : "Signed in as " + me.username;
        }
        if (logout) logout.hidden = false;
      }
      if (me.can_edit) {
        editor.hidden = false;
        // The results table gets its Fix column from the same verdict.
        lexBridge.canEdit = true;
        lexBridge.editWord = editFromResults;
        const fixHead = el("token-fix-head");
        if (fixHead) fixHead.hidden = false;
        // Redraw anything already on screen, so a table rendered before this
        // answer arrived still gets its Fix column.
        if (lastTokens.length) renderTokens(lastTokens);
      }
    })
    .catch(() => {
      /* public path: TTS still works without OAuth */
    });

  // ---- editing --------------------------------------------------------------

  function setValue(id, value) {
    const node = el(id);
    if (node) node.value = value || "";
  }

  function openEdit(row) {
    state.editing = row;
    if (!panel) return;
    panel.hidden = false;
    buildIpaPad();
    const wordField = el("lex-word-field");
    const vyField = el("lex-vy-field");
    if (wordField) wordField.hidden = true;
    if (vyField) vyField.hidden = !row.has_vav_yud;

    const mode = el("lex-edit-mode");
    const wordOut = el("lex-edit-word");
    const meta = el("lex-edit-meta");
    if (mode) mode.textContent = "Editing";
    if (wordOut) wordOut.textContent = row.word;
    if (meta) {
      const bits = [row.source_label];
      if (row.freq) bits.push(row.freq.toLocaleString() + " uses in the corpus");
      if (row.flagged) bits.push("וי class held uncertain — " + (row.flag_reason || ""));
      meta.textContent = bits.join(" · ");
    }

    setValue("lex-word", row.word);
    setValue("lex-ipa", row.ipa);
    setValue("lex-variants", (row.variants || []).filter((v) => v !== row.ipa).join(" | "));
    setValue("lex-note", row.note && row.note.length <= 240 ? row.note : "");
    setValue("lex-class", row.has_vav_yud && row.vav_yud_class ? row.vav_yud_class : "");
    setLexStatus(
      row.tier >= 4
        ? "This reading is the model's own guess. A verdict here overrides it everywhere."
        : "",
      row.tier >= 4 ? "warn" : ""
    );
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    const ipaInput = el("lex-ipa");
    if (ipaInput) ipaInput.focus();
  }

  /* The IPA pad. Every letter the engine accepts, one click each, with the
     word it comes from underneath — a reviewer should never have to find ʦ or ˈ
     on a keyboard. Built once, the first time the panel is opened. */
  function buildIpaPad() {
    const pad = el("ipa-pad");
    if (!pad || pad.dataset.built) return;
    IPA_PAD.forEach((group) => {
      const wrap = document.createElement("div");
      wrap.className = "ipa-group";
      const label = document.createElement("span");
      label.className = "ipa-group-label";
      label.textContent = group.label;
      wrap.appendChild(label);
      group.phones.forEach((entry) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "ipa-key";
        const sym = document.createElement("span");
        sym.className = "ipa-key-sym";
        sym.textContent = entry.p;
        btn.appendChild(sym);
        if (entry.w) {
          const ex = document.createElement("span");
          ex.className = "ipa-key-ex";
          ex.dir = "rtl";
          ex.textContent = entry.w;
          btn.appendChild(ex);
          btn.title = entry.p + " as in " + entry.w + " (" + entry.i + ")";
        } else {
          btn.title = entry.p;
        }
        btn.addEventListener("click", () => insertIpa(entry.p));
        wrap.appendChild(btn);
      });
      pad.appendChild(wrap);
    });

    const tools = document.createElement("div");
    tools.className = "ipa-group ipa-tools";
    [["⌫ Delete", () => backspaceIpa()], ["Clear", () => setValue("lex-ipa", "")]].forEach(
      ([text, fn]) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "ipa-key ipa-key-tool";
        btn.textContent = text;
        btn.addEventListener("click", () => {
          fn();
          const input = el("lex-ipa");
          if (input) input.focus();
        });
        tools.appendChild(btn);
      }
    );
    pad.appendChild(tools);
    pad.dataset.built = "1";
  }

  /* Insert at the caret, not at the end: a stress mark belongs in front of the
     vowel it marks, which is usually mid-word. */
  function insertIpa(chars) {
    const input = el("lex-ipa");
    if (!input) return;
    const start = input.selectionStart === null ? input.value.length : input.selectionStart;
    const end = input.selectionEnd === null ? input.value.length : input.selectionEnd;
    input.value = input.value.slice(0, start) + chars + input.value.slice(end);
    const caret = start + chars.length;
    input.focus();
    input.setSelectionRange(caret, caret);
  }

  /* One IPA letter, not one code unit: aː, ɔj and oʊ delete whole. */
  function backspaceIpa() {
    const input = el("lex-ipa");
    if (!input) return;
    const start = input.selectionStart === null ? input.value.length : input.selectionStart;
    const end = input.selectionEnd === null ? input.value.length : input.selectionEnd;
    if (start !== end) {
      input.value = input.value.slice(0, start) + input.value.slice(end);
      input.setSelectionRange(start, start);
      return;
    }
    if (!start) return;
    const two = input.value.slice(start - 2, start);
    const drop = IPA_MULTI.indexOf(two) >= 0 ? 2 : 1;
    input.value = input.value.slice(0, start - drop) + input.value.slice(start);
    input.setSelectionRange(start - drop, start - drop);
  }

  /* Fix, from the results table. The word came out of the sentence the user
     just heard, so we fetch its live reading and open the same panel the browse
     table opens. A word the engine only guessed at has no table row; that lands
     in the add form with the guess prefilled, which is the correct next step. */
  async function editFromResults(word) {
    setLexStatus("Loading " + word + "…");
    try {
      const response = await fetch(
        "/v1/lexicon/lookup?word=" + encodeURIComponent(word),
        { credentials: "same-origin" }
      );
      const body = await response.json();
      if (!response.ok) {
        setLexStatus((body.error && body.error.message) || "Could not load that word.", "err");
        return;
      }
      if (editor && editor.hidden) editor.hidden = false;
      if (body.found) {
        openEdit(body);
        setLexStatus("Editing " + body.word + " from your sentence.", "");
      } else {
        openNew();
        setValue("lex-word", body.word);
        setValue("lex-ipa", body.ipa_primary || body.ipa || "");
        const vyField = el("lex-vy-field");
        if (vyField) vyField.hidden = !body.has_vav_yud;
        const wordOut = el("lex-edit-word");
        if (wordOut) wordOut.textContent = body.word;
        const meta = el("lex-edit-meta");
        if (meta) {
          meta.textContent =
            "The engine has no verified reading for this word — it guessed. " +
            "Saving here makes your reading the answer everywhere.";
        }
        setLexStatus("Adding " + body.word + " from your sentence.", "warn");
        const ipaInput = el("lex-ipa");
        if (ipaInput) ipaInput.focus();
      }
    } catch (err) {
      setLexStatus(String(err), "err");
    }
  }

  function openNew() {
    state.editing = null;
    if (!panel) return;
    panel.hidden = false;
    buildIpaPad();
    const wordField = el("lex-word-field");
    const vyField = el("lex-vy-field");
    if (wordField) wordField.hidden = false;
    if (vyField) vyField.hidden = false;

    const mode = el("lex-edit-mode");
    const wordOut = el("lex-edit-word");
    const meta = el("lex-edit-meta");
    if (mode) mode.textContent = "New word";
    if (wordOut) wordOut.textContent = "";
    if (meta) meta.textContent = "Adding a word the engine does not know yet.";

    ["lex-word", "lex-ipa", "lex-variants", "lex-note", "lex-class"].forEach((id) => setValue(id, ""));
    setLexStatus("A new וי word reads ɔj unless you pick oʊ.", "");
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    const wordInput = el("lex-word");
    if (wordInput) wordInput.focus();
  }

  function closeEdit(keepStatus) {
    state.editing = null;
    if (panel) panel.hidden = true;
    if (!keepStatus) setLexStatus("");
  }

  [el("lex-cancel"), el("lex-cancel-2")].forEach((btn) => {
    if (btn) btn.addEventListener("click", closeEdit);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && panel && !panel.hidden) closeEdit();
  });

  function lexPayload() {
    const variantsRaw = (el("lex-variants") && el("lex-variants").value) || "";
    const klass = (el("lex-class") && el("lex-class").value) || "";
    return {
      word: (el("lex-word") && el("lex-word").value) || "",
      ipa_primary: (el("lex-ipa") && el("lex-ipa").value) || "",
      variants: variantsRaw
        ? variantsRaw.split(/[|,]/).map((s) => s.trim()).filter(Boolean)
        : null,
      note: (el("lex-note") && el("lex-note").value) || "",
      vav_yud_class: klass || null,
    };
  }

  const saveBtn = el("lex-save");
  if (saveBtn) {
    saveBtn.addEventListener("click", async () => {
      const adding = state.editing === null;
      const path = adding ? "/v1/lexicon/add" : "/v1/lexicon/update";
      setLexStatus(adding ? "Adding…" : "Saving…");
      saveBtn.disabled = true;
      try {
        const response = await fetch(path, {
          method: "POST",
          credentials: "same-origin",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(lexPayload()),
        });
        const body = await response.json();
        if (!response.ok) {
          setLexStatus((body.error && body.error.message) || "Could not save.", "err");
          return;
        }
        setLexStatus(
          (adding ? "Added " : "Saved ") + body.word + " → " + body.ipa_primary +
            ". The voice uses it from the next request on.",
          "ok"
        );
        closeEdit(true);
      } catch (err) {
        setLexStatus(String(err), "err");
      } finally {
        saveBtn.disabled = false;
      }
    });
  }
}
