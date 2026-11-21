"use strict";
const NS = "http://www.w3.org/2000/svg";
const $ = (s) => document.querySelector(s);
const VIEW = "-17 -16 29 32";

let META = null;
let mode = "forward";
const state = {
  forward: { x: null, geom: null, result: null, stale: false },
  inverse: { x: null, geom: null, result: null, busy: false },
};
let anim = { frames: null, stable: null, t0: 0, raf: 0 };

// ------------------------------------------------------------------ helpers
async function api(path, body) {
  try {
    const r = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
    return await r.json();
  } catch (e) {
    return { ok: false, message: "The model server is not responding. Restart run_app.bat and try again." };
  }
}
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
const P = (pts) => pts.map(([x, z]) => `${x.toFixed(3)},${(-z).toFixed(3)}`).join(" ");
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const clear = (svg) => { while (svg.firstChild) svg.removeChild(svg.firstChild); svg.setAttribute("viewBox", VIEW); };
function arrowHead(g, x, z, dir, cls) {               // dir: +1 right, -1 left (horizontal arrows)
  el("polygon", { points: P([[x, z], [x - dir * 0.6, z + 0.28], [x - dir * 0.6, z - 0.28]]), class: cls }, g);
}

function drawWall(g, xW) {
  el("line", { x1: xW, y1: -13, x2: xW, y2: 13, class: "wall" }, g);
  for (let z = -12.5; z <= 12.6; z += 1.5) el("line", { x1: xW, y1: -z, x2: xW - 0.9, y2: -z + 0.9, class: "hatch" }, g);
}
function drawBodies(g, bodies, cls = "") {
  for (const b of bodies) el("polygon", { points: P(b.pts), class: (b.kind === "hinge" ? "hinge" : "frame") + " " + cls }, g);
}
function dimH(g, x1, x2, z, text, above = false) {
  const a = Math.min(x1, x2), b = Math.max(x1, x2);
  el("line", { x1: a, y1: -z, x2: b, y2: -z, class: "dim" }, g);
  arrowHead(g, a, z, -1, "loadhead dimhead"); arrowHead(g, b, z, 1, "loadhead dimhead");
  const t = el("text", { x: (a + b) / 2, y: -z + (above ? -0.45 : 1.35), "text-anchor": "middle", class: "dimtext" }, g);
  t.textContent = text;
}

// ------------------------------------------------------------------ left panel: initial shape with dimensions
function drawInitial(geom, x) {
  const svg = $("#svg-initial");
  clear(svg);
  if (!geom || !geom.bodies || !geom.bodies.length) return;
  const g = el("g", { class: geom.ok ? "" : "invalid" }, svg);
  const d = geom.derived, v = Object.fromEntries(META.vars.map((m, i) => [m.key, x[i]]));
  drawWall(g, d.x_W);
  drawBodies(g, geom.bodies);
  // applied displacement
  el("line", { x1: d.x_D + 0.15, y1: 0, x2: d.x_D + 3.2, y2: 0, class: "load" }, g);
  arrowHead(g, d.x_D + 3.6, 0, 1, "loadhead");
  const tu = el("text", { x: d.x_D + 0.6, y: -0.6, class: "label" }, g); tu.textContent = "u";
  // dimensions
  const dims = el("g", { class: "dims" }, g);
  el("line", { x1: 0, y1: 13.0, x2: 0, y2: 14.6, class: "ext" }, dims);
  el("line", { x1: -v.L, y1: 1.4, x2: -v.L, y2: 14.6, class: "ext" }, dims);
  dimH(dims, -v.L, 0, -14.2, `L ${fmt(v.L)}`);
  dimH(dims, d.x_i, -v.L, -4.6, `c ${fmt(v.c)}`);
  // t_f: thickness of the upper flange
  el("line", { x1: 0.25, y1: -d.Z_o, x2: 1.6, y2: -d.Z_o, class: "ext" }, dims);
  el("line", { x1: 0.25, y1: -d.z_F, x2: 1.6, y2: -d.z_F, class: "ext" }, dims);
  const tf = el("text", { x: 1.8, y: -(d.Z_o + d.z_F) / 2 + 0.35, class: "dimtext" }, dims); tf.textContent = `tf ${fmt(v.t_f)}`;
  const th = el("text", { x: 0.9, y: -(d.H + d.z_F) / 2 + 2.0, class: "label" }, dims); th.textContent = `tip hinge ${fmt(v.t_h)} × ${fmt(v.L_h)}`;
  const ta = el("text", { x: d.x_D + 0.6, y: -1.6 - v.L_a / 2, class: "label" }, dims); ta.textContent = `apex hinge ${fmt(v.t_a)} × ${fmt(v.L_a)}`;
  const tr = el("text", { x: d.x_W - 1.0, y: -14.4, class: "label" }, dims); tr.textContent = `fillet ratio ρ ${fmt(v.rho)}`;
}

// ------------------------------------------------------------------ middle panel: animation
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const easeOut = (t) => 1 - Math.pow(1 - t, 3);
const T_LOAD = 2600, T_HOLD = 500, T_REL = 1500, T_STABLE = 1700;
const T_ALL = T_LOAD + T_HOLD + T_REL + T_STABLE;

function frameAt(frames, u) {
  let i = 0;
  while (i < frames.length - 2 && frames[i + 1].u < u) i++;
  const a = frames[i], b = frames[i + 1];
  const w = Math.min(1, Math.max(0, (u - a.u) / (b.u - a.u || 1)));
  return a.bodies.map((body, k) => ({ kind: body.kind, pts: body.pts.map((p, j) => [p[0] + w * (b.bodies[k].pts[j][0] - p[0]), p[1] + w * (b.bodies[k].pts[j][1] - p[1])]) }));
}

function startAnimation(res) {
  cancelAnimationFrame(anim.raf);
  const svg = $("#svg-anim");
  clear(svg);
  if (!res || !res.frames) return;
  const frames = res.frames, u2 = frames[res.stable_index].u, xW = Math.min(...frames[0].bodies[0].pts.map((p) => p[0]));
  const ghost = el("g", {}, svg);
  for (const b of frames[0].bodies) el("polygon", { points: P(b.pts), class: "ghostline" }, ghost);
  drawWall(el("g", {}, svg), xW);
  const layer = el("g", {}, svg);
  const polys = frames[0].bodies.map((b) => el("polygon", { class: b.kind === "hinge" ? "hinge" : "frame" }, layer));
  const ruler = el("text", { x: -16.3, y: 15.4, class: "dimtext" }, svg);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  anim.t0 = performance.now();
  const step = (now) => {
    let t = reduced ? T_LOAD + T_HOLD + T_REL + 1 : (now - anim.t0) % T_ALL;
    let u, phase;
    if (t < T_LOAD) { u = 15 * ease(t / T_LOAD); phase = "loading"; }
    else if ((t -= T_LOAD) < T_HOLD) { u = 15; phase = "loaded"; }
    else if ((t -= T_HOLD) < T_REL) { u = 15 + (u2 - 15) * easeOut(t / T_REL); phase = "released"; }
    else { u = u2; phase = "second stable state"; }
    const f = frameAt(frames, u);
    f.forEach((b, k) => polys[k].setAttribute("points", P(b.pts)));
    ruler.textContent = `u = ${u.toFixed(2)} mm`;
    $("#phase").textContent = phase;
    if (!reduced) anim.raf = requestAnimationFrame(step);
  };
  anim.raf = requestAnimationFrame(step);
}

// ------------------------------------------------------------------ right panel: second stable state
function drawStable(res) {
  const svg = $("#svg-stable");
  clear(svg);
  if (!res || !res.frames) return;
  const f0 = res.frames[0], fs = res.frames[res.stable_index];
  const xW = Math.min(...f0.bodies[0].pts.map((p) => p[0]));
  const ghost = el("g", {}, svg);
  for (const b of f0.bodies) el("polygon", { points: P(b.pts), class: "ghostline" }, ghost);
  drawWall(el("g", {}, svg), xW);
  drawBodies(el("g", {}, svg), fs.bodies);
  // angle between the initial and the current upper-arm direction (pivot to pivot), drawn at the apex pivot
  const [A0, T0] = f0.probe, [A, T] = fs.probe;
  const a0 = Math.atan2(T0[1] - A0[1], T0[0] - A0[0]), a1 = Math.atan2(T[1] - A[1], T[0] - A[0]);
  const r = 4.2, g = el("g", {}, svg);
  el("line", { x1: A[0], y1: -A[1], x2: A[0] + (r + 1.2) * Math.cos(a0), y2: -(A[1] + (r + 1.2) * Math.sin(a0)), class: "ref" }, g);
  el("line", { x1: A[0], y1: -A[1], x2: A[0] + (r + 1.2) * Math.cos(a1), y2: -(A[1] + (r + 1.2) * Math.sin(a1)), class: "ref" }, g);
  const p0 = [A[0] + r * Math.cos(a0), A[1] + r * Math.sin(a0)], p1 = [A[0] + r * Math.cos(a1), A[1] + r * Math.sin(a1)];
  el("path", { d: `M ${p0[0]} ${-p0[1]} A ${r} ${r} 0 0 0 ${p1[0]} ${-p1[1]}`, class: "angle" }, g);
  const am = (a0 + a1) / 2;
  const lx = A[0] + (r + 0.8) * Math.cos(am), ly = A[1] + (r + 0.8) * Math.sin(am);
  const lab = el("text", { x: Math.min(lx, 11.4), y: -ly - 0.4, "text-anchor": lx > 3 ? "end" : "start", class: "callout blue" }, g);
  lab.textContent = `θ₂ ${fmt(res.theta2.mean, 1)}° ± ${fmt(res.theta2.sd, 1)}`;
  // displacement of the apex inner face, where the load is applied
  const xD0 = res.x_D, xD = res.x_D + fs.u;
  const gd = el("g", {}, svg);
  el("line", { x1: xD0, y1: 2.6, x2: xD0, y2: 6.2, class: "ext" }, gd);
  el("line", { x1: xD, y1: 2.6, x2: xD, y2: 6.2, class: "ext" }, gd);
  el("line", { x1: xD0, y1: 5.8, x2: xD, y2: 5.8, class: "load" }, gd);
  arrowHead(gd, xD, -5.8, 1, "loadhead");
  const tu = el("text", { x: (xD0 + xD) / 2, y: 7.6, "text-anchor": "middle", class: "callout red" }, gd);
  tu.textContent = `u₂ ${fmt(res.u2.mean, 2)} ± ${fmt(res.u2.sd, 2)} mm`;
}

// ------------------------------------------------------------------ result blocks
function verdictHTML(res) {
  const pPartial = res.partial.p_bistable, pMono = res.full.p_monostable_given_partial_bi;
  const tag = (ok, yes, no) => `<span class="tag ${ok ? "ok" : "bad"}">${ok ? yes : no}</span>`;
  return `<div class="block"><h3>Stability</h3>
    <div class="verdict">
      <span class="state">Partially wetted</span>${tag(res.partial.bistable, "Bistable", "Monostable")}
      <span class="p">P(bistable) ${fmt(pPartial, 3)}</span>
      <span class="state">Fully wetted</span>${res.partial.bistable ? tag(!res.full.bistable, "Monostable", "Bistable") : `<span class="tag">Monostable</span>`}
      <span class="p">${res.partial.bistable ? `P(monostable) ${fmt(pMono, 3)}` : "follows partial"}</span>
    </div>
    <div class="combo">${res.combination_ok
      ? `<span class="tag ok">Bistable combination met</span> · P = ${fmt(res.p_target_combination, 3)}`
      : `<span class="tag bad">Bistable combination not met</span> · needs bistable when partially wetted and monostable when fully wetted`}</div></div>`;
}
function valuesHTML(res) {
  if (!res.combination_ok) return `<div class="block"><h3>Second stable state</h3><p class="hint">Not shown: the combination is not met.</p></div>`;
  const e = res.dE;
  return `<div class="block"><h3>Second stable state</h3><div class="kv">
      <span class="k">Arm rotation θ₂</span><span class="v">${fmt(res.theta2.mean, 2)}° <span class="pm">± ${fmt(res.theta2.sd, 2)}</span></span>
      <span class="k">Displacement u₂</span><span class="v">${fmt(res.u2.mean, 3)} mm <span class="pm">± ${fmt(res.u2.sd, 3)}</span></span>
    </div></div>`;
}
function energyHTML(res) {
  if (!res.combination_ok) return `<div class="block"><h3>Energy barrier</h3><p class="hint">Not shown: the combination is not met.</p></div>`;
  const e = res.dE;
  return `<div class="block"><h3>Energy barrier (release from the second state)</h3><div class="kv">
      <span class="k">ΔE</span><span class="v">${fmt(e.mean, 4)} mJ</span>
      <span class="k">±1σ range</span><span class="v pm">${fmt(e.lo, 4)} – ${fmt(e.hi, 4)} mJ</span>
      <span class="k">log₁₀ ΔE</span><span class="v">${fmt(e.log10_mean, 3)} <span class="pm">± ${fmt(e.log10_sd, 3)}</span></span>
    </div></div>`;
}

function showResult(res) {
  const am = $("#anim-msg"), sm = $("#stable-msg"), out = $("#results");
  cancelAnimationFrame(anim.raf); clear($("#svg-anim")); clear($("#svg-stable")); $("#phase").textContent = "";
  am.classList.remove("warn"); sm.classList.remove("warn");
  if (!res) {
    am.hidden = sm.hidden = false;
    am.textContent = sm.textContent = mode === "forward" ? "Press Compute to run the models." : "Enter targets and press Find dimensions.";
    out.hidden = true;
    return;
  }
  if (!res.ok) {
    am.hidden = sm.hidden = false; am.classList.add("warn"); sm.classList.add("warn");
    am.textContent = sm.textContent = res.message || "Geometry is not valid.";
    out.hidden = true;
    return;
  }
  out.hidden = false;
  out.innerHTML = verdictHTML(res) + valuesHTML(res) + energyHTML(res);
  if (!res.combination_ok) {
    am.hidden = sm.hidden = false; am.classList.add("warn"); sm.classList.add("warn");
    am.textContent = sm.textContent = "Bistable combination not met: this shape is not bistable when partially wetted and monostable when fully wetted.";
    return;
  }
  am.hidden = sm.hidden = true;
  startAnimation(res);
  drawStable(res);
}

// ------------------------------------------------------------------ forward mode
function readDims() {
  return META.vars.map((v) => parseFloat($(`#dims input[name="${v.key}"]`).value));
}
let geomTimer = 0, geomSeq = 0;
const sameX = (a, b) => a && b && a.length === b.length && a.every((v, i) => v === b[i]);
async function updateGeometry() {
  const x = readDims(), err = $("#geom-error"), my = ++geomSeq;
  META.vars.forEach((v, i) => {
    const bad = !(x[i] >= v.min && x[i] <= v.max);
    $(`#dims input[name="${v.key}"]`).closest(".field").classList.toggle("invalid", bad);
  });
  if (x.some((v) => Number.isNaN(v))) { err.hidden = false; err.textContent = "Every dimension needs a number."; $("#compute").disabled = true; return; }
  const geom = await api("/api/geometry", { x });
  if (my !== geomSeq) return;                         // a newer edit is already being processed
  if (geom.ok === false && !geom.bodies) { err.hidden = false; err.textContent = geom.message || "Shape could not be built."; $("#compute").disabled = true; return; }
  state.forward.x = x; state.forward.geom = geom;
  if (mode === "forward") drawInitial(geom, x);
  err.hidden = geom.ok;
  if (!geom.ok) err.textContent = "Not a valid shape: " + geom.violations.join("; ") + ".";
  $("#compute").disabled = !geom.ok;
  if (state.forward.result && !state.forward.stale) {
    state.forward.stale = true;
    if (mode === "forward") {
      showResult(null);
      $("#anim-msg").textContent = $("#stable-msg").textContent = "Dimensions changed. Press Compute to update.";
    }
  }
}
function buildDims() {
  const form = $("#dims");
  form.innerHTML = META.vars.map((v) => `
    <label class="field"><span class="name"><b>${v.key.replace("_", "")}</b>${v.label}</span>
      <span class="box"><input name="${v.key}" type="number" step="${v.unit === "-" ? 0.01 : 0.01}" min="${v.min}" max="${v.max}" value="${v.default}">
      <span class="unit">${v.unit === "-" ? "" : v.unit}</span></span>
      <span class="range">${v.min} – ${v.max}</span></label>`).join("");
  form.addEventListener("input", () => { clearTimeout(geomTimer); geomTimer = setTimeout(updateGeometry, 60); });
  form.addEventListener("submit", (e) => { e.preventDefault(); compute(); });
}
async function compute() {
  const b = $("#compute");
  if (b.disabled) return;
  const x = readDims();
  b.disabled = true; b.textContent = "Computing…";
  const res = await api("/api/forward", { x });
  b.textContent = "Compute";
  b.disabled = !(state.forward.geom && state.forward.geom.ok);
  state.forward.result = res;
  state.forward.stale = !sameX(x, readDims());       // dimensions edited while the request was running
  if (mode === "forward") {
    showResult(state.forward.stale ? null : res);
    if (state.forward.stale) $("#anim-msg").textContent = $("#stable-msg").textContent = "Dimensions changed. Press Compute to update.";
  }
}

// ------------------------------------------------------------------ inverse mode
function summaryHTML(r) {
  if (!r.ok) return `<p class="status bad">${r.message}</p>`;
  const head = r.achievable
    ? `<p class="status ok">Design found</p>`
    : `<p class="status bad">${r.message}</p>${r.x ? `<p class="sub">The closest design is shown below for reference.</p>` : ""}`;
  const sub = `<p class="sub">${r.n_valid} of ${r.n_start} starts reached a valid shape · ${r.n_acceptable ?? 0} meet every target · ${r.time_s} s</p>`;
  if (!r.x) return head + sub;
  const rows = META.vars.map((v, i) => {
    const tol = r.tolerance[i];
    const t = tol.ok_at_best ? `−${fmt(tol.minus, 3)} / +${fmt(tol.plus, 3)}` : "–";
    return `<tr><td>${v.label} <span class="pm">${v.key.replace("_", "")}</span></td><td class="num">${fmt(r.x[i], 3)} ${v.unit === "-" ? "" : v.unit}</td><td class="tol">${t}</td></tr>`;
  }).join("");
  const tg = r.targets, gp = r.gp;
  const line = (k, label, unit, val, d) => tg[k] === null ? "" :
    `<span class="k">${label}</span><span class="v">${val} <span class="pm">target ${fmt(tg[k], d)} ${unit}</span></span>`;
  const outs = `<div class="kv">
      ${line("theta2", "θ₂", "°", `${fmt(gp.theta2.mean, 2)}° ± ${fmt(gp.theta2.sd, 2)}`, 1)}
      ${line("u2", "u₂", "mm", `${fmt(gp.u2.mean, 3)} ± ${fmt(gp.u2.sd, 3)} mm`, 2)}
      ${line("dE", "ΔE", "mJ", `${fmt(Math.pow(10, gp.logE.mean), 4)} mJ (×÷${fmt(Math.pow(10, gp.logE.sd), 2)})`, 4)}
      <span class="k">Worst target error</span><span class="v">${fmt(r.max_error_tau, 2)} τ</span>
      <span class="k">P(bistable → monostable)</span><span class="v">${fmt(r.p_target_combination, 3)}</span>
    </div>`;
  return head + sub + `<div class="grid">
    <div><table class="dims"><thead><tr><th>Dimension</th><th style="text-align:right">Value</th>
      <th style="text-align:right">Tolerance</th></tr></thead><tbody>${rows}</tbody></table>
      <p class="hint small">Tolerance: how far each dimension can move on its own while the predicted outputs stay within τ of the targets.</p></div>
    <div><h3 style="margin:0 0 8px;font-size:13px;color:var(--quiet)">Predicted by the inverse model (GP)</h3>${outs}
      <p class="hint small">The panels below re-evaluate this design with the forward models (deep ensemble + ROM-GP classifier).</p>
      ${r.forward && r.forward.ok && !r.forward.combination_ok
        ? `<p class="error">The forward models disagree with the inverse model: they do not predict bistable → monostable for this design. Treat it as uncertain.</p>` : ""}</div>
  </div>`;
}
async function design() {
  const f = new FormData($("#targets")), err = $("#target-error");
  const t = Object.fromEntries(["theta2", "u2", "dE"].map((k) => [k, f.get(k) === "" ? null : parseFloat(f.get(k))]));
  if (Object.values(t).every((v) => v === null)) { err.hidden = false; err.textContent = "Enter at least one target."; return; }
  if (t.dE !== null && !(t.dE > 0)) { err.hidden = false; err.textContent = "ΔE must be a positive number."; return; }
  err.hidden = true;
  const b = $("#design"), sum = $("#inverse-summary");
  b.disabled = true; b.textContent = "Searching…"; state.inverse.busy = true;
  if (mode === "inverse") renderInverse();
  sum.hidden = false;
  sum.innerHTML = `<div class="busy"><span class="spinner"></span>Searching from 20 starting points with the GP surrogate. This takes 30 s to about 3 min.</div>`;
  const r = await api("/api/inverse", t);
  state.inverse.result = r;
  if (r.x) { state.inverse.x = r.x; state.inverse.geom = await api("/api/geometry", { x: r.x }); }
  else { state.inverse.x = null; state.inverse.geom = null; }
  b.disabled = false; b.textContent = "Find dimensions"; state.inverse.busy = false;
  if (mode === "inverse") renderInverse();
}
function renderInverse() {
  const r = state.inverse.result, sum = $("#inverse-summary");
  if (state.inverse.busy) {                           // keep the panels empty while the search runs
    clear($("#svg-initial"));
    showResult(null);
    $("#anim-msg").textContent = $("#stable-msg").textContent = "Searching…";
    return;
  }
  sum.hidden = !r;
  if (r) sum.innerHTML = summaryHTML(r);
  if (state.inverse.geom) drawInitial(state.inverse.geom, state.inverse.x); else clear($("#svg-initial"));
  showResult(r && r.forward ? r.forward : null);
  if (r && !r.x) {
    $("#anim-msg").textContent = $("#stable-msg").textContent = "No design to show.";
  }
}

// ------------------------------------------------------------------ tabs
function setMode(m) {
  mode = m;
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
  $("#side-forward").hidden = m !== "forward";
  $("#side-inverse").hidden = m !== "inverse";
  $("#inverse-summary").hidden = true;
  if (m === "forward") {
    drawInitial(state.forward.geom, state.forward.x);
    showResult(state.forward.stale ? null : state.forward.result);
    if (state.forward.stale) $("#anim-msg").textContent = $("#stable-msg").textContent = "Dimensions changed. Press Compute to update.";
  } else {
    renderInverse();
    if (state.inverse.busy) { $("#inverse-summary").hidden = false; }
  }
}

async function init() {
  META = await api("/api/meta");
  buildDims();
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#compute").addEventListener("click", compute);
  $("#reset").addEventListener("click", () => { META.vars.forEach((v) => { $(`#dims input[name="${v.key}"]`).value = v.default; }); updateGeometry(); });
  $("#design").addEventListener("click", design);
  $("#targets").addEventListener("submit", (e) => { e.preventDefault(); design(); });
  ["#svg-initial", "#svg-anim", "#svg-stable"].forEach((s) => $(s).setAttribute("viewBox", VIEW));
  await updateGeometry();
  showResult(null);
}
init();
