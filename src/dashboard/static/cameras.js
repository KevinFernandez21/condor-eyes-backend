"use strict";
// Vista «Multicámara»: maqueta de VMS. NO hay video: cada cámara es una escena
// sintética en <canvas> (determinista por cámara) con cajas dibujadas desde la
// metadata de detección. La cámara real necesita un endpoint de instantáneas /
// MJPEG del lado del pipeline (ningún frame pasa por el bus ni por la API).
const Cams = (() => {
  const { el, svg, dash, hhmm, rng, hash } = UI;
  const CLASS_COLOR = { person: "#facc15", vehicle: "#fb923c", motion: "#38bdf8" };
  const CLASS_NAME = { person: "Persona", vehicle: "Vehículo", motion: "Movimiento" };
  const STATE_NAME = { online: "En línea", degraded: "Degradada", offline: "Sin señal" };
  const REVIEW_MS = 6000;

  const st = {
    layout: 2, page: 0, focus: null, live: true, frozen: null, last: null,
    fState: "all", fZone: "all", fAlert: false,
    review: { on: false, timer: null, id: null, log: [], rand: rng(Date.now()) },
    tl: { pinned: null, hover: null, camera: "all", minutes: 60 },
  };
  const tiles = new Map();   // id -> { root, canvas, bg, cur, data, ... }
  let rafId = 0, onChange = () => {};

  // ---------- escenas sintéticas ----------
  function drawScene(ctx, w, h, kind, seed) {
    const r = rng(seed), hy = h * 0.42;
    const grad = (y0, y1, a, b) => { const g = ctx.createLinearGradient(0, y0, 0, y1); g.addColorStop(0, a); g.addColorStop(1, b); return g; };
    ctx.fillStyle = grad(0, hy, "#1a2230", "#2a3446"); ctx.fillRect(0, 0, w, hy);
    ctx.fillStyle = grad(hy, h, "#2b3140", "#14181f"); ctx.fillRect(0, hy, w, h - hy);
    const rect = (x, y, rw, rh, c) => { ctx.fillStyle = c; ctx.fillRect(x, y, rw, rh); };
    if (kind === "lobby") {
      rect(0, 0, w, hy, "#2a3243");
      rect(w * 0.42, hy * 0.25, w * 0.16, hy * 0.75, "#3b4a63");           // puerta
      rect(w * 0.43, hy * 0.28, w * 0.14, hy * 0.7, "#1d2635");
      for (let i = 0; i < 4; i++) rect(w * (0.1 + i * 0.22), hy * 0.1, w * 0.1, 6, "#e8eefc55"); // luminarias
      ctx.strokeStyle = "#ffffff12"; ctx.lineWidth = 1;
      for (let i = -6; i <= 6; i++) { ctx.beginPath(); ctx.moveTo(w / 2, hy); ctx.lineTo(w / 2 + i * w * 0.18, h); ctx.stroke(); }
      rect(w * 0.06, hy + h * 0.1, w * 0.18, h * 0.05, "#3a4152");          // banco
    } else if (kind === "parking") {
      rect(0, hy * 0.55, w, hy * 0.45, "#232a36");
      for (let i = 0; i < 9; i++) { ctx.fillStyle = "#2d4a3a"; ctx.beginPath(); ctx.arc(w * (i / 8 + r() * 0.04), hy * 0.5, 10 + r() * 14, 0, 7); ctx.fill(); }
      ctx.strokeStyle = "#ffffff30"; ctx.lineWidth = 2;
      for (let i = 0; i <= 7; i++) { const x = w * (0.06 + i * 0.13); ctx.beginPath(); ctx.moveTo(x, hy + 8); ctx.lineTo(x - w * 0.08 + i * w * 0.01, h); ctx.stroke(); }
      for (let i = 0; i < 3; i++) { const x = w * (0.1 + r() * 0.7), y = hy + h * (0.12 + r() * 0.2); rect(x, y, w * 0.12, h * 0.1, ["#3d4a63", "#5a3b3b", "#3b5a4a"][i]); rect(x + w * 0.02, y - h * 0.04, w * 0.08, h * 0.05, "#1a2130"); }
    } else if (kind === "perimeter") {
      rect(0, hy * 0.7, w * 0.3, hy * 0.3, "#202938"); rect(w * 0.65, hy * 0.6, w * 0.3, hy * 0.4, "#1e2634");
      ctx.strokeStyle = "#9aa6bb55"; ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 14) { ctx.beginPath(); ctx.moveTo(x, hy * 0.8); ctx.lineTo(x, hy + h * 0.06); ctx.stroke(); }
      for (let y = hy * 0.8; y < hy + h * 0.06; y += 12) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
      for (let x = 0; x < w; x += w / 6) rect(x, hy * 0.7, 4, hy * 0.45, "#6b7587");
      ctx.fillStyle = "#1b2a22"; ctx.fillRect(0, hy + h * 0.06, w, h);
    } else if (kind === "warehouse") {
      for (let i = 0; i < 4; i++) {
        const x = w * (0.04 + i * 0.24);
        rect(x, hy * 0.2, w * 0.2, hy * 0.85, "#273044");
        for (let s = 0; s < 4; s++) { rect(x, hy * 0.2 + s * hy * 0.21, w * 0.2, 3, "#8a6a3a"); for (let b = 0; b < 3; b++) rect(x + 6 + b * w * 0.06, hy * 0.2 + s * hy * 0.21 - hy * 0.12, w * 0.045, hy * 0.11, `hsl(${30 + r() * 20} 30% ${25 + r() * 15}%)`); }
      }
      ctx.strokeStyle = "#f0b42955"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(w * 0.5, hy); ctx.lineTo(w * 0.3, h); ctx.moveTo(w * 0.5, hy); ctx.lineTo(w * 0.7, h); ctx.stroke();
    } else {  // pasillo: fuga a un punto
      const vx = w * (0.45 + r() * 0.1);
      rect(vx - w * 0.07, hy - h * 0.12, w * 0.14, h * 0.3, "#0f141d");
      ctx.strokeStyle = "#ffffff1a"; ctx.lineWidth = 1.5;
      for (const [x, y] of [[0, 0], [w, 0], [0, h], [w, h], [0, hy], [w, hy]]) { ctx.beginPath(); ctx.moveTo(vx, hy); ctx.lineTo(x, y); ctx.stroke(); }
      for (let i = 1; i < 5; i++) { const k = i / 5; rect(vx - w * 0.07 * (1 + k * 4), hy - h * 0.12 * (1 + k * 3), 4, h * 0.3 * (1 + k * 3), "#ffffff18"); }
    }
    ctx.fillStyle = "#0004"; for (let y = 0; y < h; y += 3) ctx.fillRect(0, y, w, 1);  // líneas de barrido
  }

  function ensureBackground(t, w, h, cam) {
    const key = `${w}x${h}|${cam.scene_kind}|${cam.id}`;
    if (t.bgKey === key) return;
    t.bg = document.createElement("canvas"); t.bg.width = w; t.bg.height = h;
    drawScene(t.bg.getContext("2d"), w, h, cam.scene_kind, hash(cam.id));
    t.bgKey = key;
  }

  function drawFrame(t) {
    const cam = t.data, c = t.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(2, Math.round(c.clientWidth * dpr)), h = Math.max(2, Math.round(c.clientHeight * dpr));
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    ensureBackground(t, w, h, cam);
    const ctx = c.getContext("2d");
    ctx.drawImage(t.bg, 0, 0);
    // Cajas suavizadas hacia su destino (la metadata llega a ~1-2 Hz).
    const targets = new Map();
    cam.detections.forEach((d, i) => targets.set(d.track_id != null ? `t${d.track_id}` : `${d.cls}#${i}`, d));
    for (const key of [...t.cur.keys()]) if (!targets.has(key)) t.cur.delete(key);
    ctx.lineWidth = Math.max(2, h / 270); ctx.font = `600 ${Math.max(10, Math.round(h / 24))}px system-ui, sans-serif`;
    for (const [key, d] of targets) {
      const prev = t.cur.get(key) || d.box.slice();
      const box = prev.map((v, i) => v + (d.box[i] - v) * 0.22);
      t.cur.set(key, box);
      const [x0, y0, x1, y1] = [box[0] * w, box[1] * h, box[2] * w, box[3] * h];
      const color = CLASS_COLOR[d.cls] || "#fff";
      ctx.strokeStyle = color; ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      const label = `${CLASS_NAME[d.cls] || d.cls}${d.track_id != null ? " #" + d.track_id : ""}${d.conf != null ? " " + Math.round(d.conf * 100) + "%" : ""}`;
      const tw = ctx.measureText(label).width + 8, th = Math.max(14, Math.round(h / 20));
      const ly = y0 - th < 0 ? y0 : y0 - th;
      ctx.fillStyle = color; ctx.fillRect(x0, ly, tw, th);
      ctx.fillStyle = "#111"; ctx.textBaseline = "middle"; ctx.fillText(label, x0 + 4, ly + th / 2);
    }
    ctx.fillStyle = "#e7eaf0cc"; ctx.textBaseline = "top"; ctx.textAlign = "right";
    ctx.fillText(new Date().toLocaleTimeString("es", { hour12: false }), w - 8, 8); ctx.textAlign = "left";
  }
  function loop() {
    if (document.getElementById("view-cams").hidden) { rafId = 0; return; }
    for (const t of tiles.values()) if (t.root.isConnected && t.data) { try { drawFrame(t); } catch (e) { console.error(e); } }
    rafId = requestAnimationFrame(loop);
  }

  // ---------- filtros y rejilla ----------
  function chip(label, pressed, onclick) { return el("button", { class: "chip", type: "button", "aria-pressed": String(pressed), onclick }, label); }
  function recentAlerts(s) {
    return (s.alerts || []).filter((a) => ["high", "medium"].includes(a.severity) && a.evaluated_at && Date.now() - new Date(a.evaluated_at).getTime() < 300000);
  }
  function renderFilters(s, cams) {
    const set = (id, kids) => document.getElementById(id).replaceChildren(...kids);
    const states = [["all", "Todas"], ["online", "En línea"], ["degraded", "Degradada"], ["offline", "Sin señal"]];
    set("f-state", states.map(([k, l]) => chip(l, st.fState === k, () => { st.fState = k; st.page = 0; redraw(); })));
    const zones = [...new Set(cams.map((c) => c.scene || c.name))].sort();
    if (st.fZone !== "all" && !zones.includes(st.fZone)) st.fZone = "all";
    set("f-zone", [["all", "Todas"], ...zones.map((z) => [z, z])].map(([k, l]) => chip(l, st.fZone === k, () => { st.fZone = k; st.page = 0; redraw(); })));
    set("f-alert", [chip("Solo con alertas", st.fAlert, () => { st.fAlert = !st.fAlert; st.page = 0; redraw(); })]);
  }
  function visibleCams(s, cams) {
    const alertIds = new Set(recentAlerts(s).map((a) => a.stream_id));
    return cams.filter((c) => (st.fState === "all" || c.state === st.fState) &&
      (st.fZone === "all" || (c.scene || c.name) === st.fZone) && (!st.fAlert || alertIds.has(c.id)));
  }

  function buildTile(cam) {
    const t = { cur: new Map(), data: cam, bgKey: "" };
    t.canvas = el("canvas", { "aria-hidden": "true" });
    t.name = el("span", { class: "name" });
    t.badge = el("span", { class: "badge" });
    t.counts = el("span", { class: "counts" });
    t.veil = el("div", { class: "veil", hidden: true });
    t.root = el("div", { class: "tile", tabindex: "0", role: "button", onclick: () => toggleFocus(cam.id),
      onkeydown: (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); toggleFocus(cam.id); } } },
      t.canvas, t.badge, t.counts, t.veil, t.name);
    return t;
  }
  function updateTile(t, cam, s) {
    t.data = cam;
    t.name.textContent = cam.name;
    t.root.setAttribute("aria-label", `${cam.name}: ${STATE_NAME[cam.state]}. Pulsa para ${st.focus ? "volver" : "ampliar"}.`);
    t.badge.replaceChildren(el("i", { class: `dot ${cam.state}` }), st.live ? (cam.state === "online" ? "EN VIVO" : STATE_NAME[cam.state].toUpperCase()) : "PAUSA");
    t.counts.replaceChildren(...["person", "vehicle", "motion"].filter((k) => cam.counts[k] > 0)
      .map((k) => el("span", { class: `count ${k}`, title: CLASS_NAME[k] }, cam.counts[k])));
    t.root.classList.toggle("off", cam.state === "offline");
    t.root.classList.toggle("focus-mode", st.focus === cam.id);
    t.root.classList.toggle("review", st.review.on && st.review.id === cam.id);
    t.veil.hidden = cam.state === "online";
    t.veil.className = `veil ${cam.state}`;
    t.veil.replaceChildren(cam.state === "online" ? "" : STATE_NAME[cam.state], cam.message ? el("small", {}, cam.message) : null);
  }

  function renderGrid(s, data) {
    let cams = data.cameras;
    renderFilters(s, cams);
    cams = visibleCams(s, cams);
    const cols = st.focus ? 1 : st.layout, size = cols * cols;
    let shown;
    if (st.focus) shown = cams.filter((c) => c.id === st.focus);
    else {
      const pages = Math.max(1, Math.ceil(cams.length / size));
      st.page = Math.min(st.page, pages - 1);
      shown = cams.slice(st.page * size, st.page * size + size);
      const pager = document.getElementById("pager");
      pager.hidden = pages <= 1;
      pager.replaceChildren(el("button", { class: "chip", type: "button", disabled: st.page === 0, onclick: () => { st.page--; redraw(); } }, "‹ Anterior"),
        `Página ${st.page + 1} de ${pages}`,
        el("button", { class: "chip", type: "button", disabled: st.page >= pages - 1, onclick: () => { st.page++; redraw(); } }, "Siguiente ›"));
    }
    if (st.focus) document.getElementById("pager").hidden = true;
    if (st.focus && !shown.length) st.focus = null;
    const grid = document.getElementById("cam-grid");
    grid.style.setProperty("--cols", String(cols));
    for (const id of [...tiles.keys()]) if (!data.cameras.some((c) => c.id === id)) tiles.delete(id);
    const nodes = shown.map((cam) => {
      let t = tiles.get(cam.id);
      if (!t) { t = buildTile(cam); tiles.set(cam.id, t); }
      updateTile(t, cam, s);
      return t.root;
    });
    if (!shown.length) nodes.push(el("p", { class: "small" }, data.cameras.length ? "Ninguna cámara coincide con los filtros." : "Aún no hay cámaras reportadas por la API."));
    grid.replaceChildren(...nodes);
    if (!rafId) rafId = requestAnimationFrame(loop);
  }
  function toggleFocus(id) { st.focus = st.focus === id ? null : id; redraw(); }

  // ---------- revisión aleatoria ----------
  function reviewTick() {
    const s = st.last; if (!s) return;
    const ids = [...tiles.entries()].filter(([, t]) => t.root.isConnected).map(([id]) => id);
    if (!ids.length) return;
    const id = ids[Math.floor(st.review.rand() * ids.length)], t = tiles.get(id), cam = t.data;
    const alerted = recentAlerts(s).some((a) => a.stream_id === id && a.severity === "high");
    const count = cam.detections.length;
    st.review.id = id;
    st.review.log.unshift({ t: Date.now() / 1000, name: cam.name, count, result: cam.state !== "online" || alerted ? "alerta" : "ok", why: cam.state !== "online" ? STATE_NAME[cam.state] : alerted ? "alerta de fusión" : "" });
    st.review.log.length = Math.min(st.review.log.length, 40);
    renderReviews(); redraw();
  }
  function renderReviews() {
    const log = st.review.log;
    document.getElementById("review-empty").hidden = log.length > 0;
    document.getElementById("review-log").replaceChildren(...log.map((r) => el("li", { title: r.why },
      el("span", { class: "small" }, UI.clock(r.t)), el("span", {}, `${r.name} · ${r.count} det.`),
      el("span", { class: `res ${r.result}` }, r.result === "ok" ? "OK" : "Alerta"))));
  }
  function setReview(on) {
    st.review.on = on;
    clearInterval(st.review.timer);
    if (on) { st.review.rand = rng(Date.now()); st.review.timer = setInterval(reviewTick, REVIEW_MS); setTimeout(reviewTick, 300); }
    else st.review.id = null;
    redraw();
  }

  // ---------- línea de tiempo ----------
  const LEFT = 92, LANES = ["person", "vehicle", "motion"], LANE_Y = [22, 52, 82], LANE_H = 24;
  function renderTimeline(data) {
    const root = document.getElementById("tl"), tl = data.timeline;
    const sel = document.getElementById("tl-camera");
    const ids = ["all", ...data.cameras.map((c) => c.id)];
    if (sel.options.length !== ids.length || [...sel.options].some((o, i) => o.value !== ids[i])) {
      sel.replaceChildren(...ids.map((id) => el("option", { value: id, selected: id === st.tl.camera },
        id === "all" ? "Todas las cámaras" : (data.cameras.find((c) => c.id === id) || {}).name || id)));
    }
    if (!ids.includes(st.tl.camera)) st.tl.camera = "all";
    sel.value = st.tl.camera;
    const W = Math.max(720, Math.round(root.getBoundingClientRect().width)), H = 132;
    root.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const end = new Date(tl.to).getTime() / 1000, mins = st.tl.minutes, start = end - mins * 60;
    const ppm = (W - LEFT - 10) / mins, xOf = (t) => LEFT + ((t - start) / 60) * ppm;
    const rows = tl.series[st.tl.camera] || [];
    const out = [];
    LANES.forEach((k, i) => {
      out.push(svg("rect", { class: "lane-bg", x: LEFT, y: LANE_Y[i], width: W - LEFT - 10, height: LANE_H, rx: 4 }));
      out.push(svg("circle", { cx: 10, cy: LANE_Y[i] + LANE_H / 2, r: 4, fill: CLASS_COLOR[k] }));
      out.push(svg("text", { x: 20, y: LANE_Y[i] + LANE_H / 2 + 4, class: "tick-label", style: "fill: var(--text)" }, CLASS_NAME[k]));
    });
    const step = mins <= 15 ? 1 : mins <= 30 ? 5 : 10;
    for (let t = Math.ceil(start / 60 / step) * step * 60; t <= end; t += step * 60) {
      out.push(svg("line", { class: "tick", x1: xOf(t), x2: xOf(t), y1: 16, y2: 108 }));
      out.push(svg("text", { class: "tick-label", x: xOf(t), y: 11, "text-anchor": "middle" }, UI.hhmm(t)));
    }
    const maxv = Math.max(1, ...rows.flatMap((b) => LANES.map((k) => b[k])));
    for (const b of rows) {
      if (b.t < start || b.t > end) continue;
      LANES.forEach((k, i) => {
        if (!b[k]) return;
        const bar = svg("rect", { class: `${k}-bar`, x: xOf(b.t) + 0.5, y: LANE_Y[i] + 2, width: Math.max(ppm - 1, 2), height: LANE_H - 4, rx: 2, opacity: 0.45 + 0.55 * (b[k] / maxv) });
        bar.append(svg("title", {}, `${UI.hhmm(b.t)} · Persona ${b.person} · Vehículo ${b.vehicle} · Movimiento ${b.motion}`));
        out.push(bar);
      });
    }
    // cursor: fijado, bajo el ratón o "ahora"
    const cursorT = st.tl.pinned ?? st.tl.hover ?? end;
    const cx = Math.min(Math.max(xOf(cursorT), LEFT), W - 10);
    const bucket = rows.find((b) => b.t === Math.floor(cursorT / 60) * 60);
    const label = st.tl.pinned == null && st.tl.hover == null ? "Ahora" : UI.hhmm(cursorT);
    out.push(svg("line", { class: "cursor", x1: cx, x2: cx, y1: 16, y2: 112 }));
    out.push(svg("rect", { class: "cursor-chip", x: cx - 24, y: 112, width: 48, height: 16, rx: 8 }));
    out.push(svg("text", { class: "cursor-label", x: cx, y: 124, "text-anchor": "middle" }, label));
    root.replaceChildren(...out);
    root.dataset.start = String(start); root.dataset.ppm = String(ppm);
    const readout = document.getElementById("tl-readout");
    readout.textContent = st.tl.pinned == null && st.tl.hover == null ? (st.live ? "Ahora" : "Pausado")
      : `${label} · P ${bucket ? bucket.person : 0} · V ${bucket ? bucket.vehicle : 0} · M ${bucket ? bucket.motion : 0}`;
  }
  function tAt(ev) {
    const root = document.getElementById("tl"), r = root.getBoundingClientRect();
    const scale = (Number(root.viewBox.baseVal.width) || r.width) / r.width;
    const x = (ev.clientX - r.left) * scale, start = Number(root.dataset.start), ppm = Number(root.dataset.ppm);
    return x < LEFT ? null : start + ((x - LEFT) / ppm) * 60;
  }

  // ---------- ciclo ----------
  function redraw() { if (st.last) render(st.last, true); onChange(); }
  function render(s, local) {
    st.last = s;
    if (!st.live && !st.frozen) st.frozen = s;
    const data = st.live ? s : st.frozen;
    UI.guard("grid", () => renderGrid(s, data));
    UI.guard("timeline", () => renderTimeline(data));
    UI.guard("reviews", renderReviews);
  }
  function init(changeCb) {
    onChange = changeCb || (() => {});
    for (const b of document.querySelectorAll("#layout button")) b.addEventListener("click", () => {
      st.layout = Number(b.dataset.n); st.page = 0; st.focus = null;
      for (const o of document.querySelectorAll("#layout button")) o.setAttribute("aria-pressed", String(o === b));
      redraw();
    });
    document.getElementById("live-toggle").addEventListener("change", (ev) => {
      st.live = ev.target.checked; st.frozen = st.live ? null : st.last; if (st.live) st.tl.pinned = null; redraw();
    });
    document.getElementById("review-toggle").addEventListener("change", (ev) => setReview(ev.target.checked));
    document.getElementById("tl-camera").addEventListener("change", (ev) => { st.tl.camera = ev.target.value; redraw(); });
    document.getElementById("tl-window").addEventListener("change", (ev) => { st.tl.minutes = Number(ev.target.value); redraw(); });
    const root = document.getElementById("tl");
    root.addEventListener("mousemove", (ev) => { st.tl.hover = tAt(ev); if (st.last) renderTimeline(st.live ? st.last : st.frozen); });
    root.addEventListener("mouseleave", () => { st.tl.hover = null; if (st.last) renderTimeline(st.live ? st.last : st.frozen); });
    root.addEventListener("click", (ev) => {
      const t = tAt(ev); if (t == null) return;
      st.tl.pinned = t; st.live = false; st.frozen = st.frozen || st.last;
      document.getElementById("live-toggle").checked = false; redraw();
    });
    document.addEventListener("keydown", (ev) => { if (ev.key === "Escape" && st.focus) { st.focus = null; redraw(); } });
    window.addEventListener("resize", () => { if (st.last && !document.getElementById("view-cams").hidden) renderTimeline(st.live ? st.last : st.frozen); });
  }
  function activate() { if (!rafId) rafId = requestAnimationFrame(loop); }
  return { render, init, activate };
})();
