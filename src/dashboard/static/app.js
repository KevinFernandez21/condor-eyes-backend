"use strict";
// Renderizador sin dependencias. Toda la lógica de vista vive en Python
// (src/dashboard/viewmodel.py); aquí solo se pinta. Nunca se inserta HTML con
// datos del bus: todo entra como nodos de texto (textContent).

const SVG_NS = "http://www.w3.org/2000/svg";
const POLL_MS = 1000;
const FETCH_TIMEOUT_MS = 4000;
const STATE_TEXT = {
  ok: "en marcha", degraded: "degradado", failed: "caído", stopped: "detenido", unknown: "sin datos",
};

const filters = { topics: new Set(), q: "", correlation: "" };
const openFeed = new Set();
let lastFeedSig = "";
let lastTopicsSig = "";

function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === false || v == null) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return node;
}
function svg(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}
const fmt = (v, d = 1) => (v == null || Number.isNaN(v) ? "sin datos" : Number(v).toFixed(d));
function clock(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleTimeString("es", { hour12: false });
}
function uptime(s) {
  if (s == null) return "sin datos";
  const m = Math.floor(s / 60);
  return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : m ? `${m} min ${Math.floor(s % 60)} s` : `${Math.floor(s)} s`;
}

// ---------- píldoras y banner ----------
function renderPills(s) {
  const box = document.getElementById("pills");
  const pill = (text, cls) => el("span", { class: `pill ${cls}` }, text);
  box.replaceChildren(
    pill(s.api.connected ? "API conectada" : "API sin conexión", s.api.connected ? "ok" : "bad"),
    pill(s.ws.connected ? "Tiempo real activo" : "Tiempo real: reconectando", s.ws.connected ? "ok" : "warn"),
    pill(`Hardware: ${s.health.hardware}`, ""),
  );
  const banner = document.getElementById("banner");
  const problems = [];
  if (!s.api.connected) problems.push(`API: ${s.api.error || "sin conexión"}.`);
  if (!s.ws.connected && s.ws.error) problems.push(`WebSocket: ${s.ws.error}.`);
  if (s.lag.dropped_total > 0) problems.push(`La API descartó ${s.lag.dropped_total} mensajes por lentitud de este cliente.`);
  banner.hidden = problems.length === 0;
  banner.textContent = problems.join(" ");
}

// ---------- salud ----------
function kv(value, label) { return el("div", { class: "kv" }, el("b", {}, value), el("span", {}, label)); }
function renderHealth(s) {
  const h = s.health;
  const statusText = { ok: "OK", degraded: "Degradado", offline: "Sin conexión" }[h.status] || h.status;
  const rows = h.streams.map((st) => el("tr", {},
    el("td", {}, st.stream_id), el("td", {}, st.state),
    el("td", { class: "num" }, fmt(st.fps)),
    el("td", { class: "num" }, `${fmt(st.latency_p50_ms)} / ${fmt(st.latency_p90_ms)}`),
    el("td", { class: "num" }, st.frames_dropped == null ? "sin datos" : st.frames_dropped)));
  const table = h.streams.length
    ? el("div", { class: "table-wrap streams" }, el("table", {},
        el("thead", {}, el("tr", {}, el("th", {}, "Stream"), el("th", {}, "Estado"),
          el("th", { class: "num" }, "FPS"), el("th", { class: "num" }, "Latencia p50/p90 (ms)"),
          el("th", { class: "num" }, "Frames perdidos"))),
        el("tbody", {}, rows)))
    : el("div", { class: "small streams" }, "Sin streams reportados todavía.");
  document.getElementById("health").replaceChildren(
    kv(statusText, "estado del sistema"),
    kv(h.agents_total == null ? "—" : `${h.agents_running}/${h.agents_total}`, "agentes en marcha"),
    kv(h.agents_failed == null ? "—" : h.agents_failed, "caídos"),
    kv(uptime(h.uptime_s), "activo desde la API"),
    kv(h.ws_clients == null ? "—" : h.ws_clients, "clientes WS"),
    kv(h.ws_dropped_total == null ? "—" : h.ws_dropped_total, "descartes WS"),
    kv("sin datos", `CPU / memoria (${h.hardware})`),
    table);
}

// ---------- grafo ----------
const NODE_W = 128, NODE_H = 54, GX = 170, GY = 100, PAD = 24;
function renderGraph(s) {
  const g = s.graph, root = document.getElementById("graph");
  const maxCol = Math.max(...g.nodes.map((n) => n.col)), maxRow = Math.max(...g.nodes.map((n) => n.row));
  const W = PAD * 2 + maxCol * GX + NODE_W, H = PAD * 2 + maxRow * GY + NODE_H;
  root.setAttribute("viewBox", `0 0 ${W} ${H}`);
  root.style.opacity = g.stale ? "0.55" : "1";
  const pos = Object.fromEntries(g.nodes.map((n) => [n.id, { x: PAD + n.col * GX, y: PAD + n.row * GY }]));
  const defs = svg("defs");
  const marker = svg("marker", { id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" });
  marker.append(svg("path", { d: "M0 0L10 5L0 10z", fill: "currentColor" }));
  defs.append(marker);
  const layer = svg("g", { style: "color: var(--accent)" });
  const pairCount = {};
  const labels = [];
  for (const e of g.edges) {
    const a = pos[e.source], b = pos[e.target];
    if (!a || !b) continue;
    const key = [e.source, e.target].sort().join("|");
    const idx = (pairCount[key] = (pairCount[key] || 0) + 1) - 1;
    const x1 = a.x + NODE_W / 2, y1 = a.y + NODE_H / 2, x2 = b.x + NODE_W / 2, y2 = b.y + NODE_H / 2;
    const dir = e.source < e.target ? 1 : -1;
    const bend = (idx + 1) * 16 * dir;
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const cx = mx - (dy / len) * bend, cy = my + (dx / len) * bend;
    const hot = e.kind === "data" && e.rate_per_s > 0;
    const path = svg("path", { d: `M${x1} ${y1} Q${cx} ${cy} ${x2} ${y2}`, class: `edge ${e.kind}${hot ? " hot" : ""}`, "marker-end": "url(#arrow)" });
    path.append(svg("title", {}, `${e.label} (${e.source} → ${e.target})`));
    layer.append(path);
    // Solo etiqueta corta (tasa y descartes) en aristas de datos con tráfico; el
    // detalle completo va en el tooltip y en la tabla de abajo.
    if (e.kind === "data" && (e.rate_per_s > 0 || e.drops > 0)) {
      const lx = (x1 + 2 * cx + x2) / 4, ly = (y1 + 2 * cy + y2) / 4;
      const short = `${e.rate_per_s == null ? "" : e.rate_per_s + "/s"}${e.drops ? ` ⚠${e.drops}` : ""}`;
      labels.push(svg("text", { x: lx, y: ly, "text-anchor": "middle", class: "edge-label" }, short));
    }
  }
  const nodes = svg("g");
  for (const n of g.nodes) {
    const p = pos[n.id];
    const grp = svg("g", { class: `node ${n.state}`, transform: `translate(${p.x} ${p.y})`, tabindex: "0" });
    grp.append(svg("title", {}, `${n.label}: ${STATE_TEXT[n.state]} — ${n.detail}`));
    grp.append(svg("rect", { width: NODE_W, height: NODE_H }));
    grp.append(svg("text", { x: 10, y: 22, "font-weight": "600" }, n.label + (n.instances > 1 ? ` ×${n.instances}` : "")));
    grp.append(svg("text", { x: 10, y: 40, class: "sub" }, STATE_TEXT[n.state] + (n.planned ? " · previsto" : "")));
    nodes.append(grp);
  }
  root.replaceChildren(defs, layer, nodes, ...labels);

  const tbody = document.querySelector("#edges tbody");
  tbody.replaceChildren(...g.edges.map((e) => el("tr", {},
    el("td", {}, e.topic), el("td", {}, `${e.source} → ${e.target}`),
    el("td", { class: "num" }, e.rate_per_s == null ? "sin datos" : e.rate_per_s),
    el("td", { class: "num" }, e.drops))));
  document.getElementById("legend").replaceChildren(
    ...Object.entries(STATE_TEXT).map(([k, t]) => el("span", {},
      el("i", { class: "dot", style: `border-color: var(--${{ ok: "ok", degraded: "warn", failed: "bad" }[k] || "off"})` }), t)),
    el("span", {}, "Línea punteada: control (salud, comandos, errores)"));
}

// ---------- alertas ----------
function renderAlerts(s) {
  const box = document.getElementById("alerts");
  if (!s.alerts.length) {
    box.replaceChildren(el("div", { class: "small" }, "Sin decisiones de fusión todavía."));
    return;
  }
  box.replaceChildren(...s.alerts.map((a) => el("article", { class: `alert ${a.severity}` },
    el("header", {}, el("span", { title: `valor original: ${a.outcome_raw || "—"}` }, a.outcome_label), el("span", { class: "small" }, `${clock(a.evaluated_at)} · ${a.decision_id || "—"}`)),
    el("div", {},
      el("span", { class: "tag op" }, a.requires_operator ? "Requiere operador" : "—"),
      el("span", { class: "tag" }, `confianza ${a.confidence_pct == null ? "sin datos" : a.confidence_pct + " %"}`),
      a.zone_id ? el("span", { class: "tag" }, `zona ${a.zone_id}`) : null,
      a.stream_id ? el("span", { class: "tag" }, a.stream_id) : null,
      a.person_id ? el("span", { class: "tag" }, `persona ${a.person_id}`) : null),
    a.reasons.length ? el("div", {}, a.reasons.map((r) => el("span", { class: "tag", title: r.code }, r.label))) : el("div", { class: "small" }, "Sin códigos de razón."),
    el("details", {}, el("summary", { class: "small" }, `Evidencia (${a.evidence.length})`),
      a.evidence.length
        ? el("ul", { class: "evidence" }, a.evidence.map((ev) => el("li", {},
            el("span", { class: "tag" }, ev.kind_label), el("span", { class: "tag" }, ev.role_label),
            el("span", { class: "small", title: JSON.stringify(ev) },
              `${ev.evidence_id}${ev.confidence == null ? "" : " · confianza " + Math.round(ev.confidence * 100) + " %"}${ev.detail ? " · " + ev.detail : ""}`))))
        : el("div", { class: "small" }, "Sin evidencia adjunta.")),
    a.correlation_id ? el("button", { class: "linkbtn", type: "button", onclick: () => setCorrelation(a.correlation_id) }, "ver mensajes relacionados") : null)));
}

// ---------- feed ----------
function setCorrelation(id) { filters.correlation = id; lastFeedSig = ""; tick(true); }
function renderTopicChips(topics) {
  const sig = topics.join("|");
  if (sig === lastTopicsSig) return;
  lastTopicsSig = sig;
  document.getElementById("topic-chips").replaceChildren(...topics.map((t) =>
    el("button", { class: "chip", type: "button", "aria-pressed": String(filters.topics.has(t)), onclick: (ev) => {
      if (filters.topics.has(t)) filters.topics.delete(t); else filters.topics.add(t);
      ev.currentTarget.setAttribute("aria-pressed", String(filters.topics.has(t)));
      lastFeedSig = ""; tick(true);
    } }, t)));
}
function renderFeed(s) {
  renderTopicChips(s.feed_topics);
  const clear = document.getElementById("corr-clear");
  clear.hidden = !filters.correlation;
  clear.textContent = `Correlación: ${filters.correlation} ✕`;
  const sig = s.feed.map((e) => e.event_id).join(",");
  if (sig === lastFeedSig) return;  // no destruir <details> abiertos si nada cambió
  lastFeedSig = sig;
  const list = document.getElementById("feed");
  if (!s.feed.length) { list.replaceChildren(el("li", { class: "small" }, "Sin mensajes con este filtro.")); return; }
  list.replaceChildren(...s.feed.map((e) => {
    const d = el("details", { open: openFeed.has(e.event_id) },
      el("summary", {}, el("span", { class: "t" }, clock(e.created_at)), el("span", { class: "topic" }, e.topic),
        el("span", { class: "small" }, e.source || ""), el("span", { class: "sum" }, e.summary)),
      el("pre", {}, JSON.stringify(e.payload, null, 2)),
      el("div", { class: "small" }, `event_id ${e.event_id || "—"}`,
        e.stream_id ? ` · stream ${e.stream_id}` : "",
        e.correlation_id ? [" · correlación ", el("button", { class: "linkbtn", type: "button", onclick: () => setCorrelation(e.correlation_id) }, e.correlation_id)] : ""));
    d.addEventListener("toggle", () => { if (d.open) openFeed.add(e.event_id); else openFeed.delete(e.event_id); });
    return el("li", {}, d);
  }));
}

// ---------- mapa de sitio ----------
function renderSite(s) {
  const box = document.getElementById("site");
  if (!s.site) {
    box.replaceChildren(el("div", { class: "small" }, "No hay mapa de sitio (configs/site_map.toml). Define CONDOR_SITE_MAP para cargar uno."));
    return;
  }
  const out = s.site.zones.map((z) => el("div", { class: `zone${z.tags.length ? " has-tags" : ""}` },
    el("strong", {}, z.name), el("span", { class: "small" }, ` (${z.id})`),
    el("div", { class: "small" }, z.receiver ? `Receptor ${z.receiver.id} · ${z.receiver.kind || "?"}` : "Sin receptor",
      z.camera ? ` · cámara ${z.camera}` : "", z.calibrated ? "" : " · sin calibrar"),
    z.tags.length
      ? z.tags.map((t) => el("div", {}, el("span", { class: "tag" }, `dentro: ${t.tag_ref}`),
          el("span", { class: "small" }, `confianza ${t.confidence_pct == null ? "sin datos" : t.confidence_pct + " %"} · visto ${clock(t.last_seen)}`)))
      : el("div", { class: "small" }, s.site.has_location_data ? "Nadie dentro." : "Presencia: sin datos")));
  if (s.site.outside.length) {
    out.push(el("div", { class: "zone" }, el("strong", {}, "Fuera / desconocido"),
      s.site.outside.map((t) => el("div", {}, el("span", { class: "tag" }, t.tag_ref),
        el("span", { class: "small" }, `${t.unknown_reason || "fuera"} · visto ${clock(t.last_seen)}`)))));
  }
  box.replaceChildren(...out);
}

function renderCamera(s) {
  document.getElementById("camera").replaceChildren(el("p", {}, s.camera.message));
}

// ---------- bucle ----------
let timer = null, inflight = false;
async function tick(now) {
  if (inflight) return;
  clearTimeout(timer);
  inflight = true;
  const ctl = new AbortController();
  const guard = setTimeout(() => ctl.abort(), FETCH_TIMEOUT_MS);
  try {
    const params = new URLSearchParams({ limit: "100" });
    if (filters.topics.size) params.set("topics", [...filters.topics].join(","));
    if (filters.q) params.set("q", filters.q);
    if (filters.correlation) params.set("correlation_id", filters.correlation);
    const res = await fetch(`/api/state?${params}`, { signal: ctl.signal, cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const s = await res.json();
    for (const fn of [renderPills, renderHealth, renderGraph, renderAlerts, renderFeed, renderSite, renderCamera]) {
      try { fn(s); } catch (err) { console.error(fn.name, err); }  // un panel roto no tumba a los demás
    }
  } catch (err) {
    const banner = document.getElementById("banner");
    banner.hidden = false;
    banner.textContent = "Sin conexión con el servidor del dashboard; reintentando. Se muestran los últimos datos.";
  } finally {
    clearTimeout(guard);
    inflight = false;
    timer = setTimeout(tick, POLL_MS);
  }
}

document.getElementById("feed-q").addEventListener("input", (ev) => {
  filters.q = ev.target.value.trim(); lastFeedSig = ""; tick(true);
});
document.getElementById("corr-clear").addEventListener("click", () => setCorrelation(""));
tick(true);
