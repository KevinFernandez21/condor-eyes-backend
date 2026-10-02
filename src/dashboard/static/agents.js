"use strict";
// Vista «Agentes»: salud, componentes, grafo por capas, alertas, feed y mapa de sitio.
const Agents = (() => {
  const { el, svg, dash, fmt, clock, uptime, STATE_TEXT } = UI;
  const filters = { topics: new Set(), q: "", correlation: "" };
  const openFeed = new Set();
  let lastFeedSig = "", lastTopicsSig = "", onFilterChange = () => {};

  // ---------- salud ----------
  const kpi = (value, label, cls) => el("div", { class: `kpi ${cls || ""}` }, el("b", {}, value), el("span", {}, label));
  function renderHealth(s) {
    const h = s.health;
    const status = { ok: ["OK", "ok"], degraded: ["Degradado", "warn"], offline: ["Sin conexión", "bad"] }[h.status] || [dash(h.status), ""];
    document.getElementById("health").replaceChildren(
      kpi(status[0], "estado del sistema", status[1]),
      kpi(h.agents_total == null ? "—" : `${h.agents_running}/${h.agents_total}`, "agentes en marcha"),
      kpi(dash(h.agents_failed), "caídos", h.agents_failed > 0 ? "bad" : ""),
      kpi(uptime(h.uptime_s), "activo desde la API"),
      kpi(dash(h.ws_clients), "clientes WS"),
      kpi(dash(h.ws_dropped_total), "descartes WS"),
      kpi("—", `CPU / memoria (${h.hardware})`));
    const box = document.getElementById("streams");
    if (!h.streams.length) { box.replaceChildren(el("div", { class: "small" }, "Sin streams reportados todavía.")); return; }
    box.replaceChildren(el("div", { class: "table-wrap" }, el("table", {},
      el("thead", {}, el("tr", {}, ...["Stream", "Estado"].map((t) => el("th", {}, t)),
        ...["FPS", "Latencia p50/p90 (ms)", "Frames perdidos"].map((t) => el("th", { class: "num" }, t)))),
      el("tbody", {}, h.streams.map((st) => el("tr", {},
        el("td", {}, st.stream_id), el("td", {}, dash(st.state)),
        el("td", { class: "num" }, fmt(st.fps)),
        el("td", { class: "num" }, `${fmt(st.latency_p50_ms)} / ${fmt(st.latency_p90_ms)}`),
        el("td", { class: "num" }, dash(st.frames_dropped))))))));
  }

  function renderComponents(s) {
    const box = document.getElementById("components");
    if (!s.components.length) { box.replaceChildren(el("span", { class: "small" }, "El sistema no reporta componentes.")); return; }
    box.replaceChildren(...s.components.map((c) => el("span", { class: "comp", title: c.detail ? `${c.name}: ${c.detail}` : c.name },
      el("i", { class: `dot ${c.state_class}` }), c.name, el("span", { class: "small" }, dash(c.state)),
      c.detail ? el("span", { class: "small" }, `· ${c.detail}`) : null)));
  }

  // ---------- grafo ----------
  const NODE_W = 148, NODE_H = 62, GX = 178, GY = 104, PAD = 20, LANE_H = NODE_H + 28;
  const LANES = [["Pipeline", 0], ["Control y salida", 1], ["Plugins", 2]];
  function clip(cx, cy, tx, ty) {  // punto del borde del nodo (centro cx,cy) en dirección a (tx,ty)
    const dx = tx - cx, dy = ty - cy;
    if (!dx && !dy) return [cx, cy];
    const k = Math.min(dx ? (NODE_W / 2 + 3) / Math.abs(dx) : Infinity, dy ? (NODE_H / 2 + 3) / Math.abs(dy) : Infinity);
    return [cx + dx * k, cy + dy * k];
  }
  function renderGraph(s) {
    const g = s.graph, root = document.getElementById("graph");
    const showControl = document.getElementById("show-control").checked;
    const maxCol = Math.max(...g.nodes.map((n) => n.col)), maxRow = Math.max(...g.nodes.map((n) => n.row), 2);
    const W = PAD * 2 + maxCol * GX + NODE_W, H = PAD * 2 + maxRow * GY + NODE_H;
    root.setAttribute("viewBox", `0 0 ${W} ${H}`);
    root.style.opacity = g.stale ? "0.5" : "1";
    const pos = Object.fromEntries(g.nodes.map((n) => [n.id, { x: PAD + n.col * GX, y: PAD + n.row * GY }]));
    const defs = svg("defs");
    const marker = svg("marker", { id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto" });
    marker.append(svg("path", { d: "M0 0L10 5L0 10z", fill: "currentColor" }));
    defs.append(marker);
    const lanes = svg("g");
    for (const [label, row] of LANES) {
      const y = PAD + row * GY - 14;
      lanes.append(svg("rect", { class: "lane", x: 4, y, width: W - 8, height: LANE_H, rx: 8, opacity: row === 1 ? 0.25 : 0.5 }));
      lanes.append(svg("text", { class: "lane-label", x: 12, y: y + 11 }, label));
    }
    const edgesLayer = svg("g", { style: "color: var(--accent)" });
    const labels = [];
    const pairs = {};
    for (const e of g.edges) {
      if (e.kind === "control" && !showControl) continue;
      const a = pos[e.source], b = pos[e.target];
      if (!a || !b) continue;
      const key = [e.source, e.target].sort().join("|");
      const idx = (pairs[key] = (pairs[key] || 0) + 1) - 1;
      const acx = a.x + NODE_W / 2, acy = a.y + NODE_H / 2, bcx = b.x + NODE_W / 2, bcy = b.y + NODE_H / 2;
      const [x1, y1] = clip(acx, acy, bcx, bcy), [x2, y2] = clip(bcx, bcy, acx, acy);
      const dir = e.source < e.target ? 1 : -1, bend = idx * 14 * dir;
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
      const cx = mx - (dy / len) * bend, cy = my + (dx / len) * bend;
      const hot = e.kind === "data" && e.rate_per_s > 0;
      const path = svg("path", { d: `M${x1} ${y1} Q${cx} ${cy} ${x2} ${y2}`, class: `edge ${e.kind}${hot ? " hot" : ""}`, "marker-end": "url(#arrow)" });
      path.append(svg("title", {}, `${e.label} (${e.source} → ${e.target})`));
      edgesLayer.append(path);
      if (e.kind === "data" && (e.rate_per_s > 0 || e.drops > 0)) {
        const short = `${e.rate_per_s == null ? "" : fmt(e.rate_per_s) + "/s"}${e.drops ? ` ⚠${e.drops}` : ""}`;
        labels.push(svg("text", { x: (x1 + 2 * cx + x2) / 4, y: (y1 + 2 * cy + y2) / 4 - 3, "text-anchor": "middle", class: "edge-label" }, short));
      }
    }
    const nodes = svg("g");
    for (const n of g.nodes) {
      const p = pos[n.id];
      const grp = svg("g", { class: `node ${n.state}`, transform: `translate(${p.x} ${p.y})`, tabindex: "0" });
      grp.append(svg("title", {}, `${n.label}: ${STATE_TEXT[n.state]} — ${n.detail}`));
      grp.append(svg("rect", { width: NODE_W, height: NODE_H }));
      grp.append(svg("circle", { class: "pdot", cx: 14, cy: 19, r: 4.5 }));
      grp.append(svg("text", { x: 26, y: 23 }, n.label + (n.instances > 1 ? ` ×${n.instances}` : "")));
      grp.append(svg("text", { x: 12, y: 40, class: "sub" }, STATE_TEXT[n.state] + (n.planned ? " · previsto" : "")));
      const stats = n.state === "unknown" ? "—" :
        `${n.rate_per_s == null ? "—" : fmt(n.rate_per_s) + " msg/s"} · cola ${dash(n.queue_depth)} · reinic. ${dash(n.restarts)}`;
      grp.append(svg("text", { x: 12, y: 54, class: "sub" }, stats));
      nodes.append(grp);
    }
    root.replaceChildren(defs, lanes, edgesLayer, nodes, ...labels);

    document.getElementById("topics-count").textContent = String(g.edges.length);
    document.querySelector("#edges tbody").replaceChildren(...g.edges.map((e) => el("tr", {},
      el("td", {}, e.topic), el("td", {}, `${e.source} → ${e.target}`),
      el("td", { class: "num" }, e.rate_per_s == null ? "—" : e.rate_per_s), el("td", { class: "num" }, e.drops))));
    document.getElementById("legend").replaceChildren(
      ...["ok", "degraded", "failed", "stopped", "unknown"].map((k) => el("span", {}, el("i", { class: `dot ${k}` }), STATE_TEXT[k])),
      el("span", {}, "Flechas: tópicos de datos · punteadas: control (salud, comandos, errores)"));
  }

  // ---------- alertas ----------
  function renderAlerts(s) {
    const box = document.getElementById("alerts");
    if (!s.alerts.length) { box.replaceChildren(el("div", { class: "small" }, "Sin decisiones de fusión todavía.")); return; }
    box.replaceChildren(...s.alerts.map((a) => el("article", { class: `alert ${a.severity}` },
      el("header", {}, el("span", { title: `valor original: ${dash(a.outcome_raw)}` }, a.outcome_label),
        el("span", { class: "small" }, `${clock(a.evaluated_at)} · ${dash(a.decision_id)}`)),
      el("div", {},
        el("span", { class: "tag op" }, "Requiere operador"),
        el("span", { class: "tag" }, `confianza ${a.confidence_pct == null ? "—" : a.confidence_pct + " %"}`),
        a.zone_id ? el("span", { class: "tag" }, `zona ${a.zone_id}`) : null,
        a.stream_id ? el("span", { class: "tag" }, a.stream_id) : null,
        a.person_id ? el("span", { class: "tag" }, `persona ${a.person_id}`) : null),
      a.reasons.length ? el("div", {}, a.reasons.map((r) => el("span", { class: "tag", title: r.code }, r.label))) : el("div", { class: "small" }, "Sin códigos de razón."),
      el("details", {}, el("summary", { class: "small" }, `Evidencia (${a.evidence.length})`),
        a.evidence.length
          ? el("ul", { class: "evidence" }, a.evidence.map((ev) => el("li", {},
              el("span", { class: "tag" }, ev.kind_label), el("span", { class: "tag" }, ev.role_label),
              el("span", { class: "small", title: JSON.stringify(ev) },
                `${dash(ev.evidence_id)}${ev.confidence == null ? "" : " · confianza " + Math.round(ev.confidence * 100) + " %"}${ev.detail ? " · " + ev.detail : ""}`))))
          : el("div", { class: "small" }, "Sin evidencia adjunta.")),
      a.correlation_id ? el("button", { class: "linkbtn", type: "button", onclick: () => setCorrelation(a.correlation_id) }, "ver mensajes relacionados") : null)));
  }

  // ---------- feed ----------
  function setCorrelation(id) { filters.correlation = id; lastFeedSig = ""; onFilterChange(); }
  function renderTopicChips(topics) {
    const sig = topics.join("|");
    if (sig === lastTopicsSig) return;
    lastTopicsSig = sig;
    document.getElementById("topic-chips").replaceChildren(...topics.map((t) =>
      el("button", { class: "chip", type: "button", "aria-pressed": String(filters.topics.has(t)), onclick: (ev) => {
        if (filters.topics.has(t)) filters.topics.delete(t); else filters.topics.add(t);
        ev.currentTarget.setAttribute("aria-pressed", String(filters.topics.has(t)));
        lastFeedSig = ""; onFilterChange();
      } }, t)));
  }
  function renderFeed(s) {
    renderTopicChips(s.feed_topics);
    const clear = document.getElementById("corr-clear");
    clear.hidden = !filters.correlation;
    clear.textContent = `Correlación: ${filters.correlation} ✕`;
    const sig = s.feed.map((e) => e.event_id).join(",");
    if (sig === lastFeedSig) return;
    lastFeedSig = sig;
    const list = document.getElementById("feed");
    if (!s.feed.length) { list.replaceChildren(el("li", { class: "small" }, "Sin mensajes con este filtro.")); return; }
    list.replaceChildren(...s.feed.map((e) => {
      const d = el("details", { open: openFeed.has(e.event_id) },
        el("summary", {}, el("span", { class: "t" }, clock(e.created_at)), el("span", { class: "topic" }, e.topic),
          el("span", { class: "small" }, dash(e.source)), el("span", { class: "sum" }, e.summary)),
        el("pre", {}, JSON.stringify(e.payload, (k, v) => (v === null ? "—" : v), 2)),
        el("div", { class: "small" }, `event_id ${dash(e.event_id)}`,
          e.stream_id ? ` · stream ${e.stream_id}` : "",
          e.correlation_id ? [" · correlación ", el("button", { class: "linkbtn", type: "button", onclick: () => setCorrelation(e.correlation_id) }, e.correlation_id)] : ""));
      d.addEventListener("toggle", () => { if (d.open) openFeed.add(e.event_id); else openFeed.delete(e.event_id); });
      return el("li", {}, d);
    }));
  }

  // ---------- mapa de sitio ----------
  function renderSite(s) {
    const box = document.getElementById("site");
    if (!s.site) { box.replaceChildren(el("div", { class: "small" }, "No hay mapa de sitio (configs/site_map.toml). Define CONDOR_SITE_MAP para cargar uno.")); return; }
    const out = s.site.zones.map((z) => el("div", { class: `zone${z.tags.length ? " has-tags" : ""}` },
      el("strong", {}, z.name), el("span", { class: "small" }, ` (${z.id})`),
      el("div", { class: "small" }, z.receiver ? `Receptor ${z.receiver.id} · ${dash(z.receiver.kind)}` : "Sin receptor",
        z.camera ? ` · cámara ${z.camera}` : "", z.calibrated ? "" : " · sin calibrar"),
      z.tags.length
        ? z.tags.map((t) => el("div", {}, el("span", { class: "tag" }, `dentro: ${t.tag_ref}`),
            el("span", { class: "small" }, `confianza ${t.confidence_pct == null ? "—" : t.confidence_pct + " %"} · visto ${clock(t.last_seen)}`)))
        : el("div", { class: "small" }, s.site.has_location_data ? "Nadie dentro." : "Presencia: sin datos")));
    if (s.site.outside.length) {
      out.push(el("div", { class: "zone" }, el("strong", {}, "Fuera / desconocido"),
        s.site.outside.map((t) => el("div", {}, el("span", { class: "tag" }, t.tag_ref),
          el("span", { class: "small" }, `${dash(t.unknown_reason) === "—" ? "fuera" : t.unknown_reason} · visto ${clock(t.last_seen)}`)))));
    }
    box.replaceChildren(...out);
  }

  function render(s) {
    for (const fn of [renderHealth, renderComponents, renderGraph, renderAlerts, renderFeed, renderSite]) {
      try { fn(s); } catch (err) { console.error(fn.name, err); }
    }
  }
  function init(onChange) {
    onFilterChange = onChange;
    document.getElementById("feed-q").addEventListener("input", (ev) => { filters.q = ev.target.value.trim(); lastFeedSig = ""; onChange(); });
    document.getElementById("corr-clear").addEventListener("click", () => setCorrelation(""));
    document.getElementById("show-control").addEventListener("change", () => onChange());
  }
  return { render, init, filters };
})();
