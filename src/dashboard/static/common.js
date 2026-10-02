"use strict";
// Utilidades compartidas. Nunca se inserta HTML con datos del bus: todo entra
// como nodos de texto (textContent) y atributos.
const UI = (() => {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const STATE_TEXT = { ok: "en marcha", degraded: "degradado", failed: "caído", stopped: "detenido", unknown: "sin datos" };

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === false || v == null) continue;
      if (k === "class") node.className = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) {
      if (c == null || c === false) continue;
      node.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return node;
  }
  function svg(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    return node;
  }
  // Nunca se pinta "null"/"None": todo valor ausente es "—".
  function dash(v) {
    return v == null || v === "" || v === "null" || v === "None" || v === "undefined" ||
      (typeof v === "number" && Number.isNaN(v)) ? "—" : String(v);
  }
  const fmt = (v, d = 1) => (v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d));
  function clock(iso) {
    if (!iso) return "—";
    const d = typeof iso === "number" ? new Date(iso * 1000) : new Date(iso);
    return Number.isNaN(d.getTime()) ? "—" : d.toLocaleTimeString("es", { hour12: false });
  }
  function hhmm(epochS) {
    return new Date(epochS * 1000).toLocaleTimeString("es", { hour12: false, hour: "2-digit", minute: "2-digit" });
  }
  function uptime(s) {
    if (s == null) return "—";
    const m = Math.floor(s / 60);
    return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : m ? `${m} min ${Math.floor(s % 60)} s` : `${Math.floor(s)} s`;
  }
  // PRNG determinista (mulberry32) y hash de texto para escenas estables.
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function hash(text) {
    let h = 2166136261;
    for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
    return h >>> 0;
  }
  return { el, svg, dash, fmt, clock, hhmm, uptime, rng, hash, STATE_TEXT };
})();
