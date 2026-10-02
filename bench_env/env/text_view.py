"""Text observation of the simulator screen (for text-only models).

Selected with env var BENCH_OBS (set by ``run.py --obs``):
  ""/"screenshot"  default, nothing extracted (identical to the original benchmark)
  "view"           compact list of visible text + interactive elements with refs and
                   normalized (0-1000) boxes, read from the live DOM.

The view is built from the same page the screenshot comes from, so a text agent sees
exactly what a visual agent sees (visible viewport only, same app state).
"""
from __future__ import annotations

import json
import os
from typing import Any

MAX_ELEMENTS = int(os.environ.get("BENCH_OBS_MAX", "150"))


def obs_mode() -> str:
    m = os.environ.get("BENCH_OBS", "").strip().lower()
    return "" if m in ("", "screenshot") else m


_JS = r"""
(maxEl) => {
  const W = window.innerWidth, H = window.innerHeight;
  const norm = (r) => [Math.round(r.left / W * 1000), Math.round(r.top / H * 1000),
                       Math.round(r.right / W * 1000), Math.round(r.bottom / H * 1000)];
  const visible = (el, r) => {
    if (r.width < 2 || r.height < 2) return false;
    if (r.bottom <= 0 || r.right <= 0 || r.top >= H || r.left >= W) return false;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none' || parseFloat(cs.opacity) === 0) return false;
    return true;
  };
  const SEL = '[data-trigger],[data-action],button,a[href],input,textarea,select,[role=button],[role=tab],[role=switch],[role=checkbox],[onclick]';
  const clean = (s) => (s || '').replace(/\s+/g, ' ').trim().slice(0, 80);
  const label = (el) => clean(el.getAttribute('aria-label') || el.innerText || el.value ||
                              el.getAttribute('placeholder') || el.getAttribute('title') || '');
  const items = [];
  const seen = new Set();
  // interactive elements (skip ones fully covered by another element at their center)
  for (const el of document.querySelectorAll(SEL)) {
    const r = el.getBoundingClientRect();
    if (!visible(el, r)) continue;
    const cx = Math.min(W - 1, Math.max(0, (r.left + r.right) / 2));
    const cy = Math.min(H - 1, Math.max(0, (r.top + r.bottom) / 2));
    const top = document.elementFromPoint(cx, cy);
    if (!top || !(el === top || el.contains(top) || top.contains(el))) continue;
    const tag = el.tagName.toLowerCase();
    const type = el.getAttribute('type');
    const kind = (tag === 'input' || tag === 'textarea') ? 'input' :
                 tag === 'select' ? 'select' :
                 el.getAttribute('role') || (tag === 'a' ? 'link' : 'button');
    const key = kind + '|' + label(el) + '|' + norm(r).join(',');
    if (seen.has(key)) continue;
    seen.add(key);
    const it = {k: kind, t: label(el), b: norm(r), y: r.top, x: r.left};
    if (kind === 'input') { it.v = clean(el.value); if (type) it.type = type; }
    if (el.getAttribute('aria-checked') != null) it.checked = el.getAttribute('aria-checked');
    if (el.disabled) it.disabled = true;
    items.push(it);
  }
  // plain visible text not already inside an interactive element
  const texts = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    const s = clean(n.nodeValue);
    if (!s) continue;
    const p = n.parentElement;
    if (!p || ['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(p.tagName)) continue;
    if (p.closest(SEL)) continue;
    const range = document.createRange();
    range.selectNodeContents(n);
    const r = range.getBoundingClientRect();
    if (!visible(p, r)) continue;
    texts.push({k: 'text', t: s, b: norm(r), y: r.top, x: r.left});
  }
  const all = items.concat(texts).sort((a, b) => (Math.round(a.y / 8) - Math.round(b.y / 8)) || (a.x - b.x));
  return {viewport: [W, H], n_total: all.length, items: all.slice(0, maxEl).map(({y, x, ...rest}) => rest)};
}
"""


async def extract_view(page: Any) -> dict[str, Any]:
    """Return {"items": [...]} for the current page; refs (e1..) are added here."""
    data = await page.evaluate(_JS, MAX_ELEMENTS)
    n = 0
    for it in data["items"]:
        if it["k"] != "text":
            n += 1
            it["ref"] = f"e{n}"
    return data


def format_view(data: dict[str, Any]) -> str:
    """Compact text for the prompt. Interactive elements carry a ref and a box."""
    lines = []
    for it in data["items"]:
        if it["k"] == "text":
            lines.append(f'text "{it["t"]}"')
            continue
        extra = ""
        if "v" in it:
            extra += f' value="{it["v"]}"'
        if "checked" in it:
            extra += f' checked={it["checked"]}'
        if it.get("disabled"):
            extra += " disabled"
        lines.append(f'[{it["ref"]}] {it["k"]} "{it["t"]}"{extra} box={it["b"]}')
    if data.get("n_total", 0) > len(data["items"]):
        lines.append(f'... ({data["n_total"] - len(data["items"])} more elements not shown)')
    return "\n".join(lines)


def refs_to_points(data: dict[str, Any]) -> dict[str, list[int]]:
    """ref -> center point in the 0-1000 coordinate space used by the env."""
    out = {}
    for it in data["items"]:
        if "ref" in it:
            x1, y1, x2, y2 = it["b"]
            out[it["ref"]] = [(x1 + x2) // 2, (y1 + y2) // 2]
    return out


def view_to_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)
