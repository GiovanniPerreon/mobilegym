"""Text observation formats for MobileGym (HTML, accessibility tree, view JSON).

Selected with env var BENCH_OBS (set by ``run.py --obs``):
  "" / "screenshot"  default: nothing extracted, benchmark unchanged
  "html"             pruned HTML of the visible elements
  "a11y"             browser accessibility tree (Chromium CDP), with refs
  "json"             view JSON: app, screen (route path), visible texts, transitions from data-trigger

Every format returns the same structure, stored in ``Observation.text_view``:
  {"format": str, "text": str, "refs": {ref: [x1, y1, x2, y2]}, "stats": {...}}
``refs`` maps each addressable element to its box in the 0-1000 coordinate space the
environment already uses, so a ref can always be turned into a normal coordinate action.
Only elements visible in the viewport are included (apps in the background stay in the
DOM with display:none and are excluded by construction).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

FORMATS = ("html", "a11y", "json")
# HTML budget. The server context is fixed at launch, so the observation must fit it:
#   budget = half of what is left after the output reserve (max_tokens 4096) and the system prompt (~1200);
#   the other half is kept for the history (the agent resends its previous answers).
# Override with BENCH_OBS_MAX_TOKENS. BENCH_CTX is set by the Docker entrypoint (the -c of llama-server).
OUTPUT_RESERVE_TOKENS = 4096
SYSTEM_RESERVE_TOKENS = 1200
HTML_CHARS_PER_TOKEN = float(os.environ.get("BENCH_OBS_CHARS_PER_TOKEN", "3.0"))   # conservative for HTML markup


def html_token_budget() -> int:
    v = os.environ.get("BENCH_OBS_MAX_TOKENS", "").strip()
    if v:
        return int(v)
    ctx = int(os.environ.get("BENCH_CTX", "16384") or 16384)
    return max(1000, (ctx - OUTPUT_RESERVE_TOKENS - SYSTEM_RESERVE_TOKENS) // 2)


MAX_HTML_CHARS = int(html_token_budget() * HTML_CHARS_PER_TOKEN)


def obs_mode() -> str:
    m = os.environ.get("BENCH_OBS", "").strip().lower()
    return m if m in FORMATS else ""


def obs_with_image() -> bool:
    """Hybrid formats: screenshot + text."""
    return os.environ.get("BENCH_OBS_IMAGE", "").strip() in ("1", "true", "yes")


# ---------------------------------------------------------------- shared JS helpers
_JS_COMMON = r"""
const W = window.innerWidth, H = window.innerHeight;
const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();
const clampBox = (r) => {
  const l = Math.max(0, r.left), t = Math.max(0, r.top), rr = Math.min(W, r.right), b = Math.min(H, r.bottom);
  return [Math.round(l / W * 1000), Math.round(t / H * 1000), Math.round(rr / W * 1000), Math.round(b / H * 1000)];
};
const inView = (r) => r.width >= 2 && r.height >= 2 && r.bottom > 0 && r.right > 0 && r.top < H && r.left < W;
const shown = (el) => {
  const cs = getComputedStyle(el);
  return cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) !== 0;
};
const INTERACTIVE = '[data-trigger],[data-action],button,a[href],input,textarea,select,[role=button],[role=tab],[role=switch],[role=checkbox],[role=menuitem],[onclick]';
const unoccluded = (el, r) => {
  const cx = Math.min(W - 1, Math.max(0, (r.left + r.right) / 2));
  const cy = Math.min(H - 1, Math.max(0, (r.top + r.bottom) / 2));
  const top = document.elementFromPoint(cx, cy);
  return !!top && (el === top || el.contains(top) || top.contains(el));
};
const label = (el) => clean(el.getAttribute('aria-label') || el.innerText || el.value ||
                            el.getAttribute('placeholder') || el.getAttribute('title') || '').slice(0, 80);
"""

# ---------------------------------------------------------------- HTML
_JS_HTML = "({maxChars, lvl}) => {" + _JS_COMMON + r"""
const KEEP_ALL = ['id','class','role','type','name','placeholder','value','href','alt','title','for','checked','disabled',
              'selected','data-trigger','data-trigger-type','data-trigger-params','data-action','data-action-type',
              'data-action-params'];
// lvl 0: everything (unchanged). lvl 1: no id/class (styling noise). lvl 2: also no trigger type/params, 'for';
// non-interactive icons/images without alt dropped. The interactive elements and their refs are never dropped.
const DROP1 = new Set(['id','class']);
const DROP2 = new Set(['id','class','for','data-trigger-type','data-trigger-params','data-action-type','data-action-params']);
const KEEP = KEEP_ALL.filter(k => lvl === 0 ? true : lvl === 1 ? !DROP1.has(k) : !DROP2.has(k));
const DROP_TAGS = new Set(['script','style','noscript','template','link','meta','head','title']);
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const refs = {};
let n = 0, nodes = 0;
function ser(node) {
  if (node.nodeType === 3) { const t = clean(node.nodeValue); return t ? esc(t) : ''; }
  if (node.nodeType !== 1) return '';
  const tag = node.tagName.toLowerCase();
  if (DROP_TAGS.has(tag)) return '';
  if (!shown(node)) return '';
  const r = node.getBoundingClientRect();
  if (r.width > 0 && r.height > 0 && !inView(r)) return '';   // fully outside the viewport
  if (tag === 'svg') return lvl >= 2 && !node.matches(INTERACTIVE) ? '' : '<svg/>';
  let attrs = '';
  for (const k of KEEP) {
    let v = node.getAttribute(k);
    if (v == null) continue;
    if (k === 'class') v = v.split(/\s+/).slice(0, 4).join(' ');
    if (v.length > 120) v = v.slice(0, 120) + '…';
    attrs += ` ${k}="${v.replace(/"/g, '&quot;')}"`;
  }
  if ((tag === 'input' || tag === 'textarea') && node.value && node.getAttribute('value') == null)
    attrs += ` value="${node.value.replace(/"/g, '&quot;').slice(0, 120)}"`;
  if (node.matches(INTERACTIVE) && inView(r) && unoccluded(node, r)) {
    n += 1; const ref = 'e' + n;
    refs[ref] = clampBox(r);
    attrs += ` data-ref="${ref}"`;
  }
  let inner = '';
  for (const c of node.childNodes) inner += ser(c);
  if (tag === 'img') return (lvl >= 2 && !node.getAttribute('alt') && !node.matches(INTERACTIVE)) ? '' : `<img${attrs}>`;
  if (!attrs && (tag === 'div' || tag === 'span')) return inner;   // pure wrappers add nothing
  if (!inner && !attrs) return '';
  nodes += 1;
  return `<${tag}${attrs}>${inner}</${tag}>`;
}
let html = ser(document.body);
return {text: html, refs, truncated: false, nodes, chars: html.length};
}"""

# ---------------------------------------------------------------- JSON of the view
_JS_JSON = "() => {" + _JS_COMMON + r"""
const texts = [];
const seen = new Set();
const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
let nd;
while ((nd = walker.nextNode())) {
  const s = clean(nd.nodeValue);
  if (!s) continue;
  const p = nd.parentElement;
  if (!p || ['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(p.tagName)) continue;
  const range = document.createRange(); range.selectNodeContents(nd);
  const r = range.getBoundingClientRect();
  if (!inView(r)) continue;
  let ok = true;
  for (let e = p; e && e !== document.body; e = e.parentElement) if (!shown(e)) { ok = false; break; }
  if (!ok) continue;
  texts.push({s: s.slice(0, 120), y: r.top, x: r.left});
}
texts.sort((a, b) => (Math.round(a.y / 8) - Math.round(b.y / 8)) || (a.x - b.x));
const visible_texts = [];
for (const t of texts) { if (!seen.has(t.s)) { seen.add(t.s); visible_texts.push(t.s); } }

const transitions = [], fields = [], refs = {};
let ti = 0, fi = 0;
for (const el of document.querySelectorAll('[data-trigger],[data-action],input,textarea,select')) {
  const r = el.getBoundingClientRect();
  if (!inView(r)) continue;
  let ok = true;
  for (let e = el; e && e !== document.body; e = e.parentElement) if (!shown(e)) { ok = false; break; }
  if (!ok || !unoccluded(el, r)) continue;
  const tag = el.tagName.toLowerCase();
  if (tag === 'input' || tag === 'textarea' || tag === 'select') {
    fi += 1; const ref = 'f' + fi; refs[ref] = clampBox(r);
    fields.push({ref, type: el.getAttribute('type') || tag, placeholder: el.getAttribute('placeholder') || null,
                 value: clean(el.value) || null});
    continue;
  }
  const name = el.getAttribute('data-trigger') || el.getAttribute('data-action');
  let params = {};
  try { params = JSON.parse(el.getAttribute('data-trigger-params') || el.getAttribute('data-action-params') || '{}'); } catch (e) {}
  ti += 1; const ref = 't' + ti; refs[ref] = clampBox(r);
  transitions.push({ref, name: name, params: params, text: label(el) || null});
}
return {visible_texts, transitions, fields, refs};
}"""


async def _html(page: Any) -> dict[str, Any]:
    """Pruned HTML that fits the token budget. Levels 0..2 drop styling noise first (see _JS_HTML);
    only if level 2 still exceeds the budget is the text cut, and refs not in the cut text are removed."""
    d: dict[str, Any] = {}
    lvl = 0
    for lvl in (0, 1, 2):
        d = await page.evaluate(_JS_HTML, {"maxChars": MAX_HTML_CHARS, "lvl": lvl})
        if len(d["text"]) <= MAX_HTML_CHARS:
            break
    truncated = len(d["text"]) > MAX_HTML_CHARS
    text, refs = d["text"], d["refs"]
    if truncated:
        cut = text[:MAX_HTML_CHARS]
        cut = cut[:cut.rfind("<")] if "<" in cut else cut          # do not leave a half tag
        text = cut + "<!-- truncated -->"
        kept = set(re.findall(r'data-ref="(e\d+)"', cut))
        refs = {k: v for k, v in refs.items() if k in kept}
    return {"text": text, "refs": refs,
            "stats": {"nodes": d["nodes"], "truncated": truncated, "level": lvl, "chars_full": d["chars"]}}


async def _json_view(page: Any, route: dict[str, Any]) -> dict[str, Any]:
    d = await page.evaluate(_JS_JSON)
    view = {
        "app": str(route.get("app") or ""),
        "screen": str(route.get("path") or ""),
        "visible_texts": d["visible_texts"],
        "transitions": d["transitions"],
    }
    if d["fields"]:
        view["input_fields"] = d["fields"]   # extra vs. the report example: needed to address TYPE
    return {"text": json.dumps(view, ensure_ascii=False, indent=1), "refs": d["refs"],
            "stats": {"transitions": len(d["transitions"]), "texts": len(d["visible_texts"]),
                      "fields": len(d["fields"])}}


# ---------------------------------------------------------------- accessibility tree (CDP)
_INTERACTIVE_ROLES = {"button", "link", "textbox", "searchbox", "combobox", "checkbox", "radio", "switch",
                      "tab", "menuitem", "menuitemcheckbox", "menuitemradio", "slider", "spinbutton",
                      "option", "treeitem", "listbox", "textarea", "togglebutton"}
_SKIP_ROLES = {"none", "presentation", "generic", "InlineTextBox", "LineBreak", "RootWebArea",
               "WebArea", "ignored", "Section", "Div"}
_SHOWN_PROPS = ("checked", "selected", "expanded", "disabled", "pressed", "focused", "required")


def _ax_val(x: Any) -> Any:
    return x.get("value") if isinstance(x, dict) else None


async def _a11y(page: Any) -> dict[str, Any]:
    vw, vh = await page.evaluate("[window.innerWidth, window.innerHeight]")
    cdp = await page.context.new_cdp_session(page)
    try:
        await cdp.send("DOM.enable")
        await cdp.send("Accessibility.enable")
        await cdp.send("DOM.getDocument", {"depth": 0})
        tree = await cdp.send("Accessibility.getFullAXTree")
        nodes = {n["nodeId"]: n for n in tree["nodes"]}

        # boxes for every node that could be printed
        async def box(n: dict) -> tuple[str, list | None]:
            bid = n.get("backendDOMNodeId")
            if bid is None:
                return n["nodeId"], None
            try:
                m = await cdp.send("DOM.getBoxModel", {"backendNodeId": bid})
                q = m["model"]["border"]
                xs, ys = q[0::2], q[1::2]
                return n["nodeId"], [min(xs), min(ys), max(xs), max(ys)]
            except Exception:
                return n["nodeId"], None

        cand = [n for n in tree["nodes"] if not n.get("ignored")
                and _ax_val(n.get("role")) not in _SKIP_ROLES]
        boxes = dict(await asyncio.gather(*(box(n) for n in cand)))
    finally:
        try:
            await cdp.detach()
        except Exception:
            pass

    def on_screen(b: list | None) -> bool:
        return b is not None and (b[2] - b[0]) >= 2 and (b[3] - b[1]) >= 2 \
            and b[2] > 0 and b[3] > 0 and b[0] < vw and b[1] < vh

    lines: list[str] = []
    refs: dict[str, list[int]] = {}
    counter = {"n": 0, "named": 0, "interactive": 0}

    def norm(b: list) -> list[int]:
        l, t, r, bt = max(0, b[0]), max(0, b[1]), min(vw, b[2]), min(vh, b[3])
        return [round(l / vw * 1000), round(t / vh * 1000), round(r / vw * 1000), round(bt / vh * 1000)]

    def walk(nid: str, depth: int, parent_name: str) -> None:
        n = nodes.get(nid)
        if n is None:
            return
        role = _ax_val(n.get("role")) or ""
        name = clean_name(_ax_val(n.get("name")))
        props = {p["name"]: _ax_val(p.get("value")) for p in n.get("properties", [])}
        children = n.get("childIds", [])
        if n.get("ignored") or role in _SKIP_ROLES:
            for c in children:
                walk(c, depth, parent_name)
            return
        b = boxes.get(nid)
        if not on_screen(b):
            return
        if role == "StaticText":
            if name and name not in parent_name:
                lines.append("  " * depth + f'- text "{name}"')
            return
        interactive = role in _INTERACTIVE_ROLES
        head = f"- {role}" + (f' "{name}"' if name else "")
        if interactive:
            counter["n"] += 1
            counter["interactive"] += 1
            if name:
                counter["named"] += 1
            ref = f"e{counter['n']}"
            refs[ref] = norm(b)
            head += f" [ref={ref}]"
        for k in _SHOWN_PROPS:
            if props.get(k) not in (None, False, "false"):
                head += f" [{k}]" if props[k] in (True, "true") else f" [{k}={props[k]}]"
        val = _ax_val(n.get("value"))
        if val not in (None, ""):
            head += f': "{val}"'
        lines.append("  " * depth + head)
        for c in children:
            walk(c, depth + 1, name)

    def clean_name(s: Any) -> str:
        return " ".join(str(s or "").split())[:100]

    roots = [n["nodeId"] for n in tree["nodes"] if "parentId" not in n]
    for r in roots:
        walk(r, 0, "")
    return {"text": "\n".join(lines), "refs": refs,
            "stats": {"interactive": counter["interactive"], "interactive_named": counter["named"]}}


# ---------------------------------------------------------------- public API
async def extract_view(page: Any, fmt: str, route: dict[str, Any] | None = None) -> dict[str, Any]:
    if fmt == "html":
        d = await _html(page)
    elif fmt == "a11y":
        d = await _a11y(page)
    elif fmt == "json":
        d = await _json_view(page, route or {})
    else:
        raise ValueError(f"unknown text format: {fmt}")
    d["format"] = fmt
    d["stats"]["chars"] = len(d["text"])
    d["stats"]["refs"] = len(d["refs"])
    return d


def ref_point(box: list[int]) -> list[int]:
    return [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]


def swipe_points(box: list[int], direction: str) -> tuple[list[int], list[int]] | None:
    """Finger path inside the element's box; direction = direction the finger moves."""
    cx, cy = ref_point(box)
    w, h = box[2] - box[0], box[3] - box[1]
    dx, dy = int(w * 0.35), int(h * 0.35)
    d = (direction or "").strip().lower()
    table = {"up": ([cx, cy + dy], [cx, cy - dy]), "down": ([cx, cy - dy], [cx, cy + dy]),
             "left": ([cx + dx, cy], [cx - dx, cy]), "right": ([cx - dx, cy], [cx + dx, cy])}
    return table.get(d)
