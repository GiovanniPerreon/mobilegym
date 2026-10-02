"""Candidate action list for the "choice" action mode (Jev / Laya style).

Built the SAME way for every observation format, from the interactive elements that are
visible on the page, so across conditions only the observation format changes.
Each candidate maps to a normal MobileGym action (see the mapping table in the thesis report):

  CLICK        one per visible, unoccluded interactive element ("tap: Mamma")
  LONG_PRESS   only elements whose data-trigger-type contains "long"
  DOUBLE_TAP   only elements whose data-trigger-type contains "double"
  SWIPE        one per scrollable area and direction it can still scroll
  TYPE         one per input field ("type in: Message"); the text needs a 2nd generative call
  AWAKE        one per app in ``apps`` ("open app: wechat")
  ANSWER       "answer" (text needs a 2nd generative call)
  BACK/HOME/RECENT/ENTER/WAIT/COMPLETE/ABORT   fixed entries, always present

DRAG is excluded in this first version (hard to discretize), as in the report.
"""
from __future__ import annotations

from typing import Any, Sequence

from bench_env.env.text_view import _JS_COMMON, ref_point, swipe_points

_JS = "() => {" + _JS_COMMON + r"""
const out = {elements: [], scrollers: []};
for (const el of document.querySelectorAll(INTERACTIVE)) {
  const r = el.getBoundingClientRect();
  if (!inView(r)) continue;
  let ok = true;
  for (let e = el; e && e !== document.body; e = e.parentElement) if (!shown(e)) { ok = false; break; }
  if (!ok || !unoccluded(el, r)) continue;
  const tag = el.tagName.toLowerCase();
  const trig = el.getAttribute('data-trigger') || el.getAttribute('data-action') || '';
  out.elements.push({
    kind: (tag === 'input' || tag === 'textarea' || tag === 'select') ? 'input' : 'tap',
    label: label(el) || trig || tag,
    trig_type: (el.getAttribute('data-trigger-type') || el.getAttribute('data-action-type') || '').toLowerCase(),
    box: clampBox(r),
  });
}
for (const el of document.querySelectorAll('*')) {
  const cs = getComputedStyle(el);
  const sy = /(auto|scroll)/.test(cs.overflowY) && el.scrollHeight > el.clientHeight + 4;
  const sx = /(auto|scroll)/.test(cs.overflowX) && el.scrollWidth > el.clientWidth + 4;
  if (!sy && !sx) continue;
  const r = el.getBoundingClientRect();
  if (!inView(r) || !shown(el)) continue;
  out.scrollers.push({
    label: label(el).slice(0, 40) || el.getAttribute('data-testid') || el.className.toString().split(' ')[0] || el.tagName.toLowerCase(),
    box: clampBox(r),
    up: sy && el.scrollTop + el.clientHeight < el.scrollHeight - 4,      // more content below -> swipe up
    down: sy && el.scrollTop > 4,
    left: sx && el.scrollLeft + el.clientWidth < el.scrollWidth - 4,
    right: sx && el.scrollLeft > 4,
  });
}
return out;
}"""

_FIXED = [("BACK", "go back"), ("HOME", "go to home screen"), ("RECENT", "open recent apps"),
          ("ENTER", "press enter"), ("WAIT", "wait 2 seconds"), ("ANSWER", "answer the question"),
          ("COMPLETE", "task is complete"), ("ABORT", "give up")]


async def build_candidates(page: Any, apps: Sequence[str] = ()) -> list[dict[str, Any]]:
    raw = await page.evaluate(_JS)
    cands: list[dict[str, Any]] = []

    def add(action: str, label: str, **kw: Any) -> None:
        cands.append({"id": len(cands) + 1, "action": action, "label": label, **kw})

    for e in raw["elements"]:
        p = ref_point(e["box"])
        if e["kind"] == "input":
            add("TYPE", f'type in: {e["label"]}', point=p, needs_text=True)
            continue
        add("CLICK", f'tap: {e["label"]}', point=p, box=e["box"])
        if "long" in e["trig_type"]:
            add("LONG_PRESS", f'long press: {e["label"]}', point=p)
        if "double" in e["trig_type"]:
            add("DOUBLE_TAP", f'double tap: {e["label"]}', point=p)
    for s in raw["scrollers"]:
        for d in ("up", "down", "left", "right"):
            if s[d]:
                p1, p2 = swipe_points(s["box"], d)  # d is always a valid direction here
                add("SWIPE", f'swipe {d}: {s["label"]}', point1=p1, point2=p2)
    for a in apps:
        add("AWAKE", f"open app: {a}", value=a)
    for act, lab in _FIXED:
        add(act, lab, needs_text=(act == "ANSWER"), **({"value": 2} if act == "WAIT" else {}))
    return cands


def format_candidates(cands: list[dict[str, Any]]) -> str:
    return "\n".join(f'{c["id"]}. {c["label"]}' for c in cands)
