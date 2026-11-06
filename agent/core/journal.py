"""THE JOURNAL — the whole play of a level, remembered and compressed
(Tushar, 2026-10-14: "remember the entire play of the previous level
compressed and extract the highlight on what we actually did ... the
micromouse analogy: if we ever play twice we simply figure out the
shortest route; nothing is lost").

Every press is journaled (what was pressed, in which phase of the core's
work, what changed, the frame it led to).  At the attempt's end the
journal is compressed into the HIGHLIGHT:

  route        the presses after the last RESET, in order — the way the
               level was actually walked (the micromouse's first run)
  safe_prefix  the part of the route before the first divergence of the
               last plan from the world (where it died, or where the law
               lied): a way known to reach a state alive
  story        the reading pursued and the organ that acted — what the
               level was FOR, in the core's own words
  taught       the presses that changed something, by phase: where the
               level's lessons came from
  start_hash   the frame the attempt began on (a level seen again is known
               by it: the route is replayed, not rediscovered)

The highlight rides in the carry: a retry of the same level after a death
replays the safe prefix first (the shortest known route to where things
were still right), and the next level reads the story as a template.
"""
from collections import Counter


class Journal:
    def __init__(self, start_hash=None):
        self.start_hash = start_hash
        self.entries = []
        self.fatal = None            # (frame hash, key) of the press that lost the level, if it died

    def press(self, i, key, phase, events, frame_hash, reset=False):
        self.entries.append({"i": int(i), "key": key, "phase": (phase or "")[:60], "events": list(events or [])[:8],
                             "hash": frame_hash, "reset": bool(reset)})

    def highlight(self, how, ledger=None, story=None):
        ent = self.entries
        last_reset = max([j for j, e in enumerate(ent) if e["reset"]], default=-1)
        route = [e for e in ent[last_reset + 1:]]
        divs = (ledger or {}).get("divs") or []
        first_div = min([int(d.get("i_action") or 0) for d in divs if d.get("i_action")], default=None)
        if first_div is not None:
            safe = [e for e in route if e["i"] < first_div]
        else:
            # (a death with no divergence before it: every journaled press
            # held — the fatal press itself is never journaled — so the whole
            # route is the known way to the brink; the fatal pair is carried
            # beside it so the brink is not stepped over again)
            safe = list(route)
        phases = Counter(e["phase"].split(":")[0] for e in ent if e["events"])
        return {"start_hash": self.start_hash, "n_presses": len(ent), "route": [e["key"] for e in route],
                "safe_prefix": [e["key"] for e in safe], "safe_hashes": [e["hash"] for e in safe], "first_divergence": first_div,
                "story": story or (ledger or {}).get("reading"), "how": how, "fatal": self.fatal,
                "taught": dict(phases.most_common(6))}


def describe(h):
    """One block of text: what we actually did."""
    if not h:
        return ""
    keys = h.get("route") or []
    def _k(k):
        if isinstance(k, str):
            return k
        try:
            return f"{k[0]}{tuple(k[1])}" if isinstance(k[1], (tuple, list)) else f"{k[0]}({k[1]})"
        except Exception:
            return str(k)
    short = ", ".join(_k(k) for k in keys[:12]) + (" ..." if len(keys) > 12 else "")
    lines = [f"what we actually did: {h.get('n_presses', 0)} presses, {len(keys)} on the final route ({h.get('how')})",
             f"  route: {short}"]
    if h.get("story"):
        lines.append(f"  story: {h['story']}")
    if h.get("first_divergence") is not None:
        lines.append(f"  the world left the plan at press {h['first_divergence']}; {len(h.get('safe_prefix') or [])} presses before it are a known-good way")
    if h.get("taught"):
        lines.append("  where the changes came from: " + ", ".join(f"{p or 'start'} ({n})" for p, n in h["taught"].items()))
    return "\n".join(lines)
